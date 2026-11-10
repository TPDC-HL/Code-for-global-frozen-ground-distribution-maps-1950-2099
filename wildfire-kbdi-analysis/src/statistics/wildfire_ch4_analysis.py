#!/usr/bin/env python3
"""March-April daily methane–fire association, restricted to analyses in Results.

Negative-binomial regression with temperature, relative humidity, year fixed
effects and cubic within-season trend. Within-year residual circular shifts for
empirical P values. No unreported SEA / partial-correlation analyses are run.
Input requires field monitoring workbook with original Chinese column names.
"""
from __future__ import annotations
import argparse
import csv
import math
from datetime import date,datetime,timedelta
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from xlsx_reader import read_xlsx_grid

INPUT_COLUMNS={
    "date":"日期","fire":"火点个数","temperature":"温度",
    "humidity":"湿度","ch4_r1":"CH4-1(R-1)","ch4_r2":"CH4-1(R-2)"
}

def as_date(value):
    if isinstance(value,datetime):return value.date()
    if isinstance(value,date):return value
    if isinstance(value,(int,float)):
        return (datetime(1899,12,30)+timedelta(days=float(value))).date()
    for fmt in ("%Y-%m-%d","%Y/%m/%d","%Y.%m.%d"):
        try:return datetime.strptime(str(value).strip()[:10],fmt).date()
        except ValueError:pass
    raise ValueError(f"Unrecognized date: {value!r}")

def number(value):
    if value is None or str(value).strip() in ("","--","NA","nan","None"):
        return float("nan")
    try:return float(value)
    except (ValueError,TypeError):return float("nan")

def read_rows(xlsx,wind_csv=None):
    grid=read_xlsx_grid(xlsx,sheet=0)
    columns=[str(h).strip() if h is not None else "" for h in grid[0]]
    idx={h:i for i,h in enumerate(columns)}
    missing=[c for c in INPUT_COLUMNS.values() if c not in idx]
    if missing:raise ValueError(f"Missing required columns: {missing}")
    if wind_csv is not None:
        raise ValueError("Wind is not available in the reported analysis; do not add unverified covariates")
    rows=[]
    for row in grid[1:]:
        if len(row)<len(columns) or row[idx["日期"]] is None:continue
        d=as_date(row[idx["日期"]])
        if d.year not in (2017,2018,2019) or d.month not in (3,4):continue
        vals={k:number(row[idx[h]]) for k,h in INPUT_COLUMNS.items() if k!="date"}
        vals["date"]=d
        vals["wind"]=float("nan")
        rows.append(vals)
    rows.sort(key=lambda r:r["date"])
    if len(set(r["date"] for r in rows))!=len(rows):
        raise ValueError("Duplicate dates in monitoring data")
    return rows,False

def design(items,ch4_col,use_wind=False,exclude_capped=False):
    if use_wind:raise ValueError("Wind-adjusted model is not included in the reported results")
    needed=("fire","temperature","humidity",ch4_col)
    selected=[r for r in items if all(np.isfinite(r[k]) for k in needed)
              and (not exclude_capped or r[ch4_col]<1000)]
    if len(selected)<50:
        raise ValueError("Insufficient complete daily records")
    dates=[r["date"] for r in selected]
    y=np.asarray([r["fire"] for r in selected],float)
    methane=np.asarray([r[ch4_col] for r in selected],float)
    if np.any(y<0) or not np.all(y==np.floor(y)):
        raise ValueError("Fire counts must be nonnegative integers")
    if np.any(methane<0):raise ValueError("Methane must be nonnegative")
    temp=np.asarray([r["temperature"] for r in selected],float)
    humidity=np.asarray([r["humidity"] for r in selected],float)
    years=np.asarray([d.year for d in dates])
    days=np.asarray([(d-date(d.year,3,1)).days for d in dates],float)
    x=(days-30)/30
    Z=np.column_stack([np.ones(len(selected)),temp,humidity/10,
           (years==2018).astype(float),(years==2019).astype(float),x,x*x,x*x*x])
    X=np.column_stack([Z[:,:1],np.log1p(methane),Z[:,1:]])
    return {"rows":selected,"dates":dates,"y":y,"methane":methane,"Z":Z,
            "X":X,"years":years,"ch4_col":ch4_col,"use_wind":False}

def fit_nb(y,X,alpha=None,robust=True):
    if alpha is None:
        def loss(log_alpha):
            try:
                fit=sm.GLM(y,X,family=sm.families.NegativeBinomial(alpha=np.exp(log_alpha))).fit(maxiter=200)
                return -float(fit.llf) if np.isfinite(fit.llf) else 1e30
            except Exception:return 1e30
        opt=minimize_scalar(loss,method="bounded",bounds=(math.log(0.1),math.log(1000)),
                            options={"xatol":0.005})
        if not opt.success:raise RuntimeError("Negative-binomial dispersion fit failed")
        alpha=float(np.exp(opt.x))
    fit=sm.GLM(y,X,family=sm.families.NegativeBinomial(alpha=alpha)).fit(
        maxiter=150,**({"cov_type":"HC0"} if robust else {}))
    return fit,alpha

def permute_methane_nb(dat,alpha,observed_beta,B,rng):
    """Residual-based circular shift within each year; preserves spring season."""
    Z,X=dat["Z"],dat["X"]
    covariate_fit=sm.OLS(np.log1p(dat["methane"]),Z).fit()
    pred,resid=covariate_fit.fittedvalues,covariate_fit.resid
    years=dat["years"]
    groups=[np.where(years==year)[0] for year in sorted(set(years))]
    betas=[]
    for _ in range(B):
        shifted=resid.copy()
        for ix in groups:
            shifted[ix]=np.roll(resid[ix],rng.integers(1,len(ix)))
        Xp=X.copy()
        Xp[:,1]=pred+shifted
        try:
            result,_=fit_nb(dat["y"],Xp,alpha=alpha,robust=False)
            if np.isfinite(result.params[1]):betas.append(float(result.params[1]))
        except Exception:pass
    p=(1+sum(abs(b)>=abs(observed_beta) for b in betas))/(1+len(betas))
    return float(p),len(betas)
