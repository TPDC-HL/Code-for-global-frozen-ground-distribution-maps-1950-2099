#!/usr/bin/env python3
"""Compute per-GCM daily KBDI from NEX-GDDP-CMIP6, followed by annual outputs.

Input NetCDF: INPUT/{scenario}/{variable}_{model}_{year}.nc
  scenario: historical / ssp126 / ssp245 / ssp370 / ssp585
  variables: tasmax (normally K), pr (normally kg m-2 s-1)
For each GCM, compute historical annual rain climatology, spin KBDI up
through 2014, branch the final historical state to each SSP (2015–2099),
and output annual mean, warm-season mean, annual max and year-end KBDI.
Do NOT use FI/TI here; those are only for the frost-number model.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmip6_config import MODELS, SCENARIOS
from kbdi.kbdi_core import initialize, daily_step, KBDIState
from kbdi.cmip6_io import open_pair, open_precip

def _grid_check(lat,lon,da):
    if lat is not None and not (np.array_equal(lat,da.lat.values) and np.array_equal(lon,da.lon.values)):
        raise ValueError("Grid mismatch across years or scenarios; regridding required")

def rainfall_climatology(root,pattern,model,start,end,precip_units=None):
    """Mean historical annual precipitation using only completely observed years."""
    total=None
    lat=lon=None
    years=0
    for year in range(start,end+1):
        with open_precip(root,pattern,"historical",model,year,precip_units) as p:
            _grid_check(lat,lon,p)
            if lat is None:
                lat=p.lat.values.copy()
                lon=p.lon.values.copy()
            annual=np.zeros((len(lat),len(lon)),dtype=float)
            count=np.zeros_like(annual,dtype=np.uint16)
            for day in range(p.sizes["time"]):
                v=np.asarray(p.isel(time=day).values,dtype=float)
                if np.any(np.isfinite(v)&(v<0)):
                    raise ValueError(f"Negative precipitation: {model}, {year}")
                ok=np.isfinite(v)
                annual+=np.where(ok,v,0.0)
                count+=ok.astype(np.uint16)
            annual=np.where(count==p.sizes["time"],annual,np.nan)
            total=annual if total is None else total+annual
            years+=1
    clim=total/years
    if not np.any(np.isfinite(clim)&(clim>0)):
        raise ValueError(f"No valid precipitation climatology for {model}")
    return clim,lat,lon

def run_year(root,pattern,scenario,model,year,state,rain_clim,lat,lon,
             output_root=None,rainfall_mode="wet_spell",warm_months=(5,6,7,8,9),
             temp_units=None,precip_units=None,write_daily=False):
    with open_pair(root,pattern,scenario,model,year,temp_units,precip_units) as (tasmax,precip):
        _grid_check(lat,lon,tasmax)
        s=np.zeros_like(rain_clim)
        n=np.zeros(rain_clim.shape,dtype=np.uint16)
        sw=np.zeros_like(rain_clim)
        nw=np.zeros_like(n)
        mx=np.full_like(rain_clim,-np.inf)
        daily=[]
        months=np.asarray(tasmax.time.dt.month.values)
        for ix,month in enumerate(months):
            state,q=daily_step(state,tasmax.isel(time=ix).values,precip.isel(time=ix).values,
                               rain_clim,rainfall_mode)
            ok=np.isfinite(q)
            s+=np.where(ok,q,0)
            n+=ok.astype(np.uint16)
            mx=np.maximum(mx,np.where(ok,q,-np.inf))
            if month in warm_months:
                sw+=np.where(ok,q,0)
                nw+=ok.astype(np.uint16)
            if write_daily:
                daily.append(q.astype("float32"))
        if output_root is not None:
            ds=xr.Dataset({
                "kbdi_annual_mean":(("lat","lon"),np.where(n>0,s/np.maximum(n,1),np.nan).astype("float32")),
                "kbdi_warm_season_mean":(("lat","lon"),np.where(nw>0,sw/np.maximum(nw,1),np.nan).astype("float32")),
                "kbdi_annual_max":(("lat","lon"),np.where(n>0,mx,np.nan).astype("float32")),
                "kbdi_year_end":(("lat","lon"),np.where(n>0,state.index,np.nan).astype("float32")),
                "n_valid_days":(("lat","lon"),n),
                "annual_rain_climatology_mm":(("lat","lon"),rain_clim.astype("float32"))
            },coords={"lat":lat,"lon":lon})
            ds.attrs.update(gcm=model,experiment=scenario,year=year,
                rainfall_interception=rainfall_mode,warm_months=",".join(map(str,warm_months)),
                initial_condition="historical spin-up; continuous day-to-day state across years",
                source="reference KBDI implementation, not verified original manuscript production code")
            path=Path(output_root)/scenario/f"KBDI_{model}_{scenario}_{year}.nc"
            path.parent.mkdir(parents=True,exist_ok=True)
            ds.to_netcdf(path)
            if write_daily:
                full=xr.Dataset({"kbdi":(("time","lat","lon"),np.stack(daily))},
                    coords={"time":tasmax.time.values,"lat":lat,"lon":lon})
                full.attrs.update(ds.attrs)
                full.to_netcdf(path.with_name(path.stem+"_daily.nc"))
        return state

def simulate_model(args,model):
    clim,lat,lon=rainfall_climatology(args.input_root,args.file_template,model,
          args.baseline_start,args.baseline_end,args.precip_units)
    state=initialize(clim.shape,args.initial_kbdi)
    for year in range(args.historical_start,args.historical_end+1):
        state=run_year(args.input_root,args.file_template,"historical",model,year,
          state,clim,lat,lon,output_root=args.output_root if args.write_historical else None,
          rainfall_mode=args.rainfall_mode,warm_months=args.warm_months,
          temp_units=args.temp_units,precip_units=args.precip_units,
          write_daily=args.output_daily and args.write_historical)
        if args.verbose:print("historical",model,year,flush=True)
    base=KBDIState(state.index.copy(),state.wet_spell_total_mm.copy())
    for scenario in args.scenarios:
        state=KBDIState(base.index.copy(),base.wet_spell_total_mm.copy())
        for year in range(args.historical_end+1,args.end_year+1):
            state=run_year(args.input_root,args.file_template,scenario,model,year,
              state,clim,lat,lon,output_root=args.output_root if year>=args.start_year else None,
              rainfall_mode=args.rainfall_mode,warm_months=args.warm_months,
              temp_units=args.temp_units,precip_units=args.precip_units,
              write_daily=args.output_daily and year>=args.start_year)
            if args.verbose:print(scenario,model,year,flush=True)

def get_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",required=True,type=Path)
    p.add_argument("--output-root",required=True,type=Path)
    p.add_argument("--file-template",default="{scenario}/{variable}_{model}_{year}.nc")
    p.add_argument("--models",nargs="+",default=MODELS)
    p.add_argument("--scenarios",nargs="+",default=SCENARIOS)
    p.add_argument("--historical-start",type=int,default=1950)
    p.add_argument("--historical-end",type=int,default=2014)
    p.add_argument("--baseline-start",type=int,default=1981)
    p.add_argument("--baseline-end",type=int,default=2014)
    p.add_argument("--start-year",type=int,default=2015)
    p.add_argument("--end-year",type=int,default=2099)
    p.add_argument("--initial-kbdi",type=float,default=0.)
    p.add_argument("--rainfall-mode",choices=["wet_spell","daily"],default="wet_spell")
    p.add_argument("--warm-months",type=int,nargs="+",default=[5,6,7,8,9])
    p.add_argument("--temp-units",choices=["K","C"])
    p.add_argument("--precip-units",choices=["flux","mm-day"])
    p.add_argument("--write-historical",action="store_true")
    p.add_argument("--output-daily",action="store_true")
    p.add_argument("--allow-missing-models",action="store_true")
    p.add_argument("--verbose",action="store_true")
    args=p.parse_args(argv)
    if not (args.historical_start<=args.baseline_start<=args.baseline_end<=args.historical_end<args.start_year<=args.end_year):
        raise ValueError("Invalid historical/baseline/projection years")
    if any(m<1 or m>12 for m in args.warm_months):
        raise ValueError("warm-months must be 1..12")
    return args

def main(argv=None):
    args=get_args(argv)
    args.output_root.mkdir(parents=True,exist_ok=True)
    failures=[]
    for model in args.models:
        try:
            simulate_model(args,model)
            print("DONE",model,flush=True)
        except Exception as exc:
            failures.append((model,str(exc)))
            print("ERROR",model,exc,flush=True)
            if not args.allow_missing_models:raise
    with (args.output_root/"run_models.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["model","status","message"])
        for model in args.models:
            error=next((e for m,e in failures if m==model),"")
            writer.writerow([model,"FAILED" if error else "SUCCESS",error])
    if failures:print("Incomplete models:",failures)
if __name__=="__main__":
    main()
