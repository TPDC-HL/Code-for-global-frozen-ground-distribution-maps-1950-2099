#!/usr/bin/env python3
"""Reproduce reported negative-binomial statistics and circular-shift P values.

This script intentionally excludes exploratory SEA and partial correlation.
Use authorized original monitoring data; do not publish private raw data.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import norm
from wildfire_ch4_analysis import read_rows,design,fit_nb,permute_methane_nb

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--outdir",type=Path,default=Path("results/reproduced"))
    parser.add_argument("--permutations",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=20261109)
    args=parser.parse_args()
    if args.permutations<1:raise ValueError("permutations must be positive")
    args.outdir.mkdir(parents=True,exist_ok=True)
    items,_=read_rows(args.input)
    rows=[]
    for name,column in (("R-1","ch4_r1"),("R-2","ch4_r2")):
        data=design(items,column,use_wind=False)
        result,alpha=fit_nb(data["y"],data["X"])
        beta=float(result.params[1])
        se=float(result.bse[1])
        p_wald=float(2*norm.sf(abs(beta/se)))
        if name=="R-1":
            p_shift,n_perms=permute_methane_nb(
                data,alpha,beta,args.permutations,np.random.default_rng(args.seed))
        else:
            p_shift,n_perms="",0
        rows.append({"sensor":name,"n_days":len(data["y"]),"beta_ln1pCH4":beta,
                     "robust_se":se,"wald_p_two_sided":p_wald,
                     "temporal_shift_p":p_shift,"valid_permutations":n_perms})
    out=args.outdir/"manuscript_methane_fire_results.csv"
    with out.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:print(row)
    print("Saved",out)
if __name__=="__main__":main()
