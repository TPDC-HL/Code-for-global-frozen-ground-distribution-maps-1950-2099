"""Reproducible Frost-number classification from annual FI/TI NetCDF.

Thresholds and class IDs reproduce the author-supplied original:
TP mask == 1: PF>=0.55, SFG>=0.01, otherwise UFG; IDs 3,2,0.
Outside TP: PF>=0.53, SFG>=0.10, IFG>=0.01, UFG<0.01; IDs 3,2,1,0.
Invalid FI/TI or FI=TI=0 gives -9999. Ensemble class uses existing FI/TI
ensemble mean INPUT FILES (not the arithmetic mean of categorical maps).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cmip6_config import MODELS,SCENARIOS

def frost_number(fi,ti):
    fi=np.asarray(fi,dtype=float)
    ti=np.asarray(ti,dtype=float)
    if np.any(np.isfinite(fi)&(fi<0)) or np.any(np.isfinite(ti)&(ti<0)):
        raise ValueError("FI and TI must be nonnegative degree-day totals")
    denom=np.sqrt(fi)+np.sqrt(ti)
    with np.errstate(invalid="ignore",divide="ignore"):
        f=np.sqrt(fi)/denom
    return np.where(denom>0,f,np.nan)

def classify(fi,ti,tp_mask):
    f=frost_number(fi,ti)
    mask=np.asarray(tp_mask)==1
    if f.shape!=mask.shape:
        raise ValueError("TP mask must align with FI/TI grid")
    out=np.full(f.shape,-9999,dtype=np.int16)
    valid=np.isfinite(f)&(f>=0)&(f<=1)
    out[valid&mask&(f<0.01)]=0
    out[valid&mask&(f>=0.01)&(f<0.55)]=2
    out[valid&mask&(f>=0.55)]=3
    out[valid&~mask&(f<0.01)]=0
    out[valid&~mask&(f>=0.01)&(f<0.10)]=1
    out[valid&~mask&(f>=0.10)&(f<0.53)]=2
    out[valid&~mask&(f>=0.53)]=3
    return out

def classify_files(fi_path,ti_path,mask_path,out_path):
    with xr.open_dataset(fi_path) as a,xr.open_dataset(ti_path) as b,xr.open_dataset(mask_path) as mm:
        fi=a["FI"]
        ti=b["TI"]
        mask=mm["elevation"]
        if any(v.dims!=("lat","lon") for v in (fi,ti,mask)):
            raise ValueError("FI, TI and elevation must have dims (lat,lon)")
        if not (np.array_equal(fi.lat.values,ti.lat.values) and
                np.array_equal(fi.lon.values,ti.lon.values) and
                np.array_equal(fi.lat.values,mask.lat.values) and
                np.array_equal(fi.lon.values,mask.lon.values)):
            raise ValueError("FI/TI/mask coordinates mismatch; explicit regridding required")
        result=classify(fi.values,ti.values,mask.values)
        out=xr.Dataset({"frozen":(("lat","lon"),result)},
                       coords={"lat":fi.lat.values,"lon":fi.lon.values})
        out["frozen"].attrs.update(
            long_name="frozen-ground class by the original author thresholds",
            _FillValue=-9999,
            flag_values=np.array([0,1,2,3],dtype=np.int16),
            flag_meanings="unfrozen intermittent seasonal permafrost")
        Path(out_path).parent.mkdir(parents=True,exist_ok=True)
        out.to_netcdf(out_path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--mask",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--models",nargs="+",default=MODELS)
    p.add_argument("--scenarios",nargs="+",default=SCENARIOS)
    p.add_argument("--start-year",type=int,default=1950)
    p.add_argument("--end-year",type=int,default=2099)
    p.add_argument("--strict",action="store_true")
    args=p.parse_args()
    for scenario in args.scenarios:
        folder=args.input_root/scenario.upper()
        for year in range(args.start_year,args.end_year+1):
            for model in args.models:
                fi=folder/f"FI_{model}_{year}.nc"
                ti=folder/f"TI_{model}_{year}.nc"
                if not (fi.exists() and ti.exists()):
                    if args.strict:raise FileNotFoundError(f"Missing {fi} or {ti}")
                    print("SKIP",scenario,model,year)
                    continue
                output=args.output_root/f"Frozen_{scenario.upper()}"/f"{model}_Frozen_{scenario}_{year}.nc"
                classify_files(fi,ti,args.mask,output)
            fi=folder/f"FI_ensemble_mean_{year}.nc"
            ti=folder/f"TI_ensemble_mean_{year}.nc"
            if fi.exists() and ti.exists():
                output=args.output_root/f"Frozen_{scenario.upper()}"/f"Ensemble_Mean_Frozen_{scenario}_{year}.nc"
                classify_files(fi,ti,args.mask,output)
            elif args.strict:
                raise FileNotFoundError(f"Missing FI/TI ensemble: {fi} / {ti}")
    print("Finished annual FI/TI classification")
if __name__=="__main__":
    main()
