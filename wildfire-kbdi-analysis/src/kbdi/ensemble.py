"""Equal-weight multimodel ensemble of completed per-GCM KBDI results.

KBDI(mean daily weather) is NOT equal to mean(KBDI(daily weather)).
Use only separately calculated per-GCM outputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cmip6_config import MODELS,SCENARIOS

VARS=("kbdi_annual_mean","kbdi_warm_season_mean","kbdi_annual_max","kbdi_year_end")

def aggregate_files(paths,output,min_models=25,model_names=None):
    paths=[Path(p) for p in paths if Path(p).is_file()]
    if len(paths)<min_models:
        raise ValueError(f"Only {len(paths)} model outputs; min_models={min_models}")
    if model_names is not None and len(model_names)!=len(paths):
        raise ValueError("Model labels do not match file count")
    sums={};sums2={};counts={}
    lat=lon=None
    for path in paths:
        with xr.open_dataset(path) as ds:
            if lat is None:
                lat=ds.lat.values.copy()
                lon=ds.lon.values.copy()
            if not (np.array_equal(ds.lat.values,lat) and np.array_equal(ds.lon.values,lon)):
                raise ValueError(f"Grid mismatch: {path}")
            for var in VARS:
                x=ds[var].values.astype(float)
                valid=np.isfinite(x)
                if var not in sums:
                    sums[var]=np.zeros_like(x)
                    sums2[var]=np.zeros_like(x)
                    counts[var]=np.zeros(x.shape,dtype=np.uint16)
                sums[var]+=np.where(valid,x,0.0)
                sums2[var]+=np.where(valid,x*x,0.0)
                counts[var]+=valid.astype(np.uint16)
    out=xr.Dataset(coords={"lat":lat,"lon":lon})
    for var in VARS:
        n=counts[var].astype(float)
        with np.errstate(invalid="ignore",divide="ignore"):
            mean=np.where(n>=min_models,sums[var]/n,np.nan)
            var_sample=np.where(n>=max(2,min_models),
                (sums2[var]-sums[var]**2/n)/(n-1),np.nan)
            std=np.sqrt(np.maximum(0,var_sample))
        out[var+"_mme_mean"]=(("lat","lon"),mean.astype("float32"))
        out[var+"_intermodel_sd"]=(("lat","lon"),std.astype("float32"))
        out[var+"_n_gcms"]=(("lat","lon"),counts[var])
    out.attrs.update(
        ensemble="equal-weight mean of individual GCM KBDI",
        source_models=",".join(model_names or [p.stem for p in paths]),
        min_models_per_cell=min_models,
        caution="intermodel SD is not a 90 percent confidence interval")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    out.to_netcdf(output)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-output-root",required=True,type=Path)
    p.add_argument("--output-root",required=True,type=Path)
    p.add_argument("--models",nargs="+",default=MODELS)
    p.add_argument("--scenarios",nargs="+",default=SCENARIOS)
    p.add_argument("--start-year",type=int,default=2015)
    p.add_argument("--end-year",type=int,default=2099)
    p.add_argument("--min-models",type=int,default=25)
    a=p.parse_args()
    if a.min_models>len(a.models):
        raise ValueError("Minimum model number exceeds model list")
    for scenario in a.scenarios:
        for year in range(a.start_year,a.end_year+1):
            by_model=[(model,a.model_output_root/scenario/f"KBDI_{model}_{scenario}_{year}.nc")
                      for model in a.models]
            available=[(m,path) for m,path in by_model if path.is_file()]
            output=a.output_root/scenario/f"MME_KBDI_{scenario}_{year}.nc"
            aggregate_files([f for _,f in available],output,a.min_models,[m for m,_ in available])
            print("Wrote",output,len(available),"models")
if __name__=="__main__":
    main()
