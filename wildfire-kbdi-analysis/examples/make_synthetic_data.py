#!/usr/bin/env python3
"""Create TINY SYNTHETIC NetCDF files for testing; NEVER research inputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

def write_tiny(root,models=("MODEL-A","MODEL-B"),scenarios=("ssp126","ssp245"),
               historical_years=(2000,2001),future_years=(2002,)):
    root=Path(root)
    lat=np.array([35.0,55.0])
    lon=np.array([90.0,110.0,120.0])
    grid=np.ones((2,3),dtype=float)
    for scen in ("historical",*scenarios):
        years=historical_years if scen=="historical" else future_years
        for year in years:
            date=pd.date_range(f"{year}-01-01",f"{year}-12-31",freq="D")
            phase=2*np.pi*(date.dayofyear.values-1)/365.25
            for ix,model in enumerate(models):
                t=(273.15+9+16*np.sin(phase)[:,None,None]+ix*0.8+
                   (3 if scen!="historical" else 0)+np.zeros((len(date),2,3)))
                p=np.broadcast_to(np.where(date.dayofyear.values[:,None,None]%7==0,
                    10.0,1.5)*grid[None,:,:]*(1+ix*.1),(len(date),2,3)).copy()
                attrs={"synthetic":"true: generated for testing ONLY"}
                folder=root/scen
                folder.mkdir(parents=True,exist_ok=True)
                ds1=xr.Dataset({"tasmax":(("time","lat","lon"),t.astype("float32"),{"units":"K"})},
                    coords={"time":date,"lat":lat,"lon":lon},attrs=attrs)
                ds2=xr.Dataset({"pr":(("time","lat","lon"),(p/86400).astype("float32"),
                    {"units":"kg m-2 s-1"})},coords={"time":date,"lat":lat,"lon":lon},attrs=attrs)
                ds1.to_netcdf(folder/f"tasmax_{model}_{year}.nc")
                ds2.to_netcdf(folder/f"pr_{model}_{year}.nc")
    with xr.Dataset({"elevation":(("lat","lon"),np.array([[1,1,0],[0,0,1]],dtype=np.int16))},
                    coords={"lat":lat,"lon":lon}) as mask:
        mask.to_netcdf(root/"GTOPOP_v4.nc")
    folder=root/"SSP126"
    folder.mkdir(exist_ok=True)
    for model in models:
        for year in future_years:
            fi=xr.Dataset({"FI":(("lat","lon"),np.full((2,3),200.))},coords={"lat":lat,"lon":lon})
            ti=xr.Dataset({"TI":(("lat","lon"),np.full((2,3),150.))},coords={"lat":lat,"lon":lon})
            fi.to_netcdf(folder/f"FI_{model}_{year}.nc")
            ti.to_netcdf(folder/f"TI_{model}_{year}.nc")
    return root

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("examples/_synthetic_input"))
    args=parser.parse_args()
    write_tiny(args.output)
    print("Generated synthetic inputs at",args.output)
