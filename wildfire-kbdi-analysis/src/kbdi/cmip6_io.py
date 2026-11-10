"""Strict NetCDF I/O and unit conversion for NEX-GDDP-CMIP6.

Expected variables: tasmax (daily maximum temperature) and pr (daily
precipitation rate). No units inferred silently. Dates and grids validated.
"""
from __future__ import annotations
from contextlib import contextmanager
from glob import glob
from pathlib import Path
import numpy as np
import xarray as xr

def resolve_file(root: Path, pattern: str, scenario: str, model: str, variable: str, year: int) -> Path:
    rel = pattern.format(scenario=scenario, SCENARIO=scenario.upper(),
                         model=model, variable=variable, year=year)
    matches = sorted(glob(str(root / rel)))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {variable} file, found {len(matches)}: {root / rel}")
    return Path(matches[0])

def _units_attr(da: xr.DataArray, override, kind):
    unit = (override or da.attrs.get("units") or "").strip().lower()
    unit = unit.replace(" ", "").replace("−", "-").replace("**", "^")
    if not unit:
        raise ValueError(f"{kind} missing units; specify an explicit override")
    return unit

def temp_c(da, override=None):
    unit = _units_attr(da, override, "tasmax")
    if unit in ("k", "kelvin", "degrees_k", "degk"):
        return da - 273.15
    if unit in ("c", "degc", "°c", "degree_celsius", "degrees_celsius", "celsius"):
        return da
    raise ValueError(f"Unrecognized tasmax units: {unit!r}")

def precip_mm_day(da, override=None):
    unit = _units_attr(da, override, "pr")
    if unit in ("kgm-2s-1", "kg/m2/s", "kgm^-2s^-1", "kgm-2sec-1", "mm/s", "mms-1"):
        return da * 86400.0
    if unit in ("mm/day", "mmday-1", "mmd-1", "mm", "kgm-2day-1", "kgm-2d-1", "kg/m2/day"):
        return da
    raise ValueError(f"Unrecognized pr units: {unit!r}")

def normalise_override(kind):
    if kind is None:
        return None
    return {"K":"K", "C":"C", "flux":"kg m-2 s-1", "mm-day":"mm/day"}[kind]

def validate_daily(da, year, name):
    if da.dims != ("time", "lat", "lon"):
        raise ValueError(f"{name} requires (time,lat,lon), found {da.dims}")
    yy = np.asarray(da.time.dt.year.values, dtype=int)
    dd = np.asarray(da.time.dt.dayofyear.values, dtype=int)
    if len(yy) < 360 or len(yy) > 366 or np.any(yy != year) or not np.array_equal(dd, np.arange(1,len(dd)+1)):
        raise ValueError(f"{name}: missing, duplicate, unordered or wrong-year timestamps: {year}")
    if not np.all(np.diff(np.asarray(da.lat.values)) > 0):
        raise ValueError(f"{name}: latitude must be strictly ascending")

def check_match(a,b):
    for dim in ("time","lat","lon"):
        if not np.array_equal(a[dim].values,b[dim].values):
            raise ValueError(f"tasmax/pr coordinates differ: {dim}")

@contextmanager
def open_pair(root,pattern,scenario,model,year,temp_units=None,precip_units=None):
    tp=resolve_file(root,pattern,scenario,model,"tasmax",year)
    pp=resolve_file(root,pattern,scenario,model,"pr",year)
    with xr.open_dataset(tp) as tds, xr.open_dataset(pp) as pds:
        if "tasmax" not in tds or "pr" not in pds:
            raise KeyError(f"Expected tasmax in {tp} and pr in {pp}")
        t=tds["tasmax"].transpose("time","lat","lon")
        p=pds["pr"].transpose("time","lat","lon")
        validate_daily(t,year,"tasmax")
        validate_daily(p,year,"pr")
        check_match(t,p)
        yield temp_c(t,normalise_override(temp_units)),precip_mm_day(p,normalise_override(precip_units))

@contextmanager
def open_precip(root,pattern,scenario,model,year,precip_units=None):
    pp=resolve_file(root,pattern,scenario,model,"pr",year)
    with xr.open_dataset(pp) as pds:
        p=pds["pr"].transpose("time","lat","lon")
        validate_daily(p,year,"pr")
        yield precip_mm_day(p,normalise_override(precip_units))
