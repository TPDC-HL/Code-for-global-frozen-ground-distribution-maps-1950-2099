
import numpy as np
import xarray as xr
import os
from pathlib import Path

# Define models, variants, and scenarios
models = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CMCC-ESM2",
    "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3", "EC-Earth3-Veg-LR", "FGOALS-g3",
    "GFDL-ESM4", "GISS-E2-1-G", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR",
    "KACE-1-0-G", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
    "MRI-ESM2-0", "NorESM2-LM", "NorESM2-MM", "TaiESM1", "UKESM1-0-LL"
]

variants = [
    "r1i1p1f1_gn", "r1i1p1f1_gn", "r1i1p1f1_gn", "r1i1p1f2_gn", "r1i1p1f1_gn",
    "r1i1p1f2_gr", "r1i1p1f2_gr", "r1i1p1f1_gr", "r1i1p1f1_gr", "r1i1p1f1_gn",
    "r1i1p1f1_gr", "r1i1p1f2_gn", "r1i1p1f1_gr", "r1i1p1f1_gr", "r1i1p1f1_gr",
    "r1i1p1f1_gr", "r1i1p1f2_gn", "r1i1p1f1_gn", "r1i1p1f1_gn", "r1i1p1f1_gn",
    "r1i1p1f1_gr", "r1i1p1f1_gn", "r1i1p1f1_gn", "r1i1p1f1_gn", "r1i1p1f2_gn"
]

scenarios = ["ssp126", "ssp245", "ssp370", "ssp585"]

# Read reference file to get lat/lon coordinates
f_ref = xr.open_dataset(f"tas_day_{models[0]}_{scenarios[0]}_{variants[0]}_2015.nc")
lat = f_ref['lat'].values
lon = f_ref['lon'].values
f_ref.close()

nmodels = len(models)
nlat = len(lat)
nlon = len(lon)

# Loop through scenarios
for scenario in scenarios:
    output_dir = f"./{scenario.upper()}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Loop through years
    for y in range(1950, 2101):
        print(f"Processing {scenario} - Year {y}")
        
        ensemble_sum = np.zeros((nlat, nlon), dtype=np.float32)
        model_count = 0
        
        # Loop through models
        for m in range(nmodels):
            model = models[m]
            variant = variants[m]
            
            filename = f"tas_day_{model}_{scenario}_{variant}_{y}.nc"
            
            # Check if file exists
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found, skipping...")
                continue
            
            try:
                FI_3 = np.zeros((nlat, nlon), dtype=np.float32)
                
                # Open file and process July data
                ds = xr.open_dataset(filename)
                time = ds['time']
                
                # Convert time to calendar format and find July indices
                time_df = time.to_dataframe()
                july_mask = time_df.index.month == 7
                july_indices = np.where(july_mask)[0]
                
                if len(july_indices) > 0:
                    # Process July data
                    tas1 = ds['tas'].isel(time=slice(july_indices[0], None)).values - 273.15
                    tas2 = np.where(tas1 > 0, tas1, 0)
                    tas3 = np.sum(tas2, axis=0)
                    FI_3 += tas3
                    
                    # Process data before July (if exists)
                    if july_indices[0] > 0:
                        tas4 = ds['tas'].isel(time=slice(0, july_indices[0])).values - 273.15
                        tas5 = np.where(tas4 > 0, tas4, 0)
                        tas6 = np.sum(tas5, axis=0)
                        FI_3 += tas6
                
                ds.close()
                
                # Calculate absolute value
                FI_4 = np.abs(FI_3)
                
                # Save individual model output
                output_filename = os.path.join(output_dir, f"TI_{model}_{y}.nc")
                ds_out = xr.Dataset(
                    {
                        'TI': (['lat', 'lon'], FI_4)
                    },
                    coords={
                        'lat': lat,
                        'lon': lon
                    }
                )
                ds_out.to_netcdf(output_filename)
                ds_out.close()
                
                # Add to ensemble sum
                ensemble_sum += FI_4
                model_count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Calculate and save ensemble mean
        if model_count > 0:
            ensemble_mean = ensemble_sum / model_count
            
            ensemble_filename = os.path.join(output_dir, f"TI_ensemble_mean_{y}.nc")
            ds_ens = xr.Dataset(
                {
                    'TI': (['lat', 'lon'], ensemble_mean)
                },
                coords={
                    'lat': lat,
                    'lon': lon
                }
            )
            ds_ens.to_netcdf(ensemble_filename)
            ds_ens.close()
            
            print(f"Completed {scenario} - Year {y} with {model_count} models")
        else:
            print(f"Warning: No valid models for {scenario} - Year {y}")

print("Processing complete!")