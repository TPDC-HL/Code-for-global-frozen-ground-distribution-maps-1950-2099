from pathlib import Path
import numpy as np
import xarray as xr
from make_synthetic_data import write_tiny
from kbdi.run_cmip6_kbdi import main as run_main
from kbdi.ensemble import aggregate_files
from permafrost.frost_number_classification import classify_files

def test_end_to_end_two_models_two_scenarios(tmp_path):
    inputs=write_tiny(tmp_path/"synthetic")
    out=tmp_path/"results"
    run_main(["--input-root",str(inputs),"--output-root",str(out),
       "--historical-start","2000","--historical-end","2001",
       "--baseline-start","2000","--baseline-end","2001",
       "--start-year","2002","--end-year","2002",
       "--models","MODEL-A","MODEL-B","--scenarios","ssp126","ssp245"])
    for scenario in ("ssp126","ssp245"):
        paths=[out/scenario/f"KBDI_{model}_{scenario}_2002.nc"
               for model in ("MODEL-A","MODEL-B")]
        output=tmp_path/f"ens_{scenario}.nc"
        aggregate_files(paths,output,min_models=2,model_names=["MODEL-A","MODEL-B"])
        with xr.open_dataset(paths[0]) as m1,xr.open_dataset(paths[1]) as m2,xr.open_dataset(output) as ens:
            expected=(m1["kbdi_annual_mean"]+m2["kbdi_annual_mean"])/2
            np.testing.assert_allclose(ens["kbdi_annual_mean_mme_mean"].values,expected.values,rtol=1e-6)
            assert (ens["kbdi_annual_mean_n_gcms"].values==2).all()
            assert m1.attrs["rainfall_interception"]=="wet_spell"
    pf=tmp_path/"pf.nc"
    classify_files(inputs/"SSP126"/"FI_MODEL-A_2002.nc",
                   inputs/"SSP126"/"TI_MODEL-A_2002.nc",
                   inputs/"GTOPOP_v4.nc",pf)
    with xr.open_dataset(pf) as ds:
        assert ds["frozen"].values.shape==(2,3)
        assert set(np.unique(ds["frozen"].values)).issubset({0,1,2,3,-9999})
