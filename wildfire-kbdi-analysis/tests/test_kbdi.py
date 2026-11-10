import numpy as np
import pytest
from kbdi.kbdi_core import initialize, daily_step

def arr(v):
    return np.array([[float(v)]])

def test_temperature_threshold_and_rain_reduction():
    state=initialize((1,1),value=100)
    state,obs=daily_step(state,arr(10),arr(0),arr(1000))
    assert np.isclose(obs.item(),100.0)
    state,obs=daily_step(state,arr(5),arr(30.48),arr(1000),rainfall_mode="daily")
    assert np.isclose(obs.item(),0.0)

def test_equation_independent_scalar():
    state=initialize((1,1),value=120)
    state,obs=daily_step(state,arr(30),arr(0),arr(1000))
    temp_f=86.0
    rain_inch=1000/25.4
    expected=120+(800-120)*(.968*np.exp(.0486*temp_f)-8.30)*.001/(1+10.88*np.exp(-.0441*rain_inch))
    assert np.isclose(obs.item(),expected,rtol=1e-12)

def test_wet_spell_threshold_and_reset():
    state=initialize((1,1),value=100)
    for p in (3,3):
        state,obs=daily_step(state,arr(0),arr(p),arr(1000),rainfall_mode="wet_spell")
    assert np.isclose(obs.item(),100-(6-5.08)/0.254)
    state,obs=daily_step(state,arr(0),arr(0),arr(1000),rainfall_mode="wet_spell")
    state,obs=daily_step(state,arr(0),arr(3),arr(1000),rainfall_mode="wet_spell")
    assert np.isclose(obs.item(),100-(6-5.08)/0.254)

def test_invalid_cell_is_not_filled():
    state=initialize((1,1),value=150)
    state,obs=daily_step(state,arr(np.nan),arr(0),arr(1000))
    assert np.isnan(obs.item()) and state.index.item()==150

def test_bounds_and_bad_inputs():
    state=initialize((1,1),value=799)
    for _ in range(90):
        state,obs=daily_step(state,arr(50),arr(0),arr(200))
    assert 0<=obs.item()<=800
    with pytest.raises(ValueError):
        initialize((1,1),value=900)
    with pytest.raises(ValueError):
        daily_step(state,arr(20),arr(0),arr(500),"unknown")
