import numpy as np
import pytest
from permafrost.frost_number_classification import frost_number,classify

def test_thresholds_match_original():
    fi=np.array([[.56**2,.54**2,.005**2,.11**2],
                 [.60**2,.09**2,.02**2,0.]])
    ti=np.array([[(1-.56)**2,(1-.54)**2,(1-.005)**2,(1-.11)**2],
                 [(1-.60)**2,(1-.09)**2,(1-.02)**2,0.]])
    tp=np.array([[1,1,1,0],[0,0,0,0]])
    expected=np.array([[3,2,0,2],[3,1,1,-9999]])
    np.testing.assert_array_equal(classify(fi,ti,tp),expected)
    assert np.isnan(frost_number(np.array([0.]),np.array([0.]))[0])

def test_negative_indices_rejected():
    with pytest.raises(ValueError):
        frost_number(np.array([-3.]),np.array([10.]))
