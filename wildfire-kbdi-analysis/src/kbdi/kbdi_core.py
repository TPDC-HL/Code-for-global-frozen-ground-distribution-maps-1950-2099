"""Vectorized daily Keetch–Byram Drought Index (KBDI; 0–800).

Reference implementation (not the original source for the paper's projections).
tasmax_c: daily maximum 2 m air temperature in Celsius
pr_mm: daily precipitation total in mm
mean_annual_pr_mm: historical annual-mean precipitation in mm/year
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

RAIN_THRESHOLD_MM = 5.08
MM_PER_POINT = 0.254

@dataclass
class KBDIState:
    index: np.ndarray
    wet_spell_total_mm: np.ndarray

def initialize(shape, value=0.0):
    if not np.isfinite(value) or not 0 <= value <= 800:
        raise ValueError("Initial KBDI must lie in [0, 800]")
    return KBDIState(np.full(shape, value, dtype=np.float64),
                     np.zeros(shape, dtype=np.float64))

def daily_step(state: KBDIState, tasmax_c, pr_mm, mean_annual_pr_mm,
               rainfall_mode="wet_spell"):
    """Apply effective rain before temperature-driven drying.

    wet_spell: 0.2-inch rain interception once per continuous wet spell.
    daily: 0.2-inch threshold separately each day.
    Missing/invalid cells return NaN and retain previous state.
    """
    if rainfall_mode not in ("wet_spell", "daily"):
        raise ValueError("rainfall_mode must be wet_spell or daily")
    q = np.asarray(state.index, dtype=np.float64)
    p = np.broadcast_to(np.asarray(pr_mm, dtype=np.float64), q.shape)
    t = np.broadcast_to(np.asarray(tasmax_c, dtype=np.float64), q.shape)
    rain_clim = np.broadcast_to(np.asarray(mean_annual_pr_mm, dtype=np.float64), q.shape)
    wet_prev = np.broadcast_to(state.wet_spell_total_mm, q.shape)
    ok = (np.isfinite(q) & np.isfinite(p) & np.isfinite(t) &
          np.isfinite(rain_clim) & (rain_clim > 0) & (p >= 0))
    p_safe = np.where(ok, p, 0.0)
    t_safe = np.where(ok, t, 0.0)
    r_safe = np.where(ok, rain_clim, 1.0)
    if rainfall_mode == "wet_spell":
        wet_next = np.where(p_safe > 0, wet_prev + p_safe, 0.0)
        net_rain = np.where(
            p_safe > 0,
            np.maximum(wet_next - RAIN_THRESHOLD_MM, 0) -
            np.maximum(wet_prev - RAIN_THRESHOLD_MM, 0), 0.0)
    else:
        wet_next = np.zeros_like(q)
        net_rain = np.maximum(p_safe - RAIN_THRESHOLD_MM, 0)
    wet_next = np.where(ok, wet_next, wet_prev)
    after_rain = np.clip(q - net_rain / MM_PER_POINT, 0, 800)
    t_f = t_safe * 9.0 / 5.0 + 32.0
    rain_inch = r_safe / 25.4
    drying = (0.968 * np.exp(0.0486 * t_f) - 8.30)
    increment = (800.0 - after_rain) * np.maximum(0.0, drying) * 0.001 / (
        1.0 + 10.88 * np.exp(-0.0441 * rain_inch))
    increment = np.where(t_safe > 10.0, increment, 0.0)
    result = np.clip(after_rain + increment, 0, 800)
    return (
        KBDIState(np.where(ok, result, q), wet_next),
        np.where(ok, result, np.nan),
    )
