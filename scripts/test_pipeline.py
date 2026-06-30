"""
test_pipeline.py — unit tests for the AirVision-2026 data pipeline.

Covers the two things most worth locking down in a data project:
  1) the unit conversions used during extraction (a wrong factor silently
     corrupts every downstream number), and
  2) the cleaning/merge logic (NaN handling, no fillna(0), negative handling,
     and the outer join on city+date).

Run from the repo root:   pytest -q
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# The conversion functions, mirroring exactly what the Earth Engine scripts do.
# Kept here as plain Python so they can be tested in isolation.
# ---------------------------------------------------------------------------
def pm25_kg_to_ug(kg_per_m3):
    """CAMS PM2.5: kilograms/m^3 -> micrograms/m^3."""
    return kg_per_m3 * 1e9


def no2_to_display_units(mol_per_m2):
    """NO2: raw mol/m^2 -> the ×10^-4 reading used in charts/tables."""
    return mol_per_m2 * 1e4


def kelvin_to_celsius(kelvin):
    """ERA5 temperature: Kelvin -> Celsius."""
    return kelvin - 273.15


def metres_to_mm(metres):
    """ERA5 precipitation: metres -> millimetres."""
    return metres * 1000.0


def wind_speed(u, v):
    """ERA5 wind: combine u/v components into a scalar speed (hypotenuse)."""
    return np.hypot(u, v)


# ---------------------------------------------------------------------------
# 1. UNIT CONVERSIONS
# ---------------------------------------------------------------------------
class TestConversions:

    def test_pm25_typical_lahore_winter(self):
        # ~1.1e-7 kg/m^3 should be a hazardous ~110 ug/m^3
        assert pm25_kg_to_ug(1.1e-7) == pytest.approx(110.0)

    def test_pm25_zero(self):
        assert pm25_kg_to_ug(0.0) == 0.0

    def test_no2_scaling(self):
        # a raw 9.4e-5 mol/m^2 (Lahore-ish) -> 0.94 in display units
        assert no2_to_display_units(9.4e-5) == pytest.approx(0.94)

    def test_kelvin_to_celsius(self):
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0)
        assert kelvin_to_celsius(308.15) == pytest.approx(35.0)   # a hot Multan day

    def test_metres_to_mm(self):
        assert metres_to_mm(0.005) == pytest.approx(5.0)
        assert metres_to_mm(0.0) == 0.0

    def test_wind_speed_pythagorean(self):
        assert wind_speed(3.0, 4.0) == pytest.approx(5.0)

    def test_wind_speed_is_non_negative(self):
        # opposing-sign components still give a positive magnitude
        assert wind_speed(-3.0, -4.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Cleaning / merge helpers, mirroring the notebook's Part A.
# ---------------------------------------------------------------------------
def clean_values(series):
    """Blanks/garbage -> NaN; never fill with 0."""
    return pd.to_numeric(series, errors="coerce")


def no2_display_column(no2_raw):
    """Floor negatives to 0 for display ONLY; raw stays untouched."""
    return no2_raw.clip(lower=0)


def merge_variables(no2, pm25, weather):
    """Outer-merge the three variables on (city, date)."""
    return (no2.merge(pm25, on=["city", "date"], how="outer")
               .merge(weather, on=["city", "date"], how="outer")
               .sort_values(["city", "date"])
               .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 2. CLEANING LOGIC
# ---------------------------------------------------------------------------
class TestCleaning:

    def test_blanks_become_nan_not_zero(self):
        s = pd.Series(["0.5", "", "0.3", None])
        out = clean_values(s)
        assert out.isna().sum() == 2          # the two blanks
        assert (out == 0).sum() == 0          # nothing silently became 0
        assert out.iloc[0] == pytest.approx(0.5)

    def test_missing_is_preserved_through_mean(self):
        # a NaN must be skipped by mean, not treated as 0 (which would drag it down)
        s = pd.Series([100.0, np.nan, 120.0])
        assert s.mean() == pytest.approx(110.0)        # (100+120)/2, NaN skipped
        assert s.fillna(0).mean() != pytest.approx(110.0)  # the wrong way, for contrast

    def test_negative_no2_kept_in_raw(self):
        raw = pd.Series([-1e-6, 5e-5, -2e-6])
        # raw must retain the negatives (clipping would bias the mean upward)
        assert (raw < 0).sum() == 2

    def test_no2_display_floors_negatives(self):
        raw = pd.Series([-1e-6, 5e-5])
        disp = no2_display_column(raw)
        assert (disp < 0).sum() == 0
        assert disp.iloc[1] == pytest.approx(5e-5)     # positives unchanged


# ---------------------------------------------------------------------------
# 3. MERGE LOGIC
# ---------------------------------------------------------------------------
class TestMerge:

    def _frames(self):
        dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
        no2 = pd.DataFrame({"city": ["Lahore"] * 2, "date": dates, "no2": [9e-5, 8e-5]})
        pm25 = pd.DataFrame({"city": ["Lahore"] * 2, "date": dates, "pm25": [110.0, 95.0]})
        weather = pd.DataFrame({"city": ["Lahore"] * 2, "date": dates,
                                "temp_c": [12.0, 13.0], "wind_ms": [1.5, 2.0],
                                "precip_mm": [0.0, 0.0]})
        return no2, pm25, weather

    def test_merge_aligns_on_city_date(self):
        m = merge_variables(*self._frames())
        assert len(m) == 2
        assert set(["no2", "pm25", "temp_c", "wind_ms", "precip_mm"]).issubset(m.columns)
        row = m[m["date"] == pd.Timestamp("2023-01-01")].iloc[0]
        assert row["pm25"] == pytest.approx(110.0)
        assert row["temp_c"] == pytest.approx(12.0)

    def test_outer_merge_keeps_unmatched_day_as_nan(self):
        # PM2.5 is missing 2023-01-02 (the corrupt-image scenario);
        # an outer join must keep that day with pm25 = NaN, not drop it.
        no2, pm25, weather = self._frames()
        pm25 = pm25[pm25["date"] == pd.Timestamp("2023-01-01")]   # drop day 2
        m = merge_variables(no2, pm25, weather)
        assert len(m) == 2                                         # day 2 still present
        day2 = m[m["date"] == pd.Timestamp("2023-01-02")].iloc[0]
        assert pd.isna(day2["pm25"])                               # kept as NaN
        assert day2["no2"] == pytest.approx(8e-5)                  # other vars intact

    def test_no_duplicate_city_date_rows(self):
        m = merge_variables(*self._frames())
        assert not m.duplicated(subset=["city", "date"]).any()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
