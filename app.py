"""
Pakistan Air Pollution Dashboard
=================================
Interactive Streamlit app over the processed master layers (NO2 + PM2.5 + weather)
for 15 cities, 2019-present, plus a PM2.5 prediction model.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> share.streamlit.io -> point at app.py

Expects the processed files (written by the notebook's Part A) at:
    data/processed/master_daily.csv
    data/processed/master_monthly.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Pakistan Air Pollution", page_icon="🌫️", layout="wide")

DATA = Path("data/processed")
FEATURES = ["no2_e4", "temp_c", "wind_ms", "precip_mm", "sin_doy", "cos_doy"]

# WHO 24-hour PM2.5 guideline and rough AQI category cutoffs (ug/m3)
WHO_24H = 15
PM_BANDS = [(0, 12, "Good", "#2ecc71"), (12, 35, "Moderate", "#f1c40f"),
            (35, 55, "Unhealthy (sensitive)", "#e67e22"),
            (55, 150, "Unhealthy", "#e74c3c"),
            (150, 250, "Very unhealthy", "#8e44ad"),
            (250, 10000, "Hazardous", "#7f1d1d")]


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_daily():
    df = pd.read_csv(DATA / "master_daily.csv", parse_dates=["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


@st.cache_data
def load_monthly():
    return pd.read_csv(DATA / "master_monthly.csv", parse_dates=["date"])


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train the PM2.5 model once and cache it across reruns."""
    m = df.dropna(subset=FEATURES + ["pm25"])
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=20,
                               n_jobs=-1, random_state=0)
    rf.fit(m[FEATURES].values, m["pm25"].values)
    return rf


def pm_category(v):
    for lo, hi, name, color in PM_BANDS:
        if lo <= v < hi:
            return name, color
    return "Hazardous", "#7f1d1d"


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
try:
    daily = load_daily()
    monthly = load_monthly()
except FileNotFoundError:
    st.error("Could not find data/processed/master_daily.csv. "
             "Run Part A of the notebook first to generate the processed files.")
    st.stop()

CITIES = sorted(daily["city"].unique())
model = train_model(daily)

st.title("🌫️ Pakistan Air Pollution — 15 Cities, 2019–present")
st.caption("NO₂ (Sentinel-5P, combustion proxy) · PM2.5 (CAMS model, µg/m³) · "
           "Weather (ERA5-Land). NO₂ is a combustion proxy, not AQI; PM2.5 is modelled output.")

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Controls")
city = st.sidebar.selectbox("City", CITIES, index=CITIES.index("Lahore") if "Lahore" in CITIES else 0)
granularity = st.sidebar.radio("Time resolution", ["Monthly", "Weekly", "Daily"], index=0)
pollutant = st.sidebar.radio("Pollutant", ["PM2.5", "NO₂"], index=0)
yr_min, yr_max = int(daily["year"].min()), int(daily["year"].max())
yr_range = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

VALCOL = "pm25" if pollutant == "PM2.5" else "no2_e4"
VALLABEL = "PM2.5 (µg/m³)" if pollutant == "PM2.5" else "NO₂ (×10⁻⁴ mol/m²)"

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 City trends", "🏙️ Compare cities", "🔄 Seasonal cycle", "🔮 PM2.5 forecast"])

# ----------------------------------------------------------------------
# TAB 1 — single-city time series
# ----------------------------------------------------------------------
with tab1:
    st.subheader(f"{pollutant} over time — {city}")
    if granularity == "Daily":
        sub = daily[daily["city"] == city].copy()
        x = "date:T"
    elif granularity == "Weekly":
        s = daily[daily["city"] == city].set_index("date")[VALCOL].resample("W").mean()
        sub = s.reset_index(); x = "date:T"
    else:
        sub = monthly[monthly["city"] == city].copy(); x = "date:T"
    sub = sub[(sub["date"].dt.year >= yr_range[0]) & (sub["date"].dt.year <= yr_range[1])]

    line = alt.Chart(sub).mark_line(point=False).encode(
        x=alt.X(x, title="Date"),
        y=alt.Y(f"{VALCOL}:Q", title=VALLABEL),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip(f"{VALCOL}:Q", format=".1f")]
    ).properties(height=380).interactive()

    if pollutant == "PM2.5":
        rule = alt.Chart(pd.DataFrame({"y": [WHO_24H]})).mark_rule(
            color="green", strokeDash=[6, 4]).encode(y="y:Q")
        st.altair_chart(line + rule, use_container_width=True)
        st.caption(f"Dashed line = WHO 24-h guideline ({WHO_24H} µg/m³).")
    else:
        st.altair_chart(line, use_container_width=True)

    valid = sub[VALCOL].dropna()
    if len(valid):
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Mean {pollutant}", f"{valid.mean():.1f}")
        c2.metric("Max", f"{valid.max():.1f}")
        if pollutant == "PM2.5":
            c3.metric("% days over WHO", f"{(valid > WHO_24H).mean()*100:.0f}%")
        else:
            c3.metric("Min", f"{valid.min():.2f}")

# ----------------------------------------------------------------------
# TAB 2 — compare all cities
# ----------------------------------------------------------------------
with tab2:
    st.subheader(f"City ranking — mean {pollutant} ({yr_range[0]}–{yr_range[1]})")
    win = daily[(daily["year"] >= yr_range[0]) & (daily["year"] <= yr_range[1])]
    rank = (win.groupby("city")[VALCOL].mean().sort_values(ascending=False)
               .reset_index())
    bar = alt.Chart(rank).mark_bar().encode(
        x=alt.X(f"{VALCOL}:Q", title=f"Mean {VALLABEL}"),
        y=alt.Y("city:N", sort="-x", title=None),
        color=alt.Color(f"{VALCOL}:Q", scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["city", alt.Tooltip(f"{VALCOL}:Q", format=".2f")]
    ).properties(height=460)
    st.altair_chart(bar, use_container_width=True)
    st.caption("Tip: switch the pollutant in the sidebar — the ranking changes, "
               "and where it changes is the interesting part (local combustion vs transported particulate).")

# ----------------------------------------------------------------------
# TAB 3 — seasonal climatology
# ----------------------------------------------------------------------
with tab3:
    st.subheader(f"Seasonal cycle (month-of-year average) — {city}")
    comp = daily[daily["year"] < 2026]  # exclude partial year
    clim = (comp[comp["city"] == city].groupby("month")[["pm25", "no2_e4"]]
                 .mean().reset_index())
    base = alt.Chart(clim).encode(x=alt.X("month:O", title="Month"))
    a = base.mark_line(point=True, color="#C44E52").encode(
        y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)", axis=alt.Axis(titleColor="#C44E52")))
    b = base.mark_line(point=True, color="#4C72B0").encode(
        y=alt.Y("no2_e4:Q", title="NO₂ (×10⁻⁴)", axis=alt.Axis(titleColor="#4C72B0")))
    st.altair_chart(alt.layer(a, b).resolve_scale(y="independent").properties(height=380),
                    use_container_width=True)
    st.caption("Red = PM2.5, Blue = NO₂. Both typically peak in winter (trapped air, "
               "crop burning) and trough in the monsoon (rain scrubs the air).")

# ----------------------------------------------------------------------
# TAB 4 — interactive forecast / what-if from the trained model
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Predict PM2.5 from NO₂ + weather")
    st.write("Move the sliders to see how the trained model expects PM2.5 to respond. "
             "This illustrates the **dispersion effect** — how wind and rain change "
             "particulate levels even when emissions are held fixed.")

    cd = daily[daily["city"] == city].dropna(subset=FEATURES + ["pm25"])
    defaults = cd[FEATURES].median() if len(cd) else daily[FEATURES].median()

    col_a, col_b = st.columns(2)
    with col_a:
        no2_in = st.slider("NO₂ (×10⁻⁴ mol/m²)", 0.0, 5.0, float(round(defaults["no2_e4"], 2)), 0.05)
        temp_in = st.slider("Temperature (°C)", -5.0, 45.0, float(round(defaults["temp_c"], 1)), 0.5)
        month_in = st.slider("Month of year", 1, 12, 1)
    with col_b:
        wind_in = st.slider("Wind speed (m/s)", 0.0, 10.0, float(round(defaults["wind_ms"], 1)), 0.1)
        precip_in = st.slider("Precipitation (mm)", 0.0, 50.0, float(round(defaults["precip_mm"], 1)), 0.5)

    doy = int((month_in - 0.5) / 12 * 365.25)
    feat = np.array([[no2_in, temp_in, wind_in, precip_in,
                      np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)]])
    pred = float(model.predict(feat)[0])
    name, color = pm_category(pred)

    st.markdown(f"### Predicted PM2.5: "
                f"<span style='color:{color}'>{pred:.0f} µg/m³ — {name}</span>",
                unsafe_allow_html=True)
    st.progress(min(pred / 250, 1.0))

    # show the dispersion effect explicitly: sweep wind, hold the rest
    sweep = []
    for w in np.linspace(0, 8, 25):
        f = feat.copy(); f[0, 2] = w
        sweep.append({"wind_ms": w, "pm25": float(model.predict(f)[0])})
    sweep = pd.DataFrame(sweep)
    chart = alt.Chart(sweep).mark_line(color="#4C72B0").encode(
        x=alt.X("wind_ms:Q", title="Wind speed (m/s)"),
        y=alt.Y("pm25:Q", title="Predicted PM2.5 (µg/m³)")
    ).properties(height=300, title="Dispersion effect: PM2.5 vs wind (other inputs fixed)")
    st.altair_chart(chart, use_container_width=True)
    st.caption("A downward slope = wind disperses particulate. This is the model "
               "quantifying the dispersion mechanism behind the winter pollution peaks.")

st.sidebar.markdown("---")
st.sidebar.caption("Data: Sentinel-5P, CAMS, ERA5-Land via Google Earth Engine. "
                   "PM2.5 is modelled (~40 km), not a ground measurement.")
