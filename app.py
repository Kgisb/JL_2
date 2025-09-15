# app_compare_pro_clean.py — Blue, minimal A/B analyzer (no "compact mode"), popover filters + summaries
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ------------------------ Page & Theme ------------------------
st.set_page_config(page_title="MTD vs Cohort — A/B Compare (Clean)",
                   layout="wide", page_icon="📊")

st.markdown("""
<style>
:root{
  --bg:#ffffff; --text:#0f172a; --muted:#6b7280;
  --blue:#1e40af; --blue-600:#2563eb; --blue-700:#1d4ed8;
  --border: rgba(15,23,42,.10); --card:#ffffff;
}
html, body, [class*="css"] {
  font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji";
}
.block-container { padding-top: .4rem; padding-bottom: .8rem; }

/* Top bar */
.nav {
  position: sticky; top: 0; z-index: 20; padding: 8px 12px;
  background: linear-gradient(90deg, var(--blue-700), var(--blue-600));
  color: #fff; border-radius: 12px; margin-bottom: 10px;
}
.nav .title { font-weight: 800; letter-spacing:.2px; }
.nav .sub   { font-size:.85rem; opacity:.9; margin-top:2px; }

/* Compact filters bar */
.filters-bar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 6px 8px; border: 1px solid var(--border); border-radius: 12px;
  background: #f8fafc;
}

/* Sections & cards */
.section-title { display:flex; align-items:center; gap:.5rem; font-weight:800; margin:.25rem 0 .6rem; }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
.kpi .delta { font-size:.84rem; color: var(--blue-600); }

@media (max-width: 820px) {
  .block-container { padding-left:.5rem; padding-right:.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------ Constants ------------------------
REQUIRED_COLS = [
    "Pipeline","JetLearn Deal Source","Country",
    "Student/Academic Counsellor","Deal Stage","Create Date",
]
PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# ------------------------ Utils ------------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    date_like = []
    for col in df.columns:
        if col == "Create Date":
            continue
        if any(k in col.lower() for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed
                date_like.append(col)
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c != "Payment Received Date"]
    return date_like

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    if all_checked:
        return pd.Series(True, index=series.index)
    uniq = series.dropna().astype(str).nunique()
    if selected_values and len(selected_values) == uniq:
        return pd.Series(True, index=series.index)
    if not selected_values:
        return pd.Series(False, index=series.index)
    return series.astype(str).isin(selected_values)

def safe_minmax_date(s: pd.Series, fallback=(date(2020,1,1), date.today())):
    if s.isna().all():
        return fallback
    return (pd.to_datetime(s.min()).date(), pd.to_datetime(s.max()).date())

# Date presets
def today_bounds():
    t = pd.Timestamp.today().date(); return t, t
def this_month_so_far_bounds():
    t = pd.Timestamp.today().date(); return t.replace(day=1), t
def last_month_bounds():
    first_this = pd.Timestamp.today().date().replace(day=1)
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return first_prev, last_prev
def quarter_start(y, q): return date(y, 3*(q-1)+1, 1)
def last_quarter_bounds():
    t = pd.Timestamp.today().date(); q = (t.month - 1)//3 + 1
    if q == 1: y, lq = t.year - 1, 4
    else:      y, lq = t.year, q - 1
    start = quarter_start(y, lq)
    next_start = quarter_start(y+1, 1) if lq == 4 else quarter_start(y, lq+1)
    return start, (next_start - timedelta(days=1))
def this_year_so_far_bounds():
    t = pd.Timestamp.today().date(); return date(t.year,1,1), t

def date_range_from_preset(label, series: pd.Series, key_prefix: str):
    presets = ["Today","This month so far","Last month","Last quarter","This year","Custom"]
    choice = st.radio(label, presets, horizontal=True, key=f"{key_prefix}_preset")
    if choice == "Today": return today_bounds()
    if choice == "This month so far": return this_month_so_far_bounds()
    if choice == "Last month": return last_month_bounds()
    if choice == "Last quarter": return last_quarter_bounds()
    if choice == "This year": return this_year_so_far_bounds()
    dmin, dmax = safe_minmax_date(series)
    rng = st.date_input("Custom range", (dmin, dmax), key=f"{key_prefix}_custom")
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        return rng[0], rng[1]
    return dmin, dmax

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def alt_line(df, x, y, color=None, tooltip=None, height=260):
    enc = dict(x=alt.X(x, title=None), y=alt.Y(y, title=None), tooltip=tooltip or [])
    if color: enc["color"] = alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

# ------------------------ Top Nav -------
