# app.py — Smart MTD vs Cohort Analyzer
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, datetime, timedelta

# ---------- Page config & light theming ----------
st.set_page_config(page_title="MTD vs Cohort Analyzer", layout="wide", page_icon="📊")
st.markdown("""
<style>
/* Tighten layout a bit */
.block-container {padding-top: 1rem; padding-bottom: 1rem;}

/* Pretty section titles */
h2, h3 { margin-top: 0.2rem !important; }

/* Metric cards look */
.kpi-card {
  padding: 14px 16px;
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: 12px;
  background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* Table subtitle */
.caption { color:#6b7280; font-size:0.9rem; margin-top:-8px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Utilities
# =========================================================
REQUIRED_COLS = [
    "Pipeline",
    "JetLearn Deal Source",
    "Country",
    "Student/Academic Counsellor",
    "Deal Stage",
    "Create Date",
]

def robust_read_csv(file_or_path):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    """Find date-like columns (other than Create Date)."""
    date_like = []
    for col in df.columns:
        if col == "Create Date":
            continue
        if any(k in col.lower() for k in ["date", "time", "timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed
                date_like.append(col)
    # Prioritize Payment Received Date if present
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c != "Payment Received Date"]
    return date_like

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    """Global categorical filter helper."""
    if all_checked:
        return pd.Series(True, index=series.index)
    uniq = series.dropna().astype(str).nunique()
    if selected_values and len(selected_values) == uniq:
        return pd.Series(True, index=series.index)
    if not selected_values:
        return pd.Series(False, index=series.index)
    return series.astype(str).isin(selected_values)

def safe_minmax_date(s:
