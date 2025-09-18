# app.py — JetLearn Insights + Predictability

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# --------------------- Page & Style ---------------------
st.set_page_config(page_title="JetLearn Insights + Predictability", layout="wide", page_icon="📊")
st.markdown("""
<style>
:root{
  --text:#0f172a; --muted:#64748b; --blue:#2563eb; --border: rgba(15,23,42,.10);
  --card:#fff; --bg:#f8fafc;
}
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.6rem; padding-bottom:.75rem; }
.head {
  position:sticky; top:0; z-index:50; display:flex; gap:10px; align-items:center;
  padding:10px 12px; background:#0b1220; color:#fff; border-radius:12px; margin-bottom:10px;
}
.head .title { font-weight:800; font-size:1.02rem; margin-right:auto; }
.section-title { font-weight:800; margin:.25rem 0 .6rem; color:var(--text); }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
.popcap { font-size:.78rem; color:var(--muted); margin-top:2px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]
REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country","Student/Academic Counsellor","Deal Stage","Create Date"]

# --------------------- Utilities ---------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try:
            if hasattr(file_or_path, "read"):
                file_or_path.seek(0)
                return pd.read_csv(file_or_path, encoding=enc)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

# --------------------- Header ---------------------
with st.container():
    st.markdown('<div class="head"><div class="title">📊 JetLearn — Insights & Predictability</div></div>', unsafe_allow_html=True)

# --------------------- Data Source (shared by both tabs) ---------------------

# -------- Historical dataset
ds = st.expander("Data source (Historical / Modeling)", expanded=True)
with ds:
    c1, c2, c3 = st.columns([3,2,2])
    with c1:
        uploaded_hist = st.file_uploader("Upload HISTORICAL CSV (modeling dataset)", type=["csv"], key="hist")
    with c2:
        path_hist = st.text_input("…or CSV path (historical)", value="Master_sheet_DB_10percent.csv")
    with c3:
        exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal'", value=True)

# -------- Current/Partial dataset
ds2 = st.expander("Prediction month data (Partial current month file)", expanded=True)
with ds2:
    c21, c22 = st.columns([3,2])
    with c21:
        uploaded_curr = st.file_uploader("Upload CURRENT MONTH PARTIAL CSV (e.g., September)", type=["csv"], key="curr")
    with c22:
        path_curr = st.text_input("…or CSV path (current month partial)", value="")
