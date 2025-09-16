# app.py — Drawer UI + A/B Compare + Predictability (with robust 'Pipeline' handling)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta, datetime

# --- ML (Predictability)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# --------------------------- PAGE / THEME ---------------------------
st.set_page_config(page_title="MTD vs Cohort — Drawer UI + Predictability",
                   layout="wide", page_icon="📊")
st.markdown("""
<style>
:root{
  --text:#0f172a; --muted:#64748b; --blue-600:#2563eb; --blue-700:#1d4ed8;
  --border: rgba(15,23,42,.10); --card:#fff; --bg:#f8fafc;
}
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.5rem; padding-bottom:.75rem; }
.topbar {
  position:sticky; top:0; z-index:50; display:flex; gap:8px; align-items:center;
  padding:10px 12px; background:#0b1220; color:#fff; border-radius:12px; margin-bottom:10px;
}
.topbar .title { font-weight:800; font-size:1.02rem; margin-right:auto; white-space:nowrap; }
.section-title { font-weight:800; margin:.25rem 0 .6rem; color:var(--text); }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
.kpi .delta { font-size:.84rem; color: var(--blue-600); }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
.popcap { font-size:.78rem; color:var(--muted); margin-top:2px; }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country",
                 "Student/Academic Counsellor","Deal Stage","Create Date"]
PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# --------------------------- CLONE REQUEST HANDLING ---------------------------
def _request_clone(direction: str):
    st.session_state["__clone_request__"] = direction
    if direction == "A2B":
        st.session_state["show_b"] = True

def _perform_clone_if_requested():
    direction = st.session_state.get("__clone_request__")
    if not direction: return
    src_prefix, dst_prefix = ("A_", "B_") if direction == "A2B" else ("B_", "A_")
    for k in list(st.session_state.keys()):
        if not k.startswith(src_prefix): continue
        if k.endswith("_select_all") or k.endswith("_clear"): continue
        st.session_state[k.replace(src_prefix, dst_prefix, 1)] = st.session_state[k]
    st.session_state["__clone_request__"] = None
    st.rerun()
_perform_clone_if_requested()

# --------------------------- HELPERS (I/O & CLEANING) ---------------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try: return pd.read_csv(file_or_path, encoding=enc)
        except Exception: pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    date_like=[]
    for col in df.columns:
        if col=="Create Date": continue
        if any(k in col.lower() for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum()>0:
                df[col]=parsed; date_like.append(col)
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c!="Payment Received Date"]
    return date_like

def ensure_program_column(df: pd.DataFrame, target="Pipeline") -> pd.DataFrame:
    """Guarantee a 'Pipeline' column exists (case/space tolerant; synonyms)."""
    if target in df.columns:
        return df
    norm = {c: c.strip().lower() for c in df.columns}
    aliases = ["pipeline", "program", "programme", "course", "track"]
    hit = None
    for col, low in norm.items():
        if low in aliases:
            hit = col; break
    if hit is not None:
        df = df.rename(columns={hit: target})
        return df
    df[target] = "Unknown"
    st.warning("‘Pipeline’ column not found; created a fallback ‘Unknown’ Pipeline. "
               "Rename your CSV column to ‘Pipeline’ for best results.")
    return df

# --------------------------- SMALL UTILITIES ---------------------------
def coerce_list(x): 
    if x is None: return []
    if isinstance(x,(list,tuple,set)): return list(x)
    return [x]

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    if all_checked: return pd.Series(True, index=series.index)
    sel = [str(v) for v in coerce_list(selected_values)]
    if len(sel)==0: return pd.Series(False, index=series.index)
    return series.astype(str).isin(sel)

def safe_minmax_date(s: pd.Series, fallback=(date(2020,1,1), date.today())):
    if s.isna().all(): return fallback
    return (pd.to_datetime(s.min()).date(), pd.to_datetime(s.max()).date())
def today_bounds(): t=pd.Timestamp.today().date(); return t,t
def this_month_so_far_bounds(): t=pd.Timestamp.today().date(); return t.replace(day=1),t
def last_month_bounds():
    first_this = pd.Timestamp.today().date().replace(day=1)
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return first_prev, last_prev
def quarter_start(y,q): return date(y,3*(q-1)+1,1)
def quarter_end(y,q): return date(y,12,31) if q==4 else quarter_start(y,q+1)-timedelta(days=1)
def last_quarter_bounds():
    t=pd.Timestamp.today().date(); q=(t.month-1)//3+1
    y,lq=(t.year-1,4) if q==1 else (t.year,q-1)
    return quarter_start(y,lq), quarter_end(y,lq)
def this_year_so_far_bounds(): t=pd.Timestamp.today().date(); return date(t.year,1,1),t

def date_range_from_preset(label, series: pd.Series, key_prefix: str):
    presets=["Today","This month so far","Last month","Last quarter","This year","Custom"]
    c1,c2 = st.columns([3,2])
    with c1:
        choice = st.radio(label, presets, horizontal=True, key=f"{key_prefix}_preset")
    with c2:
        default_grain = {"Today":"Day","This month so far":"Day","Last month":"Month",
                         "Last quarter":"Month","This year":"Month","Custom":"Month"}[choice]
        st.radio("Granularity", ["Day","Week","Month"], horizontal=True,
                 index=["Day","Week","Month"].index(default_grain), key=f"{key_prefix}_grain")
    if choice=="Today": f,t=today_bounds()
    elif choice=="This month so far": f,t=this_month_so_far_bounds()
    elif choice=="Last month": f,t=last_month_bounds()
    elif choice=="Last quarter": f,t=last_quarter_bounds()
    elif choice=="This year": f,t=this_year_so_far_bounds()
    else:
        dmin,dmax=safe_minmax_date(series)
        rng=st.date_input("Custom range",(dmin,dmax),key=f"{key_prefix}_custom")
        f,t=(rng[0],rng[1]) if isinstance(rng,(tuple,list)) and len(rng)==2 else (dmin,dmax)
    if f>t: st.error("'From' is after 'To'. Please adjust.")
    return f,t

def alt_line(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

def to_csv_bytes(df: pd.DataFrame)->bytes: return df.to_csv(index=False).encode("utf-8")

def group_label_from_series(s: pd.Series, grain_key: str):
    grain=st.session_state.get(grain_key,"Month")
    if grain=="Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain=="Week":
        iso=pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str)+"-W"+iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

# --------------------------- STATE / TOPBAR ---------------------------
if "filters_open" not in st.session_state: st.session_state["filters_open"]=True
if "show_b" not in st.session_state: st.session_state["show_b"]=False
if "csv_path" not in st.session_state: st.session_state["csv_path"]="Master_sheet_DB_10percent.csv"
if "uploaded_bytes" not in st.session_state: st.session_state["uploaded_bytes"]=None

def _toggle_filters_toggle(): st.session_state["filters_open"]=not st.session_state.get("filters_open",True)
def _enable_b(): st.session_state["show_b"]=True
def _disable_b(): st.session_state["show_b"]=False
def _select_all_cb(ms_key, all_key, options):
    st.session_state[ms_key]=options; st.session_state[all_key]=True
def _clear_cb(ms_key, all_key):
    st.session_state[ms_key]=[]; st.session_state[all_key]=False
def _reset_all_cb(): st.session_state.clear()
def _store_upload(key):
    up=st.session_state.get(key)
    if up is not None:
        st.session_state["uploaded_bytes"]=up.getvalue(); st.rerun()

with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([6,2,2,3])
    with c1: st.markdown('<div class="title">☰ MTD vs Cohort — Drawer UI + Predictability</div>', unsafe_allow_html=True)
    with c2: st.button("☰ Filters", key="toggle_filters", on_click=_toggle_filters_toggle, use_container_width=True)
    with c3:
        if st.session_state["show_b"]:
            st.button("Disable B", key="disable_b", on_click=_disable_b, use_container_width=True)
        else:
            st.button("Enable B", key="enable_b", on_click=_enable_b, use_container_width=True)
    with c4:
        cb1,cb2,cb3 = st.columns([1,1,1])
        with cb1: st.button("A→B", key="clone_ab_btn", on_click=_request_clone, args=("A2B",), use_container_width=True)
        with cb2: st.button("B→A", key="clone_ba_btn", on_click=_request_clone, args=("B2A",), use_container_width=True)
        with cb3: st.button("Reset", key="reset_all", on_click=_reset_all_cb, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- LOAD DATA ---------------------------
if st.session_state["uploaded_bytes"]:
    df = robust_read_csv(BytesIO(st.session_state["uploaded_bytes"]))
else:
    df = robust_read_csv(st.session_state["csv_path"])

# Trim / normalize column names
df.columns=[c.strip() for c in df.columns]

# Core validations
missing=[c for c in REQUIRED_COLS if c not in df.columns]
# we'll allow missing 'Pipeline' here because we will ensure it below
missing = [m for m in missing if m != "Pipeline"]
if missing:
    st.error(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")
    st.stop()

# Clean and normalize
df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
df["Create Date"]=pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"]=df["Create Date"].dt.to_period("M")

# Guarantee a usable Pipeline column (fixes your earlier warning)
df = ensure_program_column(df, target="Pipeline")

# Detect other date-like columns (e.g., Payment Received Date)
date_like_cols=detect_measure_date_columns(df)
if not date_like_cols:
    st.error("No date-like columns besides 'Create Date' (e.g., 'Payment Received Date').")
    st.stop()

# --------------------------- GLOBAL FILTER WIDGETS ---------------------------
def _summary(values, all_flag, max_items=2):
    vals=coerce_list(values)
    if all_flag: return "All"
    if not vals: return "None"
    s=", ".join(map(str, vals[:max_items]))
    if len(vals)>max_items: s+=f" +{len(vals)-max_items} more"
    return s

def unified_multifilter(label, df, colname, key_prefix):
    options=sorted([v for v in df[colname].dropna().astype(str).unique()])
    all_key=f"{key_prefix}_all"; ms_key=f"{key_prefix}_ms"
    if all_key not in st.session_state: st.session_state[all_key]=True
    if ms_key not in st.session_state: st.session_state[ms_key]=options.copy()
    stored=coerce_list(st.session_state.get(ms_key, []))
    selected=[v for v in stored if v in options]
    if selected!=stored: st.session_state[ms_key]=selected
    all_flag=bool(st.session_state[all_key])
    effective = options if all_flag else selected
    header=f"{label}: {_summary(effective, all_flag)}"
    ctx = st.popover(header) if hasattr(st,"popover") else st.expander(header, expanded=False)
    with ctx:
        left,right = st.columns([1,3])
        with left: st.checkbox("All", value=all_flag, key=all_key)
        with right:
            disabled=st.session_state[all_key]
            st.multiselect(label, options=options, default=selected, key=ms_key,
                           placeholder=f"Type to search {label.lower()}…",
                           label_visibility="collapsed", disabled=disabled)
            c1,c2 = st.columns(2)
            with c1: st.button("Select all", key=f"{key_prefix}_select_all", use_container_width=True,
                               on_click=_select_all_cb, args=(ms_key, all_key, options))
            with c2: st.button("Clear", key=f"{key_prefix}_clear", use_container_width=True,
                               on_click=_clear_cb, args=(ms_key, all_key))
    all_flag=bool(st.session_state[all_key])
    selected=[v for v in coerce_list(st.session_state.get(ms_key, [])) if v in options]
    effective = options if all_flag else selected
    return all_flag, effective, f"{label}: {_summary(effective, all_flag)}"

def scenario_filters_block(name: str, df: pd.DataFrame):
    st.markdown(f"**Scenario {name}** <span class='badge'>independent</span>", unsafe_allow_html=True)
    pipe_all, pipe_sel, s1 = unified_multifilter("Pipeline", df, "Pipeline", f"{name}_pipe")
    src_all,  src_sel,  s2 = unified_multifilter("Deal Source", df, "JetLearn Deal Source", f"{name}_src")
    ctry_all, ctry_sel, s3 = unified_multifilter("Country", df, "Country", f"{name}_ctry")
    cslr_all, cslr_sel, s4 = unified_multifilter("Counsellor", df, "Student/Academic Counsellor", f"{name}_cslr")
    st.markdown(f"<div class='popcap'>Filters — {s1} · {s2} · {s3} · {s4}</div>", unsafe_allow_html=True)

    mrow = st.columns([3,2])
    with mrow[0]:
        st.multiselect(f"[{name}] Measure date(s)", options=date_like_cols,
                       key=f"{name}_measures",
                       default=st.session_state.get(f"{name}_measures", [date_like_cols[0]]))
    with mrow[1]:
        st.radio(f"[{name}] Mode", ["MTD","Cohort","Both"], horizontal=True, key=f"{name}_mode",
                 index=["MTD","Cohort","Both"].index(st.session_state.get(f"{name}_mode","Both")))

    mode=st.session_state.get(f"{name}_mode","Both")
    if mode in ("MTD","Both"):
        st.caption("Create-Date window (MTD)")
        date_range_from_preset(f"[{name}] MTD Range", df["Create Date"], f"{name}_mtd")
    if mode in ("Cohort","Both"):
        st.caption("Measure-Date window (Cohort)")
        meas=st.session_state.get(f"{name}_measures", [])
        series=df[meas[0]] if meas else df["Create Date"]
        date_range_from_preset(f"[{name}] Cohort Range", series, f"{name}_coh")

    st.markdown("---")
    st.toggle(f"[{name}] Top 5 Countries", value=st.session_state.get(f"{name}_top_ctry", True), key=f"{name}_top_ctry")
    st.toggle(f"[{name}] Top 3 Deal Sources", value=st.session_state.get(f"{name}_top_src", True), key=f"{name}_top_src")
    st.toggle(f"[{name}] Top 5 Counsellors", value=st.session_state.get(f"{name}_top_cslr", False), key=f"{name}_top_cslr")
    st.multiselect(f"[{name}] Split by", ["JetLearn Deal Source","Country","Student/Academic Counsellor"],
                   key=f"{name}_split", default=st.session_state.get(f"{name}_split", []))
    st.toggle(f"[{name}] Country × Deal Source (Top 10)", value=st.session_state.get(f"{name}_pair", False), key=f"{name}_pair")

def ensure_month_cols(base: pd.DataFrame, measures):
    for m in measures:
        col=f"{m}_Month"
        if m in base.columns and col not in base.columns:
            base[col]=base[m].dt.to_period("M")
    return base

def assemble_meta(name: str, df: pd.DataFrame):
    pipe_all=st.session_state.get(f"{name}_pipe_all", True); pipe_sel=st.session_state.get(f"{name}_pipe_ms", [])
    src_all=st.session_state.get(f"{name}_src_all", True);  src_sel=st.session_state.get(f"{name}_src_ms", [])
    ctry_all=st.session_state.get(f"{name}_ctry_all", True); ctry_sel=st.session_state.get(f"{name}_ctry_ms", [])
    cslr_all=st.session_state.get(f"{name}_cslr_all", True); cslr_sel=st.session_state.get(f"{name}_cslr_ms", [])
    mask = (in_filter(df["Pipeline"], pipe_all, pipe_sel) &
            in_filter(df["JetLearn Deal Source"], src_all, src_sel) &
            in_filter(df["Country"], ctry_all, ctry_sel) &
            in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel))
    base=df[mask].copy()
    measures=st.session_state.get(f"{name}_measures", [])
    base=ensure_month_cols(base, measures)
    mode=st.session_state.get(f"{name}_mode","Both")
    mtd = mode in ("MTD","Both"); cohort = mode in ("Cohort","Both")
    def fetch_range(kind):
        preset=st.session_state.get(f"{name}_{kind}_preset","This month so far")
        series = base["Create Date"] if kind=="mtd" else (base[measures[0]] if measures else base["Create Date"])
        if preset=="Today": f,t=today_bounds()
        elif preset=="This month so far": f,t=this_month_so_far_bounds()
        elif preset=="Last month": f,t=last_month_bounds()
        elif preset=="Last quarter": f,t=last_quarter_bounds()
        elif preset=="This year": f,t=this_year_so_far_bounds()
        else:
            dmin,dmax=safe_minmax_date(series); custom=st.session_state.get(f"{name}_{kind}_custom",(dmin,dmax))
            f,t = custom if isinstance(custom,(tuple,list)) and len(custom)==2 else (dmin,dmax)
        return f,t
    mtd_from=mtd_to=None; coh_from=coh_to=None
    if mtd: mtd_from,mtd_to=fetch_range("mtd")
    if cohort: coh_from,coh_to=fetch_range("coh")
    return dict(
        name=name, base=base, measures=measures, mtd=mtd, cohort=cohort,
        mtd_from=mtd_from, mtd_to=mtd_to, coh_from=coh_from, coh_to=coh_to,
        split_dims=st.session_state.get(f"{name}_split", []),
        show_top_countries=st.session_state.get(f"{name}_top_ctry", True),
        show_top_sources=st.session_state.get(f"{name}_top_src", True),
        show_top_counsellors=st.session_state.get(f"{name}_top_cslr", False),
        show_combo_pairs=st.session_state.get(f"{name}_pair", False),
        pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
        ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel
    )

def compute_outputs(meta):
    base=meta["base"]; measures=meta["measures"]
    mtd=meta["mtd"]; cohort=meta["cohort"]
    mtd_from, mtd_to = meta["mtd_from"], meta["mtd_to"]
    coh_from, coh_to = meta["coh_from"], meta["coh_to"]
    split_dims=meta["split_dims"]
    show_top_countries=meta["show_top_countries"]
    show_top_sources=meta["show_top_sources"]
    show_top_counsellors=meta["show_top_counsellors"]
    show_combo_pairs=meta["show_combo_pairs"]
    metrics_rows, tables, charts = [], {}, {}

    if mtd and mtd_from and mtd_to and measures:
        in_cre=base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
        sub=base[in_cre].copy()
        flags=[]
        for m in measures:
            if m not in sub.columns: continue
            flg=f"__MTD__{m}"
            sub[flg]=((sub[m].notna()) & (sub[f"{m}_Month"]==sub["Create_Month"])).astype(int)
            flags.append(flg)
            metrics_rows.append({"Scope":"MTD","Metric":f"Count on '{m}'","Window":f"{mtd_from} → {mtd_to}","Value":int(sub[flg].sum())})
        metrics_rows.append({"Scope":"MTD","Metric":"Create Count in window","Window":f"{mtd_from} → {mtd_to}","Value":int(len(sub))})
        if flags:
            if split_dims:
                sub["_CreateCount"]=1
                grp=sub.groupby(split_dims, dropna=False)[flags+["_CreateCount"]].sum().reset_index()
                rename_map={"_CreateCount":"Create Count in window"}
                for f,m in zip(flags,measures): rename_map[f]=f"MTD: {m}"
                grp=grp.rename(columns=rename_map).sort_values(by=f"MTD: {measures[0]}", ascending=False)
                tables[f"MTD split by {', '.join(split_dims)}"]=grp
            if show_top_countries and "Country" in sub.columns:
                g=sub.groupby("Country", dropna=False)[flags].sum().reset_index()
                g=g.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                tables["Top 5 Countries — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)
            if show_top_sources and "JetLearn Deal Source" in sub.columns:
                g=sub.groupby("JetLearn Deal Source", dropna=False)[flags].sum().reset_index()
                g=g.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                tables["Top 3 Deal Sources — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(3)
            if show_top_counsellors and "Student/Academic Counsellor" in sub.columns:
                g=sub.groupby("Student/Academic Counsellor", dropna=False)[flags].sum().reset_index()
                g=g.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                tables["Top 5 Counsellors — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)
            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(sub.columns):
                both=sub.groupby(["Country","JetLearn Deal Source"], dropna=False)[flags].sum().reset_index()
                both=both.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                tables["Top Country × Deal Source — MTD"]=both.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(10)
            trend=sub.copy()
            trend["Bucket"]=group_label_from_series(trend["Create Date"], f"{meta['name']}_mtd_grain")
            t=trend.groupby("Bucket")[flags].sum().reset_index()
            t=t.rename(columns={f:m for f,m in zip(flags,measures)})
            long=t.melt(id_vars="Bucket", var_name="Measure", value_name="Count")
            charts["MTD Trend"]=alt_line(long,"Bucket:O","Count:Q",color="Measure:N",tooltip=["Bucket","Measure","Count"])

    if cohort and coh_from and coh_to and measures:
        tmp=base.copy(); ch_flags=[]
        for m in measures:
            if m not in tmp.columns: continue
            flg=f"__COH__{m}"
            tmp[flg]=tmp[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both").astype(int)
            ch_flags.append(flg)
            metrics_rows.append({"Scope":"Cohort","Metric":f"Count on '{m}'","Window":f"{coh_from} → {coh_to}","Value":int(tmp[flg].sum())})
        in_cre_coh=base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
        metrics_rows.append({"Scope":"Cohort","Metric":"Create Count in Cohort window","Window":f"{coh_from} → {coh_to}","Value":int(in_cre_coh.sum())})
        if ch_flags:
            if split_dims:
                tmp["_CreateInCohort"]=in_cre_coh.astype(int)
                grp2=tmp.groupby(split_dims, dropna=False)[ch_flags+["_CreateInCohort"]].sum().reset_index()
                rename_map2={"_CreateInCohort":"Create Count in Cohort window"}
                for f,m in zip(ch_flags,measures): rename_map2[f]=f"Cohort: {m}"
                grp2=grp2.rename(columns=rename_map2).sort_values(by=f"Cohort: {measures[0]}", ascending=False)
                tables[f"Cohort split by {', '.join(split_dims)}"]=grp2
            if show_top_countries and "Country" in base.columns:
                g=tmp.groupby("Country", dropna=False)[ch_flags].sum().reset_index()
                g=g.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                tables["Top 5 Countries — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)
            if show_top_sources and "JetLearn Deal Source" in base.columns:
                g=tmp.groupby("JetLearn Deal Source", dropna=False)[ch_flags].sum().reset_index()
                g=g.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                tables["Top 3 Deal Sources — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(3)
            if show_top_counsellors and "Student/Academic Counsellor" in base.columns:
                g=tmp.groupby("Student/Academic Counsellor", dropna=False)[ch_flags].sum().reset_index()
                g=g.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                tables["Top 5 Counsellors — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)
            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(base.columns):
                both2=tmp.groupby(["Country","JetLearn Deal Source"], dropna=False)[ch_flags].sum().reset_index()
                both2=both2.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                tables["Top Country × Deal Source — Cohort"]=both2.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(10)
            frames=[]
            for m in measures:
                mask=base[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
                sel=base.loc[mask,[m]].copy()
                if sel.empty: continue
                sel["Bucket"]=group_label_from_series(sel[m], f"{meta['name']}_coh_grain")
                t=sel.groupby("Bucket")[m].count().reset_index(name="Count"); t["Measure"]=m; frames.append(t)
            if frames:
                trend=pd.concat(frames, ignore_index=True)
                charts["Cohort Trend"]=alt_line(trend,"Bucket:O","Count:Q",color="Measure:N",tooltip=["Bucket","Measure","Count"])

    return metrics_rows, tables, charts

def kpi_grid(dfk, label_prefix=""):
    if dfk.empty: st.info("No KPIs yet."); return
    cols=st.columns(4)
    for i,row in dfk.iterrows():
        with cols[i%4]:
            st.markdown(f"""
<div class="kpi">
  <div class="label">{label_prefix}{row['Scope']} — {row['Metric']}</div>
  <div class="value">{row['Value']:,}</div>
  <div class="delta">{row['Window']}</div>
</div>
""", unsafe_allow_html=True)

def build_compare_delta(dfA, dfB):
    if dfA.empty or dfB.empty: return pd.DataFrame()
    key=["Scope","Metric"]
    a=dfA[key+["Value"]].copy().rename(columns={"Value":"A"}); a["A"]=pd.to_numeric(a["A"], errors="coerce")
    b=dfB[key+["Value"]].copy().rename(columns={"Value":"B"}); b["B"]=pd.to_numeric(b["B"], errors="coerce")
    out=pd.merge(a,b,on=key,how="inner")
    out["Δ"]=pd.to_numeric(out["B"]-out["A"], errors="coerce")
    denom=out["A"].astype(float); denom=denom.where(~(denom.isna() | (denom==0)))
    out["Δ%"]=((out["Δ"].astype(float)/denom)*100).round(1)
    return out

# --------------------------- PREDICTABILITY HELPERS ---------------------------
def subset_for_program(df: pd.DataFrame, program_choice: str) -> pd.DataFrame:
    """Filter by Pipeline values for AI-Coding / Math / Both, tolerant to variants."""
    if "Pipeline" not in df.columns:
        return df
    if program_choice == "Both":
        return df

    s = df["Pipeline"].astype(str).str.strip().str.lower()
    if program_choice == "AI-Coding":
        mask = (s.eq("ai-coding") | s.str.contains(r"\bai\b") | s.str.contains("coding"))
    elif program_choice == "Math":
        mask = (s.eq("math") | s.str.contains("math") | s.str.contains("vedic"))
    else:
        mask = pd.Series(True, index=df.index)

    sub = df[mask].copy()
    if sub.empty:
        st.warning(f"No rows matched **{program_choice}** in Pipeline. Showing all records instead.")
        return df
    return sub

def build_daily_counts(df: pd.DataFrame, measure_col: str):
    """Aggregate daily counts per JetLearn Deal Source (treat NaT as 0)."""
    if measure_col not in df.columns:
        return pd.DataFrame(columns=["ds","JetLearn Deal Source","y"])
    data = df.dropna(subset=[measure_col]).copy()
    data["ds"] = pd.to_datetime(data[measure_col]).dt.date
    grp = (data.groupby(["ds","JetLearn Deal Source"])
                .size()
                .reset_index(name="y"))
    return grp

def add_time_features(frame: pd.DataFrame):
    f = frame.copy()
    f["ds"] = pd.to_datetime(f["ds"])
    f["dow"] = f["ds"].dt.dayofweek
    f["month"] = f["ds"].dt.month
    f["dom"] = f["ds"].dt.day
    f["week"] = f["ds"].dt.isocalendar().week.astype(int)
    f["year"] = f["ds"].dt.year
    return f

def add_lags_and_rolls(frame: pd.DataFrame, group_col="JetLearn Deal Source"):
    """Create lag/rolling features per source."""
    f = frame.sort_values(["JetLearn Deal Source","ds"]).copy()
    f["y_lag7"] = f.groupby(group_col)["y"].shift(7)
    f["y_lag14"] = f.groupby(group_col)["y"].shift(14)
    f["y_roll7"] = f.groupby(group_col)["y"].rolling(7, min_periods=1).mean().reset_index(0,drop=True)
    f["y_roll28"] = f.groupby(group_col)["y"].rolling(28, min_periods=1).mean().reset_index(0,drop=True)
    return f

def train_forecast_model(df_counts: pd.DataFrame):
    """Fit a simple GBM with one-hot on source; time-aware split."""
    if df_counts.empty or df_counts["ds"].nunique() < 40:
        return None, None, None  # not enough history

    X = add_time_features(df_counts)
    X = add_lags_and_rolls(X)
    X = X.dropna().reset_index(drop=True)
    if X.empty:
        return None, None, None

    # Features and target
    features = ["dow","month","dom","week","year","y_lag7","y_lag14","y_roll7","y_roll28","JetLearn Deal Source"]
    target = "y"
    cat_cols = ["JetLearn Deal Source"]

    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    model = SKPipeline(steps=[
        ("prep", ct),
        ("gbm", GradientBoostingRegressor(random_state=42))
    ])

    # Time-aware split (last 20% for test)
    X = X.sort_values("ds")
    split_idx = int(len(X) * 0.8)
    train, test = X.iloc[:split_idx], X.iloc[split_idx:]
    if train.empty or test.empty:
        return None, None, None

    model.fit(train[features], train[target])
    pred = model.predict(test[features]).clip(min=0)
    mae = mean_absolute_error(test[target], pred)
    mape = float(np.mean(np.abs((test[target].values - pred) / np.maximum(1, test[target].values))) * 100)

    metrics = {"MAE": round(mae, 2), "MAPE%": round(mape, 1), "train_n": len(train), "test_n": len(test)}
    return model, features, metrics

def make_future_frame(horizon: str):
    """Create future dates for forecasting horizon."""
    today = pd.Timestamp.today().date()
    if horizon == "Today":
        dates = [today]
    elif horizon == "Tomorrow":
        dates = [today + timedelta(days=1)]
    elif horizon == "This month (rest)":
        start = today
        end = (date(today.year + (today.month==12), 1 if today.month==12 else today.month+1, 1) - timedelta(days=1))
        dates = pd.date_range(start, end, freq="D").date.tolist()
    elif horizon == "Next month":
        start = date(today.year + (today.month==12), 1 if today.month==12 else today.month+1, 1)
        end = date(start.year + (start.month==12), 1 if start.month==12 else start.month+1, 1) - timedelta(days=1)
        dates = pd.date_range(start, end, freq="D").date.tolist()
    else:
        dates = [today]
    return pd.DataFrame({"ds": dates})

def predict_by_source(model, feature_cols, sources, future_dates):
    frames=[]
    for s in sources:
        f = future_dates.copy()
        f["JetLearn Deal Source"] = s
        f = add_time_features(f)
        # add naive lag/roll placeholders (use seasonal averages if needed -> zeros here)
        for c in ["y_lag7","y_lag14","y_roll7","y_roll28"]:
            f[c] = np.nan  # will be treated as missing -> we fill with 0
        f = f.fillna(0)
        yhat = model.predict(f[feature_cols]).clip(min=0)
        out = f[["ds","JetLearn Deal Source"]].copy()
        out["yhat"] = yhat
        frames.append(out)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["ds","JetLearn Deal Source","yhat"])

# --------------------------- SIDEBAR (Drawer) ---------------------------
if st.session_state.get("filters_open", True):
    with st.sidebar:
        with st.expander("Scenario A controls", expanded=True):
            scenario_filters_block("A", df)
        if st.session_state["show_b"]:
            with st.expander("Scenario B controls", expanded=False):
                scenario_filters_block("B", df)
        with st.expander("Data source (optional)", expanded=False):
            st.caption("Upload a CSV or provide a file path. If both are given, upload takes precedence.")
            st.file_uploader("Upload CSV", type=["csv"], key="__uploader__", on_change=_store_upload, args=("__uploader__",))
            st.text_input("CSV path", key="csv_path")

# --------------------------- ANALYZE (A/B) ---------------------------
def mk_caption(meta):
    return (f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
            f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(coerce_list(meta['pipe_sel'])) or 'None'} · "
            f"Deal Source: {'All' if meta['src_all'] else ', '.join(coerce_list(meta['src_sel'])) or 'None'} · "
            f"Country: {'All' if meta['ctry_all'] else ', '.join(coerce_list(meta['ctry_sel'])) or 'None'} · "
            f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(coerce_list(meta['cslr_sel'])) or 'None'}")

# --------------------------- MAIN TABS ---------------------------
main_tabs = st.tabs(["Analyze", "Predictability"])

with main_tabs[0]:
    metaA = assemble_meta("A", df)
    with st.spinner("Crunching scenario A…"):
        metricsA, tablesA, chartsA = compute_outputs(metaA)

    if st.session_state["show_b"]:
        metaB = assemble_meta("B", df)
        with st.spinner("Crunching scenario B…"):
            metricsB, tablesB, chartsB = compute_outputs(metaB)

    if st.session_state["show_b"]:
        tabA, tabB, tabC = st.tabs(["Scenario A", "Scenario B", "Compare"])
    else:
        tabA, = st.tabs(["Scenario A"])

    with tabA:
        st.markdown("<div class='section-title'>📌 KPI Overview — A</div>", unsafe_allow_html=True)
        dfA=pd.DataFrame(metricsA); kpi_grid(dfA, "A · ")
        st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — A</div>", unsafe_allow_html=True)
        if not tablesA: st.info("No tables — open Filters and enable splits/leaderboards.")
        else:
            for name,frame in tablesA.items():
                st.subheader(name); st.dataframe(frame, use_container_width=True)
                st.download_button("Download CSV — "+name, to_csv_bytes(frame),
                                   file_name=f"A_{name.replace(' ','_')}.csv", mime="text/csv")
        st.markdown("<div class='section-title'>📈 Trends — A</div>", unsafe_allow_html=True)
        if "MTD Trend" in chartsA: st.altair_chart(chartsA["MTD Trend"], use_container_width=True)
        if "Cohort Trend" in chartsA: st.altair_chart(chartsA["Cohort Trend"], use_container_width=True)
        st.caption("**Scenario A** — " + mk_caption(metaA))
        st.caption("Excluded globally: 1.2 Invalid Deal")

    if st.session_state["show_b"]:
        with tabB:
            st.markdown("<div class='section-title'>📌 KPI Overview — B</div>", unsafe_allow_html=True)
            dfB=pd.DataFrame(metricsB); kpi_grid(dfB, "B · ")
            st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — B</div>", unsafe_allow_html=True)
            if not tablesB: st.info("No tables — open Filters and enable splits/leaderboards.")
            else:
                for name,frame in tablesB.items():
                    st.subheader(name); st.dataframe(frame, use_container_width=True)
                    st.download_button("Download CSV — "+name, to_csv_bytes(frame),
                                       file_name=f"B_{name.replace(' ','_')}.csv", mime="text/csv")
            st.markdown("<div class='section-title'>📈 Trends — B</div>", unsafe_allow_html=True)
            if "MTD Trend" in chartsB: st.altair_chart(chartsB["MTD Trend"], use_container_width=True)
            if "Cohort Trend" in chartsB: st.altair_chart(chartsB["Cohort Trend"], use_container_width=True)
            st.caption("**Scenario B** — " + mk_caption(metaB))
            st.caption("Excluded globally: 1.2 Invalid Deal")

        with tabC:
            st.markdown("<div class='section-title'>🧠 Smart Compare (A vs B)</div>", unsafe_allow_html=True)
            dfA=pd.DataFrame(metricsA); dfB=pd.DataFrame(metricsB)
            if dfA.empty or dfB.empty: st.info("Turn on KPIs for both scenarios to enable compare.")
            else:
                cmp=build_compare_delta(dfA, dfB)
                if cmp.empty: st.info("Adjust scenarios to produce comparable KPIs.")
                else:
                    st.dataframe(cmp, use_container_width=True)
                    try:
                        if set(metaA["measures"]) == set(metaB["measures"]):
                            sub=cmp[cmp["Metric"].str.startswith("Count on '")].copy()
                            if not sub.empty:
                                sub["Measure"]=sub["Metric"].str.extract(r"Count on '(.+)'")
                                a_long=sub.rename(columns={"A":"Value"})[["Measure","Scope","Value"]]; a_long["Scenario"]="A"
                                b_long=sub.rename(columns={"B":"Value"})[["Measure","Scope","Value"]]; b_long["Scenario"]="B"
                                long=pd.concat([a_long,b_long], ignore_index=True)
                                ch=alt.Chart(long).mark_bar().encode(
                                    x=alt.X("Scope:N", title=None), y=alt.Y("Value:Q"),
                                    color=alt.Color("Scenario:N", scale=alt.Scale(range=PALETTE[:2])),
                                    column=alt.Column("Measure:N", header=alt.Header(title=None, labelAngle=0)),
                                    tooltip=["Measure","Scenario","Scope","Value"]
                                ).properties(height=260)
                                st.altair_chart(ch, use_container_width=True)
                    except Exception:
                        pass

# --------------------------- PREDICTABILITY TAB ---------------------------
with main_tabs[1]:
    st.markdown("### 🔮 Predictability (JetLearn Deal Source totals)")
    st.caption("Train a simple model on historical **daily** counts (per deal source), then forecast by horizon. "
               "Program filter uses the **Pipeline** column (robust to naming variants).")

    # Controls
    pc1, pc2, pc3 = st.columns([2,2,2])
    with pc1:
        program_choice = st.selectbox("Program", ["AI-Coding", "Math", "Both"], index=2)
    with pc2:
        # Choose the date column representing an "enrollment" / event (default Payment Received Date if present)
        default_measure = "Payment Received Date" if "Payment Received Date" in date_like_cols else date_like_cols[0]
        pred_measure = st.selectbox("Event date column to count", date_like_cols, index=date_like_cols.index(default_measure))
    with pc3:
        horizon = st.selectbox("Forecast horizon", ["Today","Tomorrow","This month (rest)","Next month"], index=2)

    run_btn = st.button("Run prediction", type="primary")
    if run_btn:
        # Apply program subset safely
        df_prog = subset_for_program(df, program_choice)

        # Build daily counts by deal source
        daily = build_daily_counts(df_prog, pred_measure)
        if daily.empty:
            st.warning("Not enough data in the chosen event column to model. Try a different measure.")
        else:
            # Train model
            with st.spinner("Training model & backtesting..."):
                model, feats, back_metrics = train_forecast_model(daily)

            # If model is None, fallback to naive average (last 7 days)
            if model is None:
                st.info("Not enough history to train a model. Falling back to a 7-day average baseline.")
                # naive per source
                naive = daily.groupby("JetLearn Deal Source").tail(7).groupby("JetLearn Deal Source")["y"].mean().reset_index()
                naive = naive.rename(columns={"y":"naive_mean"})
                future_dates = make_future_frame(horizon)
                if future_dates.empty:
                    st.info("No future dates formed for the selected horizon.")
                else:
                    # Expand across sources
                    sources = sorted(daily["JetLearn Deal Source"].unique())
                    out = pd.MultiIndex.from_product([future_dates["ds"], sources], names=["ds","JetLearn Deal Source"]).to_frame(index=False)
                    out = out.merge(naive, on="JetLearn Deal Source", how="left").fillna(0)
                    out["yhat"] = out["naive_mean"].clip(lower=0).round(2)
                    # Present
                    st.markdown("#### Forecast (Naive baseline)")
                    st.dataframe(out[["ds","JetLearn Deal Source","yhat"]], use_container_width=True)
                    total = out.groupby("ds")["yhat"].sum().reset_index()
                    ch = alt.Chart(total).mark_bar().encode(x="ds:T", y="yhat:Q", tooltip=["ds","yhat"])
                    st.altair_chart(ch, use_container_width=True)
            else:
                # Show backtest metrics
                st.markdown("#### Backtest (time-aware split)")
                st.write(pd.DataFrame([back_metrics]))
                # Forecast with GBM
                future_dates = make_future_frame(horizon)
                sources = sorted(daily["JetLearn Deal Source"].unique())
                fc = predict_by_source(model, feats, sources, future_dates)
                if fc.empty:
                    st.info("Could not produce forecast for the selected horizon.")
                else:
                    fc["yhat"] = fc["yhat"].round(2)
                    st.markdown("#### Forecast by Deal Source")
                    st.dataframe(fc, use_container_width=True)
                    totals = fc.groupby("ds")["yhat"].sum().reset_index()
                    st.markdown("#### Total Forecast")
                    st.dataframe(totals, use_container_width=True)
                    ch = alt.Chart(totals).mark_line(point=True).encode(
                        x="ds:T", y="yhat:Q", tooltip=["ds","yhat"]
                    ).properties(height=260)
                    st.altair_chart(ch, use_container_width=True)

    with st.expander("Notes", expanded=False):
        st.markdown("""
- **Counts** are daily occurrences in the chosen event date column (e.g. *Payment Received Date*), grouped by **JetLearn Deal Source**.
- The model uses calendar features (DOW, week, month), plus simple lags/rolling means; it’s trained with a time-aware split.
- If there isn’t enough history, the app falls back to a **7-day mean** per source.
- ‘Program’ is read from the **Pipeline** column; common synonyms (Program/Programme/Course/Track) are auto-detected.
        """)
