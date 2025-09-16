# app.py — Drawer UI + Analytics + Predictability (Pipeline-aware)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta, datetime
from typing import List, Tuple, Dict, Optional

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------- Page & Styles --------------------
st.set_page_config(page_title="JetLearn MIS — MTD/Cohort & Predictability", layout="wide", page_icon="✨")
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

PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]
REQUIRED_COLS = ["Pipeline", "JetLearn Deal Source", "Country",
                 "Student/Academic Counsellor", "Deal Stage", "Create Date"]

PROGRAM_COL = "Pipeline"  # predictability "program" comes from Pipeline

# -------------------- Utilities --------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try: return pd.read_csv(file_or_path, encoding=enc)
        except Exception: pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame) -> List[str]:
    date_like=[]
    for col in df.columns:
        if col=="Create Date": continue
        low = col.lower()
        if any(k in low for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum()>0:
                df[col]=parsed; date_like.append(col)
    # prefer Payment Received Date first if exists
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c!="Payment Received Date"]
    return date_like

def coerce_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
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

def alt_line(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

def to_csv_bytes(df: pd.DataFrame)->bytes: return df.to_csv(index=False).encode("utf-8")

# -------------------- Program/Pipeline mapping for Predictability --------------------
def _is_ai_coding(text: str) -> bool:
    t = str(text).strip().lower()
    return (t == "ai-coding") or ("ai" in t) or ("coding" in t)

def _is_math(text: str) -> bool:
    t = str(text).strip().lower()
    return (t == "math") or ("vedic" in t) or ("math" in t)

def subset_for_program(df: pd.DataFrame, program_choice: str) -> pd.DataFrame:
    if program_choice == "Both" or PROGRAM_COL not in df.columns:
        if PROGRAM_COL not in df.columns:
            st.info(f"ℹ️ Using all records because `{PROGRAM_COL}` column wasn’t found.")
        return df
    s = df[PROGRAM_COL].astype(str)
    if program_choice == "AI-Coding":
        mask = s.apply(_is_ai_coding)
    elif program_choice == "Math":
        mask = s.apply(_is_math)
    else:
        mask = pd.Series(True, index=df.index)
    sub = df[mask].copy()
    if sub.empty:
        st.warning(f"⚠️ No rows matched **{program_choice}** in `{PROGRAM_COL}`. Using all rows instead.")
        return df
    return sub

# -------------------- Clone helpers --------------------
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

# -------------------- State defaults --------------------
if "filters_open" not in st.session_state: st.session_state["filters_open"]=True
if "show_b" not in st.session_state: st.session_state["show_b"]=False
if "csv_path" not in st.session_state: st.session_state["csv_path"]="Master_sheet_DB_10percent.csv"
if "uploaded_bytes" not in st.session_state: st.session_state["uploaded_bytes"]=None
_perform_clone_if_requested()

# -------------------- Top bar --------------------
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([5,2.2,2.5,2.5])
    with c1: st.markdown('<div class="title">☰ JetLearn MIS — MTD/Cohort & Predictability</div>', unsafe_allow_html=True)
    with c2:
        st.button("☰ Filters", key="toggle_filters",
                  on_click=lambda: st.session_state.update(filters_open=not st.session_state.get("filters_open",True)),
                  use_container_width=True)
    with c3:
        if st.session_state["show_b"]:
            st.button("Disable B", key="disable_b",
                      on_click=lambda: st.session_state.update(show_b=False),
                      use_container_width=True)
        else:
            st.button("Enable B", key="enable_b",
                      on_click=lambda: st.session_state.update(show_b=True),
                      use_container_width=True)
    with c4:
        cb1,cb2,cb3 = st.columns([1,1,1])
        with cb1: st.button("A→B", key="clone_ab_btn", on_click=_request_clone, args=("A2B",), use_container_width=True)
        with cb2: st.button("B→A", key="clone_ba_btn", on_click=_request_clone, args=("B2A",), use_container_width=True)
        with cb3: st.button("Reset", key="reset_all", on_click=lambda: st.session_state.clear(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Data --------------------
if st.session_state["uploaded_bytes"]:
    df = robust_read_csv(BytesIO(st.session_state["uploaded_bytes"]))
else:
    df = robust_read_csv(st.session_state["csv_path"])
df.columns=[c.strip() for c in df.columns]
missing=[c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}"); st.stop()
df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
df["Create Date"]=pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"]=df["Create Date"].dt.to_period("M")
date_like_cols=detect_measure_date_columns(df)
if not date_like_cols:
    st.error("No date-like columns besides 'Create Date' (e.g., 'Payment Received Date')."); st.stop()

# -------------------- Filter widgets --------------------
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
                               on_click=lambda: (st.session_state.__setitem__(ms_key, options),
                                                 st.session_state.__setitem__(all_key, True)))
            with c2: st.button("Clear", key=f"{key_prefix}_clear", use_container_width=True,
                               on_click=lambda: (st.session_state.__setitem__(ms_key, []),
                                                 st.session_state.__setitem__(all_key, False)))
    all_flag=bool(st.session_state[all_key])
    selected=[v for v in coerce_list(st.session_state.get(ms_key, [])) if v in options]
    effective = options if all_flag else selected
    return all_flag, effective, f"{label}: {_summary(effective, all_flag)}"

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

def group_label_from_series(s: pd.Series, grain_key: str):
    grain=st.session_state.get(grain_key,"Month")
    if grain=="Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain=="Week":
        iso=pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str)+"-W"+iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

def ensure_month_cols(base: pd.DataFrame, measures: List[str]):
    for m in measures:
        col=f"{m}_Month"
        if m in base.columns and col not in base.columns:
            base[col]=base[m].dt.to_period("M")
    return base

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

def assemble_meta(name: str, df: pd.DataFrame) -> Dict:
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

def compute_outputs(meta: Dict):
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

    # MTD
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

    # Cohort
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

def mk_caption(meta):
    return (f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
            f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(coerce_list(meta['pipe_sel'])) or 'None'} · "
            f"Deal Source: {'All' if meta['src_all'] else ', '.join(coerce_list(meta['src_sel'])) or 'None'} · "
            f"Country: {'All' if meta['ctry_all'] else ', '.join(coerce_list(meta['ctry_sel'])) or 'None'} · "
            f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(coerce_list(meta['cslr_sel'])) or 'None'}")

# -------------------- Sidebar drawer --------------------
def store_upload(key):
    up=st.session_state.get(key)
    if up is not None:
        st.session_state["uploaded_bytes"]=up.getvalue(); st.rerun()

if st.session_state.get("filters_open", True):
    with st.sidebar:
        with st.expander("Scenario A controls", expanded=True):
            scenario_filters_block("A", df)
        if st.session_state["show_b"]:
            with st.expander("Scenario B controls", expanded=False):
                scenario_filters_block("B", df)
        with st.expander("Data source (optional)", expanded=False):
            st.caption("Upload a CSV or provide a file path. If both are given, upload takes precedence.")
            st.file_uploader("Upload CSV", type=["csv"], key="__uploader__", on_change=store_upload, args=("__uploader__",))
            st.text_input("CSV path", key="csv_path")

# -------------------- Compute A/B --------------------
metaA = assemble_meta("A", df)
with st.spinner("Crunching scenario A…"):
    metricsA, tablesA, chartsA = compute_outputs(metaA)

if st.session_state["show_b"]:
    metaB = assemble_meta("B", df)
    with st.spinner("Crunching scenario B…"):
        metricsB, tablesB, chartsB = compute_outputs(metaB)

# -------------------- Tabs --------------------
if st.session_state["show_b"]:
    tabA, tabB, tabC, tabP = st.tabs(["Scenario A", "Scenario B", "Compare", "Predictability"])
else:
    tabA, tabP = st.tabs(["Scenario A", "Predictability"])

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

# -------------------- Predictability Tab --------------------
def _resample_counts(df_in: pd.DataFrame, date_col: str, grain: str, by_source: bool):
    # grain -> Pandas rule
    rule = "D" if grain=="Day" else ("W-MON" if grain=="Week" else "MS")
    cols = [date_col]
    if by_source: cols += ["JetLearn Deal Source"]
    s = df_in[cols].dropna(subset=[date_col]).copy()
    s["_dt"] = pd.to_datetime(s[date_col])
    if by_source:
        g = (s.groupby([pd.Grouper(key="_dt", freq=rule), "JetLearn Deal Source"])
               .size().reset_index(name="y"))
        # complete each source's timeline
        frames=[]
        for src, sub in g.groupby("JetLearn Deal Source"):
            full_idx = pd.date_range(sub["_dt"].min(), sub["_dt"].max(), freq=rule)
            sub = sub.set_index("_dt").reindex(full_idx).fillna(0).rename_axis("_dt").reset_index()
            sub["JetLearn Deal Source"]=src
            frames.append(sub)
        g = pd.concat(frames, ignore_index=True)
    else:
        g = (s.groupby(pd.Grouper(key="_dt", freq=rule))
               .size().reset_index(name="y"))
        full_idx = pd.date_range(g["_dt"].min(), g["_dt"].max(), freq=rule)
        g = g.set_index("_dt").reindex(full_idx).fillna(0).rename_axis("_dt").reset_index()
    g["y"] = g["y"].astype(float)
    return g, rule

def _lag_features(frame: pd.DataFrame, grain: str) -> pd.DataFrame:
    # simple supervised lags + time features
    lags = [1,2,3]
    if grain=="Day": lags += [7,14,28]
    dfX = frame.copy()
    for L in lags:
        dfX[f"lag_{L}"] = dfX["y"].shift(L)
    # time features
    dt = pd.to_datetime(dfX["_dt"])
    dfX["month"] = dt.dt.month
    if grain=="Day":
        dfX["dow"]   = dt.dt.dayofweek
        dfX["week"]  = dt.dt.isocalendar().week.astype(int)
    elif grain=="Week":
        dfX["week"]  = dt.dt.isocalendar().week.astype(int)
    dfX = dfX.dropna().reset_index(drop=True)
    return dfX

def _walk_forward_backtest(dfX: pd.DataFrame, n_splits: int = 5) -> Tuple[float,float,float,float]:
    # rolling CV (de-biased)
    if len(dfX) < 40:  # small data guard
        n_splits = max(2, min(4, len(dfX)//10))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, mapes = [], []
    X = dfX.drop(columns=["y","_dt"])
    y = dfX["y"].values
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        pipe = SkPipeline([
            ("sc", StandardScaler(with_mean=False)),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
        ])
        pipe.fit(Xtr, ytr)
        pred = np.maximum(0, pipe.predict(Xte))  # non-negative
        maes.append(mean_absolute_error(yte, pred))
        denom = np.where(yte==0, np.nan, yte)
        mape = np.nanmean(np.abs((pred - yte)/denom))*100
        mapes.append(mape)
    mae = float(np.nanmean(maes))
    mape = float(np.nanmean(mapes))
    # 95% CI via fold errors
    mae_ci = 1.96*np.nanstd(maes, ddof=1)/np.sqrt(len(maes)) if len(maes)>1 else 0.0
    mape_ci = 1.96*np.nanstd(mapes, ddof=1)/np.sqrt(len(mapes)) if len(mapes)>1 else 0.0
    return mae, mae_ci, mape, mape_ci

def _fit_and_forecast(dfX: pd.DataFrame, horizon: int) -> np.ndarray:
    Xfull = dfX.drop(columns=["y","_dt"]).copy()
    yfull = dfX["y"].values
    pipe = SkPipeline([
        ("sc", StandardScaler(with_mean=False)),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
    ])
    pipe.fit(Xfull, yfull)
    # recursive (use last rows to build lags)
    df_for = dfX.copy()
    preds = []
    for _ in range(horizon):
        # build next-row features from last known row
        last = df_for.iloc[-1:].copy()
        new = last.copy()
        # shift lags
        for col in [c for c in df_for.columns if c.startswith("lag_")]:
            L = int(col.split("_")[1])
            # lag_L (new) = lag_{L-1} (last) or y for L==1
            if L==1:
                new[col] = last["y"].values
            else:
                prev = f"lag_{L-1}"
                new[col] = last[prev].values
        # time step forward (month/week/dow approximations)
        new["_dt"] = pd.to_datetime(last["_dt"].values[0]) + (pd.Timedelta(days=1))
        new["month"] = int(pd.to_datetime(new["_dt"]).dt.month)
        if "dow" in new.columns:
            new["dow"] = int((int(last["dow"]) + 1) % 7)
        if "week" in new.columns:
            new["week"] = int(pd.to_datetime(new["_dt"]).dt.isocalendar().week)
        Xnew = new.drop(columns=["y","_dt"])
        yhat = float(pipe.predict(Xnew)[0])
        yhat = max(0.0, yhat)
        new["y"] = yhat
        preds.append(yhat)
        df_for = pd.concat([df_for, new], ignore_index=True)
    return np.array(preds, dtype=float)

def _preset_to_horizon(grain: str, choice: str) -> Tuple[pd.Timestamp, int, List[str]]:
    today = pd.Timestamp.today().normalize()
    if grain=="Day":
        if choice=="Today":    start, periods, idx = today, 1, [str(today.date())]
        elif choice=="Tomorrow": 
            d = today + pd.Timedelta(days=1); start, periods, idx = d, 1, [str(d.date())]
        elif choice=="This Month":
            start = today.replace(day=today.day)  # next steps will fill
            last_day = (today.replace(day=1) + pd.offsets.MonthEnd(1))
            days = (last_day - today).days + 1
            periods = days; idx = [str((today + pd.Timedelta(days=i)).date()) for i in range(days)]
        elif choice=="Next Month":
            first_next = (today.replace(day=1) + pd.offsets.MonthBegin(1))
            last_next = (first_next + pd.offsets.MonthEnd(1))
            days = (last_next - first_next).days + 1
            periods = days; idx = [str((first_next + pd.Timedelta(days=i)).date()) for i in range(days)]
        else:  # Custom handled outside
            start, periods, idx = today, 1, [str(today.date())]
    elif grain=="Week":
        # Forecast 1 period for This/Next month as weekly sums
        if choice in ("Today","Tomorrow"): choice="This Month"
        if choice=="This Month":
            start = today
            last_day = (today.replace(day=1) + pd.offsets.MonthEnd(1))
            weeks = int(np.ceil((last_day - today).days/7))+1
            periods = max(1, weeks)
            idx = [f"{(today + pd.Timedelta(weeks=i)).date()}" for i in range(periods)]
        else:  # Next Month
            first_next = (today.replace(day=1) + pd.offsets.MonthBegin(1))
            weeks = 4
            periods = weeks
            idx = [f"{(first_next + pd.Timedelta(weeks=i)).date()}" for i in range(periods)]
    else:  # Month grain
        if choice in ("Today","Tomorrow"): choice="This Month"
        if choice=="This Month":
            start = today.replace(day=1)
            periods = 1; idx=[str(start.date())]
        else: # Next Month
            first_next = (today.replace(day=1) + pd.offsets.MonthBegin(1))
            start = first_next
            periods = 1; idx=[str(first_next.date())]
    return pd.to_datetime(start), int(periods), idx

with tabP:
    st.markdown("### 🔮 Predictability")
    with st.expander("Settings", expanded=True):
        # Base: use Scenario A global filters as context
        base_pred = metaA["base"].copy()
        # Measure to predict (default Payment Received Date if present)
        default_measure = "Payment Received Date" if "Payment Received Date" in date_like_cols else date_like_cols[0]
        target_measure = st.selectbox("Target date to predict (count of events)", options=date_like_cols,
                                      index=date_like_cols.index(default_measure))
        # Program filter via Pipeline
        prog_choice = st.radio("Program (from Pipeline)", ["Both", "AI-Coding", "Math"], horizontal=True)
        base_pred = subset_for_program(base_pred, prog_choice)

        # Grain + horizon
        grain = st.radio("Forecast granularity", ["Day","Week","Month"], horizontal=True, index=0)
        horizon_choice = st.radio("Horizon preset", ["Today","Tomorrow","This Month","Next Month","Custom"], horizontal=True, index=2)
        custom_range = None
        if horizon_choice=="Custom":
            dmin, dmax = safe_minmax_date(base_pred[target_measure].dropna())
            custom_range = st.date_input("Custom forecast window (start → end)", (dmin, dmax))
        split_by_source = st.toggle("Split by JetLearn Deal Source", value=True)
        st.caption("Backtest uses walk-forward rolling windows with no look-ahead, scalers fit on train only, and non-negative predictions.")

    # Build training series
    tmp = base_pred.dropna(subset=[target_measure]).copy()
    if tmp.empty:
        st.warning("No rows for selected target. Pick another date column.")
    else:
        if split_by_source:
            # per-source
            series, rule = _resample_counts(tmp, target_measure, grain, by_source=True)
            outputs = []
            charts = []
            metrics_list = []
            for src, sub in series.groupby("JetLearn Deal Source"):
                fX = _lag_features(sub[["_dt","y"]].copy(), grain)
                if fX.empty or fX["y"].sum()==0:
                    continue
                mae, mae_ci, mape, mape_ci = _walk_forward_backtest(fX)
                # Forecast horizon
                if horizon_choice!="Custom":
                    start_dt, periods, idx = _preset_to_horizon(grain, horizon_choice)
                else:
                    if isinstance(custom_range,(tuple,list)) and len(custom_range)==2:
                        start_dt = pd.to_datetime(custom_range[0])
                        end_dt   = pd.to_datetime(custom_range[1])
                        if grain=="Day":
                            periods = max(1, (end_dt - start_dt).days + 1)
                        elif grain=="Week":
                            periods = max(1, int(np.ceil((end_dt - start_dt).days/7)))
                        else:
                            # months between
                            periods = max(1, (end_dt.year - start_dt.year)*12 + (end_dt.month - start_dt.month) + 1)
                        idx = [str((start_dt + (pd.offsets.Day(i) if grain=="Day" else (pd.offsets.Week(i) if grain=="Week" else pd.offsets.MonthBegin(i)))).date())
                               for i in range(periods)]
                    else:
                        start_dt, periods, idx = _preset_to_horizon(grain, "This Month")
                preds = _fit_and_forecast(fX, periods)
                df_src = pd.DataFrame({"When": idx, "Predicted Count": preds.astype(int)})
                df_src.insert(0, "JetLearn Deal Source", src)
                outputs.append(df_src)
                metrics_list.append({"Source": src, "MAE": round(mae,2), "MAE±95%": round(mae_ci,2),
                                     "MAPE%": round(mape,1), "MAPE±95%": round(mape_ci,1)})
                # Chart
                hist = sub.rename(columns={"_dt":"When", "y":"Count"}).copy()
                hist["When"] = pd.to_datetime(hist["When"])
                last_hist = hist.iloc[-min(len(hist), 100):]  # last window
                fore = pd.DataFrame({"When": pd.to_datetime(idx), "Count": preds})
                ch = alt.Chart(last_hist).mark_line().encode(
                    x=alt.X("When:T", title=None), y=alt.Y("Count:Q", title="Count"),
                    tooltip=[alt.Tooltip("When:T"), alt.Tooltip("Count:Q")]
                ).properties(height=220)
                ch2 = alt.Chart(fore).mark_line(color="#2563eb", strokeDash=[4,3]).encode(
                    x="When:T", y="Count:Q", tooltip=["When:T","Count:Q"]
                )
                charts.append((src, ch + ch2))
            if not outputs:
                st.warning("No per-source data to forecast. Try turning off 'Split by source' or pick another target.")
            else:
                st.subheader("Per-source forecast")
                out = pd.concat(outputs, ignore_index=True)
                st.dataframe(out, use_container_width=True)
                st.download_button("Download per-source forecast CSV", to_csv_bytes(out),
                                   file_name="forecast_per_source.csv", mime="text/csv")
                st.subheader("Per-source backtest metrics")
                met = pd.DataFrame(metrics_list).sort_values("MAE")
                st.dataframe(met, use_container_width=True)
                st.subheader("Per-source charts")
                for src, chart in charts:
                    st.markdown(f"**{src}**")
                    st.altair_chart(chart, use_container_width=True)
                # Overall aggregate from per-source predictions
                agg = out.groupby("When", as_index=False)["Predicted Count"].sum()
                st.subheader("Overall predicted total")
                st.dataframe(agg, use_container_width=True)
                st.download_button("Download overall forecast CSV", to_csv_bytes(agg),
                                   file_name="forecast_overall.csv", mime="text/csv")
        else:
            # overall only
            series, rule = _resample_counts(tmp, target_measure, grain, by_source=False)
            fX = _lag_features(series[["_dt","y"]].copy(), grain)
            if fX.empty or fX["y"].sum()==0:
                st.warning("Not enough history to train. Try another target or grain.")
            else:
                mae, mae_ci, mape, mape_ci = _walk_forward_backtest(fX)
                if horizon_choice!="Custom":
                    start_dt, periods, idx = _preset_to_horizon(grain, horizon_choice)
                else:
                    dmin, dmax = safe_minmax_date(tmp[target_measure])
                    rng = st.date_input("Custom forecast window (start → end)", (dmin, dmax), key="pred_custom")
                    if isinstance(rng,(tuple,list)) and len(rng)==2:
                        start_dt = pd.to_datetime(rng[0]); end_dt = pd.to_datetime(rng[1])
                        if grain=="Day":
                            periods = max(1, (end_dt - start_dt).days + 1)
                        elif grain=="Week":
                            periods = max(1, int(np.ceil((end_dt - start_dt).days/7)))
                        else:
                            periods = max(1, (end_dt.year - start_dt.year)*12 + (end_dt.month - start_dt.month) + 1)
                        idx = [str((start_dt + (pd.offsets.Day(i) if grain=="Day" else (pd.offsets.Week(i) if grain=="Week" else pd.offsets.MonthBegin(i)))).date())
                               for i in range(periods)]
                    else:
                        start_dt, periods, idx = _preset_to_horizon(grain, "This Month")
                preds = _fit_and_forecast(fX, periods)
                out = pd.DataFrame({"When": idx, "Predicted Count": preds.astype(int)})
                st.subheader("Forecast (overall)")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download forecast CSV", to_csv_bytes(out),
                                   file_name="forecast_overall.csv", mime="text/csv")
                st.markdown("**Backtest (de-biased)**")
                st.write(f"MAE: **{mae:.2f} ± {mae_ci:.2f}**, MAPE: **{mape:.1f}% ± {mape_ci:.1f}%** (95% CI)")
                # Chart
                hist = series.rename(columns={"_dt":"When", "y":"Count"}).copy()
                hist["When"] = pd.to_datetime(hist["When"])
                last_hist = hist.iloc[-min(len(hist), 120):]
                fore = pd.DataFrame({"When": pd.to_datetime(idx), "Count": preds})
                ch = alt.Chart(last_hist).mark_line().encode(
                    x=alt.X("When:T", title=None), y=alt.Y("Count:Q", title="Count"),
                    tooltip=[alt.Tooltip("When:T"), alt.Tooltip("Count:Q")]
                ).properties(height=260)
                ch2 = alt.Chart(fore).mark_line(color="#2563eb", strokeDash=[4,3]).encode(
                    x="When:T", y="Count:Q", tooltip=["When:T","Count:Q"]
                )
                st.altair_chart(ch + ch2, use_container_width=True)

# -------------------- Footers --------------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.caption("**Scenario A** — " + mk_caption(metaA))
if st.session_state["show_b"]:
    st.caption("**Scenario B** — " + mk_caption(assemble_meta("B", df)))
st.caption("Excluded globally: 1.2 Invalid Deal")
