# app_drawer_ui_v8.py
# Drawer UI (v5 baseline) + upgraded Predictability with backtesting & model selection
# A/B scenarios + Compare + Predictability (auto-selects best model per source)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

# ---------------- Page & Style ----------------
st.set_page_config(page_title="MTD vs Cohort — Drawer UI + Predictability (v8)", layout="wide", page_icon="📊")
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

# ---------------- Constants ----------------
REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country",
                 "Student/Academic Counsellor","Deal Stage","Create Date"]
PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# ---------------- Clone helpers ----------------
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

# ---------------- Utilities ----------------
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

def detect_program_column(df: pd.DataFrame):
    candidates = ["Program","Course","Product","Track","Program Name","Course Name","Package","Interest","Program Interested"]
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        cl=c.lower()
        if "program" in cl or "course" in cl: return c
    return None

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

# Presets for A/B
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

# Predictability-presets
def this_week_bounds():
    t = pd.Timestamp.today().date()
    start = t - timedelta(days=t.weekday())
    return start, start + timedelta(days=6)
def next_week_bounds():
    s, _ = this_week_bounds()
    s = s + timedelta(days=7)
    return s, s + timedelta(days=6)
def this_month_bounds():
    t = pd.Timestamp.today().date()
    start = t.replace(day=1)
    next_start = (start.replace(year=start.year+1, month=1) if start.month==12 else start.replace(month=start.month+1))
    return start, next_start - timedelta(days=1)
def next_month_bounds():
    s, _ = this_month_bounds()
    nm_start = (s.replace(year=s.year+1, month=1) if s.month==12 else s.replace(month=s.month+1))
    nm_end = (nm_start.replace(year=nm_start.year+1, month=1) if nm_start.month==12 else nm_start.replace(month=nm_start.month+1)) - timedelta(days=1)
    return nm_start, nm_end

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

def alt_line(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

def alt_bar(df, x, y, color=None, tooltip=None, height=280):
    enc = dict(x=alt.X(x, title=None), y=alt.Y(y, title=None), tooltip=tooltip or [])
    if color: enc["color"] = alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_bar().encode(**enc).properties(height=height)

def to_csv_bytes(df: pd.DataFrame)->bytes: return df.to_csv(index=False).encode("utf-8")

# ---------------- State defaults ----------------
if "filters_open" not in st.session_state: st.session_state["filters_open"]=True
if "show_b" not in st.session_state: st.session_state["show_b"]=False
if "csv_path" not in st.session_state: st.session_state["csv_path"]="Master_sheet_DB_10percent.csv"
if "uploaded_bytes" not in st.session_state: st.session_state["uploaded_bytes"]=None

def _toggle_filters_toggle():
    st.session_state["filters_open"]=not st.session_state.get("filters_open",True)
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

# ---------------- Top bar ----------------
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([5,2.2,2.5,2.5])
    with c1: st.markdown('<div class="title">☰ MTD vs Cohort — Drawer UI</div>', unsafe_allow_html=True)
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

# ---------------- Data load ----------------
if st.session_state["uploaded_bytes"]:
    df = robust_read_csv(BytesIO(st.session_state["uploaded_bytes"]))
else:
    df = robust_read_csv(st.session_state["csv_path"])
df.columns=[c.strip() for c in df.columns]
missing=[c for c in REQUIRED_COLS if c not in df.columns]
if missing: st.error(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}"); st.stop()

# Clean + enrich
df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
df["Create Date"]=pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"]=df["Create Date"].dt.to_period("M")
date_like_cols=detect_measure_date_columns(df)
if not date_like_cols: st.error("No date-like columns besides 'Create Date' (e.g., 'Payment Received Date')."); st.stop()
program_col = detect_program_column(df)  # for Predictability

# ---------------- Global filter widgets ----------------
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
                for f, m in zip(flags, measures):
                    rename_map[f] = f"MTD: {m}"
                grp=grp.rename(columns=rename_map).sort_values(by=f"MTD: {measures[0]}", ascending=False)
                tables[f"MTD split by {', '.join(split_dims)}"]=grp
            if show_top_countries and "Country" in sub.columns:
                g=sub.groupby("Country", dropna=False)[flags].sum().reset_index()
                for f, m in zip(flags, measures): g=g.rename(columns={f:f"MTD: {m}"})
                tables["Top 5 Countries — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)
            if show_top_sources and "JetLearn Deal Source" in sub.columns:
                g=sub.groupby("JetLearn Deal Source", dropna=False)[flags].sum().reset_index()
                for f, m in zip(flags, measures): g=g.rename(columns={f:f"MTD: {m}"})
                tables["Top 3 Deal Sources — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(3)
            if show_top_counsellors and "Student/Academic Counsellor" in sub.columns:
                g=sub.groupby("Student/Academic Counsellor", dropna=False)[flags].sum().reset_index()
                for f, m in zip(flags, measures): g=g.rename(columns={f:f"MTD: {m}"})
                tables["Top 5 Counsellors — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)
            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(sub.columns):
                both=sub.groupby(["Country","JetLearn Deal Source"], dropna=False)[flags].sum().reset_index()
                for f, m in zip(flags, measures): both=both.rename(columns={f:f"MTD: {m}"})
                tables["Top Country × Deal Source — MTD"]=both.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(10)
            trend=sub.copy()
            trend["Bucket"]=group_label_from_series(trend["Create Date"], f"{meta['name']}_mtd_grain")
            t=trend.groupby("Bucket")[flags].sum().reset_index()
            for f, m in zip(flags, measures): t=t.rename(columns={f:m})
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
                for f, m in zip(ch_flags, measures): rename_map2[f]=f"Cohort: {m}"
                grp2=grp2.rename(columns=rename_map2).sort_values(by=f"Cohort: {measures[0]}", ascending=False)
                tables[f"Cohort split by {', '.join(split_dims)}"]=grp2
            if show_top_countries and "Country" in base.columns:
                g=tmp.groupby("Country", dropna=False)[ch_flags].sum().reset_index()
                for f, m in zip(ch_flags, measures): g=g.rename(columns={f:f"Cohort: {m}"})
                tables["Top 5 Countries — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)
            if show_top_sources and "JetLearn Deal Source" in base.columns:
                g=tmp.groupby("JetLearn Deal Source", dropna=False)[ch_flags].sum().reset_index()
                for f, m in zip(ch_flags, measures): g=g.rename(columns={f:f"Cohort: {m}"})
                tables["Top 3 Deal Sources — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(3)
            if show_top_counsellors and "Student/Academic Counsellor" in base.columns:
                g=tmp.groupby("Student/Academic Counsellor", dropna=False)[ch_flags].sum().reset_index()
                for f, m in zip(ch_flags, measures): g=g.rename(columns={f:f"Cohort: {m}"})
                tables["Top 5 Counsellors — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)
            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(base.columns):
                both2=tmp.groupby(["Country","JetLearn Deal Source"], dropna=False)[ch_flags].sum().reset_index()
                for f, m in zip(ch_flags, measures): both2=both2.rename(columns={f:f"Cohort: {m}"})
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

# ---------------- Predictability (Upgraded ML) ----------------
@st.cache_data(show_spinner=False)
def build_enrolment_daily(df: pd.DataFrame, program_col: str|None):
    """Daily enrolments by JetLearn Deal Source from Payment Received Date (fallback: first other date-like)."""
    if "Payment Received Date" in df.columns:
        measure = "Payment Received Date"
    else:
        dcols = [c for c in df.columns if c != "Create Date" and ("date" in c.lower() or "time" in c.lower())]
        measure = dcols[0] if dcols else "Create Date"
    work = df.copy()
    work[measure] = pd.to_datetime(work[measure], errors="coerce")
    work = work.dropna(subset=[measure])
    work["d"] = work[measure].dt.date
    cols = ["d", "JetLearn Deal Source"]
    if program_col and program_col in work.columns:
        cols.append(program_col)
    g = work.groupby(cols, dropna=False).size().reset_index(name="y")
    return g, measure

def expand_calendar(start: date, end: date) -> pd.DataFrame:
    rng = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"d": rng.date})
    out["dow"]  = rng.dayofweek
    out["dom"]  = rng.day
    out["mon"]  = rng.month
    out["doy"]  = rng.dayofyear
    out["woy"]  = rng.isocalendar().week.astype(int)
    out["trend"]= np.arange(len(out))
    return out

def make_feats(full_cal: pd.DataFrame, hist: pd.DataFrame):
    x = pd.merge(full_cal, hist.rename(columns={"y":"y_hist"}), on="d", how="left")
    x["y"] = x["y_hist"].fillna(0.0)
    x = x.drop(columns=["y_hist"])
    # lags & rolling mean
    x["lag1"]  = x["y"].shift(1)
    x["lag7"]  = x["y"].shift(7)
    x["roll7"] = x["y"].rolling(7, min_periods=1).mean()
    # fill features
    X = x[["trend","dow","dom","mon","doy","woy","lag1","lag7","roll7"]].fillna(0.0)
    y = x["y"].values
    return x, X, y

def fit_model(X, y, name: str):
    if name == "Linear":
        mdl = LinearRegression()
    elif name == "Ridge":
        mdl = Ridge(alpha=1.0, random_state=None)
    elif name == "RandomForest":
        mdl = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1, min_samples_leaf=2)
    elif name == "GBDT":
        mdl = GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=300, learning_rate=0.05)
    else:
        return None
    mdl.fit(X, y)
    return mdl

def naive_mean_forecaster(y_hist: np.ndarray, days:int):
    mu = float(pd.Series(y_hist).tail(28).mean() if len(y_hist)>0 else 0.0)
    sigma = float(pd.Series(y_hist).tail(28).std(ddof=1) if len(y_hist)>1 else 0.0)
    return np.full(days, mu), sigma

def walk_forward_backtest(ts: pd.DataFrame, model_name: str, fold_size:int=14, n_folds:int=3):
    """
    Rolling-origin backtest. Each fold trains on all data up to split point and
    predicts next 'fold_size' days. Returns metrics & residual stats.
    """
    ts = ts.sort_values("d").reset_index(drop=True)
    if len(ts) < max(28, fold_size*(n_folds+1)):
        return None  # not enough data

    start_idx = len(ts) - fold_size*n_folds
    folds = []
    errors = []
    preds_all = []
    acts_all  = []
    for f in range(n_folds):
        end_train = start_idx + f*fold_size
        train = ts.iloc[:end_train].copy()
        test  = ts.iloc[end_train:end_train+fold_size].copy()
        # calendar covering train + test
        cal = expand_calendar(train["d"].min(), test["d"].max())
        _, X_full, y_full = make_feats(cal, train.rename(columns={"y":"y"}))
        # indexes for test part
        X_train = X_full.iloc[:len(cal)-fold_size]
        y_train = y_full[:len(cal)-fold_size]
        X_test  = X_full.iloc[len(cal)-fold_size:]
        # predict
        if model_name == "Naive":
            yhat, sigma = naive_mean_forecaster(y_train, fold_size)
        else:
            mdl = fit_model(X_train, y_train, model_name)
            if mdl is None:
                yhat, sigma = naive_mean_forecaster(y_train, fold_size)
            else:
                yhat = mdl.predict(X_test)
                # residual sigma from train fit
                y_fit = mdl.predict(X_train)
                sigma = float(np.sqrt(np.mean((y_train - y_fit)**2))) if len(y_train)>0 else 0.0
        y_true = test["y"].values.astype(float)
        preds_all.append(yhat); acts_all.append(y_true)
        err = y_true - yhat
        errors.append(err)
        folds.append(dict(y_true=y_true, yhat=yhat, sigma=sigma))

    y_true_all = np.concatenate(acts_all)
    y_pred_all = np.concatenate(preds_all)
    abs_err = np.abs(y_true_all - y_pred_all)
    mae  = float(abs_err.mean())
    mape = float(np.mean(abs_err / np.maximum(1e-9, y_true_all)))  # guard zero
    wape = float(abs_err.sum() / np.maximum(1e-9, y_true_all.sum()))
    bias = float((y_pred_all.sum() - y_true_all.sum()) / np.maximum(1e-9, y_true_all.sum()))
    sigma_resid = float(np.sqrt(np.mean(np.concatenate(errors)**2)))
    return dict(MAE=mae, MAPE=mape, WAPE=wape, Bias=bias, Sigma=sigma_resid)

def train_best_model(ts: pd.DataFrame, candidates=("GBDT","RandomForest","Ridge","Linear","Naive")):
    """
    Backtests all candidates, picks best by MAPE, then fits final on full history.
    Returns (best_name, metrics, final_model_or_None, sigma, bias_for_calibration)
    """
    # evaluate
    results = {}
    for name in candidates:
        bt = walk_forward_backtest(ts, name, fold_size=14, n_folds=3)
        if bt is not None:
            results[name] = bt
    if not results:
        # fallback if insufficient data
        name = "Naive"
        bt = dict(MAE=np.nan, MAPE=np.nan, WAPE=np.nan, Bias=0.0, Sigma=0.0)
        final = None; sigma=0.0; bias=0.0
        return name, bt, final, sigma, bias

    # pick best by MAPE (lowest)
    best = min(results.items(), key=lambda kv: (np.inf if np.isnan(kv[1]["MAPE"]) else kv[1]["MAPE"]))
    best_name, best_metrics = best

    # fit final
    cal = expand_calendar(ts["d"].min(), ts["d"].max())
    _, X_full, y_full = make_feats(cal, ts.rename(columns={"y":"y"}))
    if best_name == "Naive":
        final = None
        sigma = float(pd.Series(y_full).tail(28).std(ddof=1) if len(y_full)>1 else 0.0)
    else:
        final = fit_model(X_full, y_full, best_name)
        if final is None:
            sigma = float(pd.Series(y_full).tail(28).std(ddof=1) if len(y_full)>1 else 0.0)
        else:
            y_fit = final.predict(X_full)
            sigma = float(np.sqrt(np.mean((y_full - y_fit)**2))) if len(y_full)>0 else 0.0

    # bias from backtest to de-bias forecasts
    bias = best_metrics.get("Bias", 0.0)
    return best_name, best_metrics, final, sigma, bias

def predict_with_model(ts: pd.DataFrame, model_name: str, final_model, start: date, end: date, sigma: float, de_bias: float=0.0):
    cal_hist = expand_calendar(ts["d"].min(), end)
    _, X_full, y_full = make_feats(cal_hist, ts.rename(columns={"y":"y"}))
    # horizon rows
    cal_h = expand_calendar(start, end)
    horizon_days = len(cal_h)
    X_pred = X_full.iloc[-horizon_days:]
    # predict
    if model_name == "Naive" or final_model is None:
        yhat, sigma_est = naive_mean_forecaster(y_full, horizon_days)
        sigma_use = sigma if sigma>0 else sigma_est
        yhat = yhat
    else:
        yhat = final_model.predict(X_pred)
        sigma_use = sigma
    # de-bias: if bias > 0 means over-forecast; reduce by factor (1+bias)
    if np.isfinite(de_bias) and de_bias != 0.0:
        yhat = yhat / (1.0 + de_bias)
    lo = np.clip(yhat - 1.64*sigma_use, 0, None)
    hi = np.clip(yhat + 1.64*sigma_use, 0, None)
    out = pd.DataFrame({"d": cal_h["d"], "yhat": yhat, "yhat_lo": lo, "yhat_hi": hi})
    return out

# ---------------- Sidebar drawer ----------------
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

# ---------------- Compute A/B ----------------
metaA = assemble_meta("A", df)
with st.spinner("Crunching scenario A…"):
    metricsA, tablesA, chartsA = compute_outputs(metaA)
if st.session_state["show_b"]:
    metaB = assemble_meta("B", df)
    with st.spinner("Crunching scenario B…"):
        metricsB, tablesB, chartsB = compute_outputs(metaB)

# ---------------- Tabs ----------------
tabs = ["Scenario A"]
if st.session_state["show_b"]:
    tabs += ["Scenario B", "Compare"]
tabs += ["Predictability"]
t_objs = st.tabs(tabs)
tabA = t_objs[0]
tabB = t_objs[1] if st.session_state["show_b"] else None
tabC = t_objs[2] if st.session_state["show_b"] else None
tabPred = t_objs[-1]

# ---------------- Render A ----------------
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

# ---------------- Render B + Compare ----------------
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

# ---------------- Predictability Tab (Upgraded) ----------------
with tabPred:
    st.markdown("<div class='section-title'>🔮 Predictability — Source-wise Enrolment Forecast (with Backtesting)</div>", unsafe_allow_html=True)
    st.caption("Forecasts **enrolments** by **JetLearn Deal Source** from *Payment Received Date* (or nearest date-like). "
               "Uses rolling-origin backtests to choose the best model per source and reports accuracy metrics.")

    colH, colP, colM = st.columns([2,2,3])
    with colH:
        horizon = st.selectbox("Horizon", ["Today","Tomorrow","This Week","Next Week","This Month","Next Month","Custom"])
    with colP:
        # Program filter
        if program_col:
            prog_choice = st.selectbox("Program", ["Both","AI-Coding","Math","Custom filter…"], index=0)
            prog_custom = None
            if prog_choice == "Custom filter…":
                all_vals = sorted([v for v in df[program_col].dropna().astype(str).unique()])
                prog_custom = st.multiselect("Choose program values", all_vals, default=all_vals)
        else:
            prog_choice = "Both"
            st.info("No program column found — treating **Both** as all records.", icon="ℹ️")
    with colM:
        model_pref = st.selectbox("Model preference", ["Auto (best MAPE)","GBDT","RandomForest","Ridge","Linear","Naive"], index=0)
        do_backtest = st.toggle("Show backtest details", value=True)
        de_bias_on  = st.toggle("De-bias forecasts using backtest bias", value=True)

    # Horizon dates
    if horizon == "Today":
        H_from, H_to = today_bounds()
    elif horizon == "Tomorrow":
        t = pd.Timestamp.today().date() + timedelta(days=1)
        H_from, H_to = t, t
    elif horizon == "This Week":
        H_from, H_to = this_week_bounds()
    elif horizon == "Next Week":
        H_from, H_to = next_week_bounds()
    elif horizon == "This Month":
        H_from, H_to = this_month_bounds()
    elif horizon == "Next Month":
        H_from, H_to = next_month_bounds()
    else:
        dmin, dmax = safe_minmax_date(df["Create Date"])
        rng = st.date_input("Custom forecast range", (dmin, dmax))
        H_from, H_to = (rng[0], rng[1]) if isinstance(rng,(tuple,list)) and len(rng)==2 else (dmin,dmax)

    with st.spinner("Preparing time series…"):
        daily, used_measure = build_enrolment_daily(df, program_col)
        # Program filter
        if program_col and prog_choice != "Both":
            if prog_choice == "AI-Coding":
                daily = daily[daily[program_col].astype(str).str.contains("ai", case=False, na=False)]
            elif prog_choice == "Math":
                daily = daily[daily[program_col].astype(str).str.contains("math", case=False, na=False)]
            elif prog_choice == "Custom filter…":
                if prog_custom:
                    daily = daily[daily[program_col].astype(str).isin(prog_custom)]
                else:
                    st.warning("No program values selected; using all.")
        sources = sorted(daily["JetLearn Deal Source"].dropna().astype(str).unique())

    st.caption(f"Using enrolment signal from **{used_measure}**.")
    rows = []
    daily_forecasts = {}
    bt_rows = []

    with st.spinner("Backtesting & forecasting per source…"):
        for src in sources:
            ts = (daily[daily["JetLearn Deal Source"].astype(str)==src]
                    .groupby("d")["y"].sum().reset_index())
            if ts.empty or len(ts)<21:
                continue

            if model_pref.startswith("Auto"):
                best_name, metrics, final_model, sigma, bias = train_best_model(ts)
            else:
                # Evaluate the chosen model too (so we still have metrics)
                metrics = walk_forward_backtest(ts, "Naive", fold_size=14, n_folds=3) or dict(MAE=np.nan, MAPE=np.nan, WAPE=np.nan, Bias=0.0, Sigma=0.0)
                # Train on full for selected model
                cal = expand_calendar(ts["d"].min(), ts["d"].max())
                _, X_full, y_full = make_feats(cal, ts.rename(columns={"y":"y"}))
                if model_pref == "Naive":
                    best_name, final_model = "Naive", None
                    sigma = float(pd.Series(y_full).tail(28).std(ddof=1) if len(y_full)>1 else 0.0)
                    bias = 0.0
                else:
                    mdl = fit_model(X_full, y_full, model_pref)
                    if mdl is None:
                        best_name, final_model = "Naive", None
                        sigma = float(pd.Series(y_full).tail(28).std(ddof=1) if len(y_full)>1 else 0.0)
                        bias = 0.0
                    else:
                        best_name, final_model = model_pref, mdl
                        y_fit = mdl.predict(X_full)
                        sigma = float(np.sqrt(np.mean((y_full - y_fit)**2))) if len(y_full)>0 else 0.0
                        # Also compute model's own backtest metrics
                        mt = walk_forward_backtest(ts, model_pref, fold_size=14, n_folds=3)
                        if mt is not None:
                            metrics = mt
                        bias = metrics.get("Bias", 0.0)

            # Forecast
            pred_df = predict_with_model(ts, best_name, final_model, H_from, H_to, sigma=sigma, de_bias=bias if de_bias_on else 0.0)
            pred_sum = float(pred_df["yhat"].sum())
            lo = float(pred_df["yhat_lo"].sum())
            hi = float(pred_df["yhat_hi"].sum())
            acc = (1.0 - (metrics["MAPE"] if np.isfinite(metrics["MAPE"]) else np.nan))*100.0

            rows.append({
                "JetLearn Deal Source": src,
                "Model": best_name,
                "Accuracy % (1-MAPE)": None if np.isnan(acc) else round(acc,1),
                "MAE": None if np.isnan(metrics["MAE"]) else round(metrics["MAE"],2),
                "Bias %": round(metrics["Bias"]*100.0,1) if np.isfinite(metrics["Bias"]) else None,
                "Predicted": round(pred_sum,1),
                "Lower (90%)": round(lo,1),
                "Upper (90%)": round(hi,1),
            })
            if do_backtest:
                bt_rows.append({
                    "JetLearn Deal Source": src,
                    "Model": best_name,
                    "MAPE": None if np.isnan(metrics["MAPE"]) else round(metrics["MAPE"]*100.0,1),
                    "WAPE": None if np.isnan(metrics["WAPE"]) else round(metrics["WAPE"]*100.0,1),
                    "MAE": None if np.isnan(metrics["MAE"]) else round(metrics["MAE"],2),
                    "Bias %": round(metrics["Bias"]*100.0,1) if np.isfinite(metrics["Bias"]) else None,
                    "Sigma (RMSE of resid)": round(metrics["Sigma"],2)
                })
            daily_forecasts[src] = pred_df.assign(source=src)

    if not rows:
        st.warning("Not enough historical enrolments to produce a robust forecast.")
    else:
        pred_table = pd.DataFrame(rows).sort_values("Predicted", ascending=False)
        # Totals
        tot = {
            "JetLearn Deal Source": "TOTAL",
            "Model": "—",
            "Accuracy % (1-MAPE)": None,
            "MAE": None,
            "Bias %": None,
            "Predicted": round(pred_table["Predicted"].sum(),1),
            "Lower (90%)": round(pred_table["Lower (90%)"].sum(),1),
            "Upper (90%)": round(pred_table["Upper (90%)"].sum(),1),
        }
        # Weighted MAPE across sources (by actuals is better; we don't have future actuals—so weight by predicted)
        valid_acc = pred_table.dropna(subset=["Accuracy % (1-MAPE)"])
        if not valid_acc.empty:
            weights = valid_acc["Predicted"].replace(0, np.nan)
            w_acc = np.nansum(valid_acc["Accuracy % (1-MAPE)"]*weights)/np.nansum(weights)
            tot["Accuracy % (1-MAPE)"] = round(w_acc,1)
        # Render table
        st.subheader(f"Forecast for {H_from} → {H_to}")
        st.dataframe(pd.concat([pd.DataFrame([tot]), pred_table], ignore_index=True), use_container_width=True)

        # Bar chart for predictions
        chart_df = pred_table.rename(columns={"Predicted":"yhat"})
        st.altair_chart(
            alt_bar(chart_df, x="JetLearn Deal Source:N", y="yhat:Q",
                    tooltip=["JetLearn Deal Source","yhat","Lower (90%)","Upper (90%)","Model","Accuracy % (1-MAPE)"]),
            use_container_width=True
        )

        if do_backtest and bt_rows:
            st.markdown("**Backtest metrics (rolling-origin)**")
            bt_df = pd.DataFrame(bt_rows).sort_values("MAPE", ascending=True)
            st.dataframe(bt_df, use_container_width=True)

        with st.expander("Daily forecast details"):
            long_df = pd.concat(daily_forecasts.values(), ignore_index=True) if daily_forecasts else pd.DataFrame()
            if not long_df.empty:
                st.dataframe(long_df.sort_values(["source","d"]), use_container_width=True)
                st.download_button("Download daily forecast (CSV)", to_csv_bytes(long_df),
                                   file_name="daily_forecast.csv", mime="text/csv")

# ---------------- Footer ----------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
def mk_caption(meta):
    return (f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
            f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(coerce_list(meta['pipe_sel'])) or 'None'} · "
            f"Deal Source: {'All' if meta['src_all'] else ', '.join(coerce_list(meta['src_sel'])) or 'None'} · "
            f"Country: {'All' if meta['ctry_all'] else ', '.join(coerce_list(meta['ctry_sel'])) or 'None'} · "
            f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(coerce_list(meta['cslr_sel'])) or 'None'}")
st.caption("**Scenario A** — " + mk_caption(assemble_meta("A", df)))
if st.session_state["show_b"]:
    st.caption("**Scenario B** — " + mk_caption(assemble_meta("B", df)))
st.caption("Excluded globally: 1.2 Invalid Deal")
