# app.py
# MTD vs Cohort (A/B) + Predictability (EB + Logistic blend, walk-forward backtest)
# Minimal, smart UI with drawer filters

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="JL Analyzer + Predictability", layout="wide", page_icon="📈")
st.markdown("""
<style>
:root{
  --text:#0f172a; --muted:#64748b; --blue:#2563eb; --border: rgba(15,23,42,.10);
  --card:#fff; --bg:#f8fafc;
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
.kpi .delta { font-size:.84rem; color: var(--blue); }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
.popcap { font-size:.78rem; color:var(--muted); margin-top:2px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]
REQUIRED_COLS = ["Pipeline", "JetLearn Deal Source", "Country",
                 "Student/Academic Counsellor", "Deal Stage", "Create Date"]

# --------------- HELPERS -----------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    date_like=[]
    for col in df.columns:
        if col=="Create Date": continue
        if any(k in col.lower() for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum()>0:
                df[col]=parsed; date_like.append(col)
    # Prefer Payment Received Date up front if present
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c!="Payment Received Date"]
    return date_like

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

# ---------- STATE & TOP BAR ----------
if "filters_open" not in st.session_state: st.session_state["filters_open"]=True
if "show_b" not in st.session_state: st.session_state["show_b"]=False
if "csv_path" not in st.session_state: st.session_state["csv_path"]="Master_sheet_DB_10percent.csv"
if "uploaded_bytes" not in st.session_state: st.session_state["uploaded_bytes"]=None

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

def _toggle_filters_toggle():
    st.session_state["filters_open"]=not st.session_state.get("filters_open",True)
def _enable_b(): st.session_state["show_b"]=True
def _disable_b(): st.session_state["show_b"]=False
def _reset_all_cb(): st.session_state.clear()
def _store_upload(key):
    up=st.session_state.get(key)
    if up is not None:
        st.session_state["uploaded_bytes"]=up.getvalue(); st.rerun()

# Top bar
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([5,2.2,2.5,2.5])
    with c1: st.markdown('<div class="title">☰ JL Analyzer + Predictability</div>', unsafe_allow_html=True)
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

# ---------- DATA LOAD ----------
_perform_clone_if_requested()

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
if not date_like_cols:
    st.error("No date-like columns besides 'Create Date' (e.g., 'Payment Received Date').")
    st.stop()

# Try detect payment col
def detect_payment_col(cols):
    lc=[c.lower() for c in cols]
    # strong rules
    for c in cols:
        cl=c.lower()
        if "payment" in cl and "received" in cl and "date" in cl:
            return c
    for c in cols:
        cl=c.lower()
        if "payment" in cl and "date" in cl:
            return c
    return None
PAYMENT_COL = detect_payment_col(df.columns) or ("Payment Received Date" if "Payment Received Date" in df.columns else None)

# ---------- FILTER WIDGETS ----------
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
        with left:
            st.checkbox("All", value=all_flag, key=all_key,
                        help="Tick to include all values; untick to select specific values.")
        with right:
            disabled=st.session_state[all_key]
            st.multiselect(label, options=options, default=selected, key=ms_key,
                           placeholder=f"Type to search {label.lower()}…",
                           label_visibility="collapsed", disabled=disabled)
            c1,c2 = st.columns(2)
            with c1:
                if st.button("Select all", key=f"{key_prefix}_select_all", use_container_width=True):
                    st.session_state[ms_key]=options; st.session_state[all_key]=True; st.rerun()
            with c2:
                if st.button("Clear", key=f"{key_prefix}_clear", use_container_width=True):
                    st.session_state[ms_key]=[]; st.session_state[all_key]=False; st.rerun()
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

def group_label_from_series(s: pd.Series, grain_key: str):
    grain=st.session_state.get(grain_key,"Month")
    if grain=="Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain=="Week":
        iso=pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str)+"-W"+iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

def assemble_meta(name: str, df: pd.DataFrame):
    # Read selected filters
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
                for f,m in zip(flags,measures):
                    rename_map[f]=f"MTD: {m}"
                grp=grp.rename(columns=rename_map).sort_values(by=f"MTD: {measures[0]}", ascending=False)
                tables[f"MTD split by {', '.join(split_dims)}"]=grp
            if show_top_countries and "Country" in sub.columns:
                g=sub.groupby("Country", dropna=False)[flags].sum().reset_index()
                ren = {f:f"MTD: {m}" for f,m in zip(flags,measures)}
                g=g.rename(columns=ren).sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)
                tables["Top 5 Countries — MTD"]=g
            if show_top_sources and "JetLearn Deal Source" in sub.columns:
                g=sub.groupby("JetLearn Deal Source", dropna=False)[flags].sum().reset_index()
                ren = {f:f"MTD: {m}" for f,m in zip(flags,measures)}
                g=g.rename(columns=ren).sort_values(by=f"MTD: {measures[0]}", ascending=False).head(3)
                tables["Top 3 Deal Sources — MTD"]=g
            if show_top_counsellors and "Student/Academic Counsellor" in sub.columns:
                g=sub.groupby("Student/Academic Counsellor", dropna=False)[flags].sum().reset_index()
                ren = {f:f"MTD: {m}" for f,m in zip(flags,measures)}
                g=g.rename(columns=ren).sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)
                tables["Top 5 Counsellors — MTD"]=g
            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(sub.columns):
                both=sub.groupby(["Country","JetLearn Deal Source"], dropna=False)[flags].sum().reset_index()
                ren = {f:f"MTD: {m}" for f,m in zip(flags,measures)}
                both=both.rename(columns=ren).sort_values(by=f"MTD: {measures[0]}", ascending=False).head(10)
                tables["Top Country × Deal Source — MTD"]=both
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
                for f,m in zip(ch_flags,measures):
                    rename_map2[f]=f"Cohort: {m}"
                grp2=grp2.rename(columns=rename_map2).sort_values(by=f"Cohort: {measures[0]}", ascending=False)
                tables[f"Cohort split by {', '.join(split_dims)}"]=grp2
            if show_top_countries and "Country" in base.columns:
                g=tmp.groupby("Country", dropna=False)[ch_flags].sum().reset_index()
                ren = {f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)}
                g=g.rename(columns=ren).sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)
                tables["Top 5 Countries — Cohort"]=g
            if show_top_sources and "JetLearn Deal Source" in base.columns:
                g=tmp.groupby("JetLearn Deal Source", dropna=False)[ch_flags].sum().reset_index()
                ren = {f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)}
                g=g.rename(columns=ren).sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(3)
                tables["Top 3 Deal Sources — Cohort"]=g
            if show_top_counsellors and "Student/Academic Counsellor" in base.columns:
                g=tmp.groupby("Student/Academic Counsellor", dropna=False)[ch_flags].sum().reset_index()
                ren = {f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)}
                g=g.rename(columns=ren).sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)
                tables["Top 5 Counsellors — Cohort"]=g
            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(base.columns):
                both2=tmp.groupby(["Country","JetLearn Deal Source"], dropna=False)[ch_flags].sum().reset_index()
                ren = {f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)}
                both2=both2.rename(columns=ren).sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(10)
                tables["Top Country × Deal Source — Cohort"]=both2
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

# ---------- SIDEBAR (drawer) ----------
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

# ---------- ANALYZE TABS ----------
tabA_label = "Scenario A"
tabs = [tabA_label]
if st.session_state["show_b"]:
    tabs.append("Scenario B")
tabs.append("Compare")
tabs.append("Predictability")
tab_objects = st.tabs(tabs)

# ----- Scenario A
metaA = assemble_meta("A", df)
with st.spinner("Crunching scenario A…"):
    metricsA, tablesA, chartsA = compute_outputs(metaA)

with tab_objects[0]:
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

# ----- Scenario B (optional)
if st.session_state["show_b"]:
    metaB = assemble_meta("B", df)
    with st.spinner("Crunching scenario B…"):
        metricsB, tablesB, chartsB = compute_outputs(metaB)
    with tab_objects[1]:
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

# ----- Compare
cmp_tab_index = 1 if not st.session_state["show_b"] else 2
with tab_objects[cmp_tab_index]:
    st.markdown("<div class='section-title'>🧠 Smart Compare (A vs B)</div>", unsafe_allow_html=True)
    if not st.session_state["show_b"]:
        st.info("Enable Scenario B to compare.")
    else:
        dA=pd.DataFrame(metricsA); dB=pd.DataFrame(metricsB)
        if dA.empty or dB.empty:
            st.info("Turn on KPIs for both scenarios to enable compare.")
        else:
            cmp=build_compare_delta(dA, dB)
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

# ---------------- PREDICTABILITY ----------------
# Empirical-Bayes smoothing for rates by segment (Source x Country x Pipeline)
def make_cohorts(df: pd.DataFrame, payment_col: str):
    out = df.copy()
    out["Create_Month"] = out["Create Date"].dt.to_period("M").astype(str)
    out["paid_flag"] = out[payment_col].notna().astype(int) if payment_col in out.columns else 0
    grp_cols = ["Create_Month", "JetLearn Deal Source", "Country", "Pipeline"]
    g = out.groupby(grp_cols, dropna=False).agg(
        created=("Create Date", "count"),
        paid=("paid_flag","sum")
    ).reset_index()
    g["rate"] = g["paid"] / g["created"].replace(0, np.nan)
    g["rate"] = g["rate"].fillna(0)
    return g

def eb_rate(segment_created, segment_paid, global_alpha, global_beta, shrink=1.0):
    # Posterior mean of Beta-Binomial with prior Beta(alpha, beta)
    return (segment_paid + shrink*global_alpha) / (segment_created + shrink*(global_alpha + global_beta))

def global_prior(g: pd.DataFrame):
    # Weakly-informative prior from overall conversion
    total_c = g["created"].sum()
    total_p = g["paid"].sum()
    # alpha,beta scaled by 'strength' ~ a couple of months overall
    strength = max(50.0, total_c * 0.05)
    overall = (total_p / total_c) if total_c>0 else 0.01
    alpha = overall * strength
    beta  = (1-overall) * strength
    # guard
    alpha = max(alpha, 1e-3); beta = max(beta, 1e-3)
    return alpha, beta

# --- Logistic on aggregated counts (two-rows-per-group + sample weights)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def _logit_rates(cohorts: pd.DataFrame):
    need = ["JetLearn Deal Source", "Country", "created", "paid"]
    if not set(need).issubset(cohorts.columns):
        return None

    dfc = cohorts[need].copy()
    dfc = dfc.replace([np.inf, -np.inf], np.nan).dropna(subset=need)

    dfc["created"] = pd.to_numeric(dfc["created"], errors="coerce")
    dfc["paid"]    = pd.to_numeric(dfc["paid"], errors="coerce")
    dfc = dfc[dfc["created"] > 0]
    if dfc.empty: return None

    dfc["paid"] = dfc[["paid","created"]].min(axis=1).clip(lower=0)

    X_pos = dfc[["JetLearn Deal Source", "Country"]].copy()
    X_neg = X_pos.copy()
    y = np.r_[np.ones(len(dfc)), np.zeros(len(dfc))]
    w = np.r_[dfc["paid"].values, (dfc["created"] - dfc["paid"]).values]
    X2 = pd.concat([X_pos, X_neg], ignore_index=True)

    mask = w > 0
    if not np.any(mask): return None
    X2 = X2.iloc[mask].reset_index(drop=True)
    y  = y[mask]
    w  = w[mask]
    if len(np.unique(y[w > 0])) < 2: return None

    cat_cols = ["JetLearn Deal Source", "Country"]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(solver="lbfgs", max_iter=500)
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(X2, y, clf__sample_weight=w)
    return pipe

def _predict_rate_from_logit(logit_model, jetlearn_source: str, country: str) -> float:
    if logit_model is None:
        return None
    X = pd.DataFrame({"JetLearn Deal Source":[jetlearn_source], "Country":[country]})
    p = logit_model.predict_proba(X)[0, 1]
    if not np.isfinite(p):
        return None
    return float(np.clip(p, 0.0, 1.0))

def recent_creation_baseline(df: pd.DataFrame, months=3):
    # Recent moving-average of created by segment for volume forecast
    df["Create_Month"] = df["Create Date"].dt.to_period("M").astype(str)
    last_month = df["Create_Month"].max()
    if last_month is None or pd.isna(last_month): return pd.DataFrame(columns=["JetLearn Deal Source","Country","Pipeline","created_ma"])
    # take last N months
    all_months = sorted(df["Create_Month"].unique())
    take = all_months[-months:] if len(all_months)>=months else all_months
    recent = df[df["Create_Month"].isin(take)]
    g = recent.groupby(["JetLearn Deal Source","Country","Pipeline"], dropna=False)["Create Date"].count().reset_index(name="created")
    g = g.groupby(["JetLearn Deal Source","Country","Pipeline"], dropna=False)["created"].mean().reset_index(name="created_ma")
    return g

def month_days(dt: date)->int:
    first = dt.replace(day=1)
    next_first = (date(dt.year+1,1,1) if dt.month==12 else date(dt.year, dt.month+1, 1))
    return (next_first - first).days

def month_str(dt: date)->str:
    return pd.Timestamp(dt).to_period("M").astype(str)

def walk_forward_backtest(df: pd.DataFrame, payment_col: str, lookback_months=6, blend=0.6):
    """
    Train on rolling window of lookback_months, predict next month using:
      predicted_paid = created_next_month * (0.6*logit_rate + 0.4*EB_rate)
    Returns metrics DataFrame.
    """
    if payment_col not in df.columns:
        return pd.DataFrame(columns=["test_month","MAE","MAPE","WAPE"])
    tmp = df.copy()
    tmp["Create_Month"] = tmp["Create Date"].dt.to_period("M").astype(str)
    months_sorted = sorted(tmp["Create_Month"].dropna().unique())
    if len(months_sorted) < (lookback_months + 2):
        return pd.DataFrame(columns=["test_month","MAE","MAPE","WAPE"])

    records=[]
    for i in range(lookback_months, len(months_sorted)-1):
        train_months = months_sorted[i-lookback_months:i]
        test_month   = months_sorted[i]
        train_df = tmp[tmp["Create_Month"].isin(train_months)].copy()
        test_df  = tmp[tmp["Create_Month"]==test_month].copy()

        # Cohorts on training
        cohorts = make_cohorts(train_df, payment_col)
        alpha,beta = global_prior(cohorts)
        # EB map for (source,country)
        eb_map = cohorts.groupby(["JetLearn Deal Source","Country"], dropna=False).agg(
            c=("created","sum"), p=("paid","sum")
        ).reset_index()
        eb_map["eb"] = eb_map.apply(lambda r: eb_rate(r["c"], r["p"], alpha, beta, shrink=1.0), axis=1)

        # Logistic on training (source,country)
        logit = _logit_rates(cohorts)

        # next-month created per segment
        created_next = test_df.groupby(["JetLearn Deal Source","Country"], dropna=False)["Create Date"].count().reset_index(name="created")
        if created_next.empty: 
            continue

        # predicted rate per segment
        def seg_rate(row):
            src = row["JetLearn Deal Source"]; ctry = row["Country"]
            eb = eb_map.loc[(eb_map["JetLearn Deal Source"]==src) & (eb_map["Country"]==ctry), "eb"]
            eb_val = float(eb.iloc[0]) if len(eb)>0 else float((cohorts["paid"].sum()+alpha)/(cohorts["created"].sum()+alpha+beta))
            p_logit = _predict_rate_from_logit(logit, src, ctry) if logit is not None else None
            return (blend * (p_logit if p_logit is not None else eb_val)) + ((1-blend) * eb_val)

        created_next["pred_rate"] = created_next.apply(seg_rate, axis=1).clip(0,1)
        created_next["pred_paid"] = created_next["created"] * created_next["pred_rate"]

        # actual paid in test month
        paid_mask = test_df[payment_col].notna()
        actual_paid = int(paid_mask.sum())

        y_hat = float(created_next["pred_paid"].sum())
        mae = abs(y_hat - actual_paid)
        mape = (mae / actual_paid * 100) if actual_paid>0 else np.nan

        # WAPE over segments (weighted absolute percentage error)
        # here equal to MAE / total actual (same as MAPE numerator), include guard
        wape = (mae / actual_paid) if actual_paid>0 else np.nan

        records.append({"test_month":test_month, "Pred":y_hat, "Actual":actual_paid, "MAE":mae, "MAPE":mape, "WAPE":wape})

    return pd.DataFrame(records)

# ---- Predictability tab UI & logic
pred_tab_index = cmp_tab_index + 1
with tab_objects[pred_tab_index]:
    st.markdown("### 🔮 Predictability")
    if PAYMENT_COL is None:
        st.warning("No Payment Received Date column detected. Add a column like 'Payment Received Date' to enable predictions.")
    else:
        st.caption(f"Using payment column: **{PAYMENT_COL}**")

    cc1, cc2, cc3 = st.columns([2,2,2])
    with cc1:
        lookback = st.slider("Backtest window (months)", 3, 18, 6, 1,
                             help="Rolling training window length for walk-forward validation.")
    with cc2:
        blend = st.slider("Blend weight (Logit vs EB)", 0.0, 1.0, 0.6, 0.05,
                          help="0 = only Empirical-Bayes, 1 = only Logistic. Default blends both.")
    with cc3:
        vol_ma = st.slider("Volume baseline (recent months MA)", 1, 6, 3, 1,
                           help="Number of recent months to average for creation volume baseline.")

    # Optional scope filters for prediction (reuse unified_multifilter style)
    st.markdown("#### Scope Filters (optional)")
    p1, p2, p3 = st.columns(3)
    with p1:
        f_src_all, f_src_sel, _ = unified_multifilter("Deal Source", df, "JetLearn Deal Source", "P_src")
    with p2:
        f_cty_all, f_cty_sel, _ = unified_multifilter("Country", df, "Country", "P_cty")
    with p3:
        f_pip_all, f_pip_sel, _ = unified_multifilter("Pipeline", df, "Pipeline", "P_pip")

    scope_mask = (
        in_filter(df["JetLearn Deal Source"], f_src_all, f_src_sel) &
        in_filter(df["Country"], f_cty_all, f_cty_sel) &
        in_filter(df["Pipeline"], f_pip_all, f_pip_sel)
    )
    df_scoped = df[scope_mask].copy()

    # Backtest
    with st.spinner("Running walk-forward backtest..."):
        bt = walk_forward_backtest(df_scoped, PAYMENT_COL, lookback_months=lookback, blend=blend)
    if bt.empty:
        st.info("Not enough data to backtest with the chosen window.")
    else:
        st.subheader("Backtest summary")
        agg = {
            "MAE": bt["MAE"].mean(),
            "MAPE": bt["MAPE"].dropna().mean() if not bt["MAPE"].dropna().empty else np.nan,
            "WAPE": bt["WAPE"].dropna().mean() if not bt["WAPE"].dropna().empty else np.nan
        }
        cols = st.columns(3)
        cols[0].metric("Avg MAE", f"{agg['MAE']:.1f}")
        cols[1].metric("Avg MAPE", f"{agg['MAPE']:.1f}%" if pd.notna(agg["MAPE"]) else "—")
        cols[2].metric("Avg WAPE", f"{agg['WAPE']:.2f}" if pd.notna(agg["WAPE"]) else "—")

        chart_bt = alt.Chart(bt).mark_line(point=True).encode(
            x=alt.X("test_month:O", title=None),
            y=alt.Y("MAE:Q", title="MAE"),
            tooltip=["test_month","Pred","Actual","MAE","MAPE","WAPE"]
        ).properties(height=260)
        st.altair_chart(chart_bt, use_container_width=True)

    st.markdown("---")
    # Forecasts for This month / Next month (+ daily scaling)
    st.subheader("Forecasts")
    today = pd.Timestamp.today().date()
    this_m = month_str(today)
    next_m = month_str(today.replace(day=28) + timedelta(days=4))  # safe next month trick

    # Train on recent window for live forecast
    # Use same components as backtest on the last 'lookback' months
    df_scoped["Create_Month"] = df_scoped["Create Date"].dt.to_period("M").astype(str)
    months_sorted = sorted(df_scoped["Create_Month"].dropna().unique())
    if len(months_sorted) >= lookback:
        train_months = months_sorted[-lookback:]
        train_df = df_scoped[df_scoped["Create_Month"].isin(train_months)].copy()
        cohorts = make_cohorts(train_df, PAYMENT_COL)
        alpha,beta = global_prior(cohorts)
        eb_map = cohorts.groupby(["JetLearn Deal Source","Country"], dropna=False).agg(
            c=("created","sum"), p=("paid","sum")
        ).reset_index()
        eb_map["eb"] = eb_map.apply(lambda r: eb_rate(r["c"], r["p"], alpha, beta, shrink=1.0), axis=1)
        logit = _logit_rates(cohorts)

        # Creation baseline per segment
        vol_base = recent_creation_baseline(df_scoped, months=vol_ma)
        if vol_base.empty:
            st.info("Not enough data to build a volume baseline.")
        else:
            # Build segment grid (source x country x pipeline) present in baseline
            seg = vol_base.copy()
            # predict rate for each (src,country) and apply to created_ma
            def rate_for(segrow):
                src=segrow["JetLearn Deal Source"]; ctry=segrow["Country"]
                eb = eb_map.loc[(eb_map["JetLearn Deal Source"]==src) & (eb_map["Country"]==ctry), "eb"]
                eb_val = float(eb.iloc[0]) if len(eb)>0 else float((cohorts["paid"].sum()+alpha)/(cohorts["created"].sum()+alpha+beta))
                p_logit = _predict_rate_from_logit(logit, src, ctry) if logit is not None else None
                return (blend * (p_logit if p_logit is not None else eb_val)) + ((1-blend) * eb_val)

            seg["rate"] = seg.apply(rate_for, axis=1).clip(0,1)
            seg["pred_paid_month"] = seg["created_ma"] * seg["rate"]

            # Aggregate options
            st.markdown("**Monthly forecasts (total)**")
            tot_this = float(seg["pred_paid_month"].sum())
            tot_next = float(seg["pred_paid_month"].sum())  # same baseline unless you add growth factors
            # daily scaling for "Today" / "Tomorrow"
            md = month_days(today)
            passed = today.day
            frac = max(0.0, min(1.0, passed/md))
            today_to_date = tot_this * frac
            tomorrow_to_date = tot_this * min(1.0, (passed+1)/md)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Today (to-date est.)", f"{today_to_date:.0f}")
            c2.metric("Tomorrow (to-date est.)", f"{tomorrow_to_date:.0f}")
            c3.metric("This month (EOM)", f"{tot_this:.0f}")
            c4.metric("Next month (EOM)", f"{tot_next:.0f}")

            st.markdown("**By JetLearn Deal Source × Country × Pipeline**")
            st.dataframe(seg.rename(columns={"created_ma":"Created (MA)", "pred_paid_month":"Pred Enrolments"}),
                         use_container_width=True)
            st.download_button("Download forecast table", to_csv_bytes(seg),
                               file_name="forecast_segments.csv", mime="text/csv")
    else:
        st.info("Not enough months to fit a live forecast. Increase data or reduce lookback.")

# ---- Footer captions
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.caption("Excluded globally: 1.2 Invalid Deal")
