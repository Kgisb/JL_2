# app.py  — JetLearn Insights (MTD/Cohort) + Predictability (M0 vs Carryover with partial-month extrapolation)

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

def detect_measure_date_columns(df: pd.DataFrame):
    """Find all date-like columns except Create Date; coerce to datetime."""
    date_like=[]
    for col in df.columns:
        if col == "Create Date": continue
        cl = col.lower()
        if any(k in cl for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum()>0:
                df[col] = parsed
                date_like.append(col)
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

def alt_line(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

def alt_bar(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_bar().encode(**enc).properties(height=height)

def to_csv_bytes(df: pd.DataFrame)->bytes: return df.to_csv(index=False).encode("utf-8")

def group_label_from_series(s: pd.Series, grain: str):
    if grain=="Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain=="Week":
        iso=pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str)+"-W"+iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

# --------------------- Header ---------------------
with st.container():
    st.markdown('<div class="head"><div class="title">📊 JetLearn — Insights & Predictability</div></div>', unsafe_allow_html=True)

# --------------------- Data Source (shared by both tabs) ---------------------
ds = st.expander("Data source (Historical / Modeling)", expanded=True)
with ds:
    c1, c2, c3 = st.columns([3,2,2])
    with c1:
        uploaded_hist = st.file_uploader("Upload HISTORICAL CSV (modeling dataset)", type=["csv"], key="hist")
    with c2:
        path_hist = st.text_input("…or CSV path (historical)", value="Master_sheet_DB_10percent.csv")
    with c3:
        exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal'", value=True)

# Current/partial month dataset for predictability
ds2 = st.expander("Prediction month data (Partial current month file)", expanded=False)
with ds2:
    c21, c22 = st.columns([3,2])
    with c21:
        uploaded_curr = st.file_uploader("Upload CURRENT MONTH PARTIAL CSV (optional)", type=["csv"], key="curr")
    with c22:
        path_curr = st.text_input("…or CSV path (current month partial)", value="")

# -------- Load Historical --------
try:
    if uploaded_hist is not None:
        df = robust_read_csv(BytesIO(uploaded_hist.getvalue()))
    else:
        df = robust_read_csv(path_hist)
    st.success("Historical data loaded ✅")
except Exception as e:
    st.error(str(e)); st.stop()

df.columns = [c.strip() for c in df.columns]
missing=[c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns in historical data: {missing}\nAvailable: {list(df.columns)}")
    st.stop()

if exclude_invalid:
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"] = df["Create Date"].dt.to_period("M").dt.to_timestamp()

# Coerce date-like cols for Insights
date_like_cols = detect_measure_date_columns(df)

# -------- Load Current/Partial (optional) --------
df_curr = None
if uploaded_curr is not None or path_curr.strip():
    try:
        if uploaded_curr is not None:
            df_curr = robust_read_csv(BytesIO(uploaded_curr.getvalue()))
        else:
            df_curr = robust_read_csv(path_curr)
        df_curr.columns = [c.strip() for c in df_curr.columns]
        if exclude_invalid and "Deal Stage" in df_curr.columns:
            df_curr = df_curr[~df_curr["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
        if "Create Date" in df_curr.columns:
            df_curr["Create Date"] = pd.to_datetime(df_curr["Create Date"], errors="coerce", dayfirst=True)
            df_curr["Create_Month"] = df_curr["Create Date"].dt.to_period("M").dt.to_timestamp()
        st.success("Current/partial month data loaded ✅")
    except Exception as e:
        st.warning(f"Could not load current/partial file: {e}. Proceeding without it.")

# --------------------- Tabs ---------------------
tab_insights, tab_predict = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictability (M0 + Carryover)"])

# =========================================================
# ===============  INSIGHTS (MTD / Cohort)  ===============
# =========================================================
with tab_insights:

    if not date_like_cols:
        st.error("No usable date-like columns (other than Create Date) found. Add a column like 'Payment Received Date'.")
        st.stop()

    # ---- Compact global multi-filters with search ----
    def summary_label(values, all_flag, max_items=2):
        vals = coerce_list(values)
        if all_flag: return "All"
        if not vals: return "None"
        s = ", ".join(map(str, vals[:max_items]))
        if len(vals)>max_items: s += f" +{len(vals)-max_items} more"
        return s

    def unified_multifilter(label, df, colname, key_prefix):
        options = sorted([v for v in df[colname].dropna().astype(str).unique()]) if colname in df.columns else []
        all_key = f"{key_prefix}_all"
        ms_key  = f"{key_prefix}_ms"
        header = f"{label}: " + (summary_label(options, True) if options else "—")
        ctx = st.popover(header) if hasattr(st, "popover") else st.expander(header, expanded=False)
        with ctx:
            c1,c2 = st.columns([1,3])
            all_flag = c1.checkbox("All", value=True, key=all_key, disabled=(len(options)==0))
            disabled = st.session_state.get(all_key, True) or (len(options)==0)
            _sel = st.multiselect(label, options=options,
                                  default=options, key=ms_key,
                                  placeholder=f"Type to search {label.lower()}…",
                                  label_visibility="collapsed",
                                  disabled=disabled)
        all_flag = bool(st.session_state.get(all_key, True))
        selected = [v for v in coerce_list(st.session_state.get(ms_key, options)) if v in options]
        effective = options if all_flag else selected
        st.markdown(f"<div class='popcap'>{label}: {summary_label(effective, all_flag)}</div>", unsafe_allow_html=True)
        return all_flag, effective

    def date_preset_row(name, base_series, key_prefix, default_grain="Month"):
        presets=["Today","This month so far","Last month","Last quarter","This year","Custom"]
        c1,c2 = st.columns([3,2])
        with c1:
            choice = st.radio(f"[{name}] Range", presets, horizontal=True, key=f"{key_prefix}_preset")
        with c2:
            grain = st.radio("Granularity", ["Day","Week","Month"], horizontal=True,
                             index=["Day","Week","Month"].index(default_grain),
                             key=f"{key_prefix}_grain")
        if choice=="Today": f,t=today_bounds()
        elif choice=="This month so far": f,t=this_month_so_far_bounds()
        elif choice=="Last month": f,t=last_month_bounds()
        elif choice=="Last quarter": f,t=last_quarter_bounds()
        elif choice=="This year": f,t=this_year_so_far_bounds()
        else:
            dmin,dmax=safe_minmax_date(base_series)
            rng=st.date_input("Custom range",(dmin,dmax),key=f"{key_prefix}_custom")
            f,t = (rng if isinstance(rng,(tuple,list)) and len(rng)==2 else (dmin,dmax))
        if f>t: st.error("'From' is after 'To'. Adjust the range.")
        return f,t,grain

    def scenario_controls(name: str, df: pd.DataFrame, date_like_cols):
        st.markdown(f"**Scenario {name}** <span class='badge'>independent</span>", unsafe_allow_html=True)
        pipe_all, pipe_sel = unified_multifilter("Pipeline", df, "Pipeline", f"{name}_pipe")
        src_all,  src_sel  = unified_multifilter("Deal Source", df, "JetLearn Deal Source", f"{name}_src")
        cty_all,  cty_sel  = unified_multifilter("Country", df, "Country", f"{name}_cty")
        csl_all,  csl_sel  = unified_multifilter("Counsellor", df, "Student/Academic Counsellor", f"{name}_csl")

        mask = (
            in_filter(df["Pipeline"], pipe_all, pipe_sel) &
            in_filter(df["JetLearn Deal Source"], src_all, src_sel) &
            in_filter(df["Country"], cty_all, cty_sel) &
            in_filter(df["Student/Academic Counsellor"], csl_all, csl_sel)
        ) if all(c in df.columns for c in ["Pipeline","JetLearn Deal Source","Country","Student/Academic Counsellor"]) else pd.Series(True, index=df.index)
        base = df[mask].copy()

        st.markdown("##### Measures & Windows")
        mcol1,mcol2 = st.columns([3,2])
        with mcol1:
            measures = st.multiselect(f"[{name}] Measure date(s)",
                                      options=date_like_cols,
                                      default=(["Payment Received Date"] if "Payment Received Date" in date_like_cols else date_like_cols[:1]),
                                      key=f"{name}_measures")
        with mcol2:
            mode = st.radio(f"[{name}] Mode", ["MTD","Cohort","Both"], horizontal=True, key=f"{name}_mode")

        for m in measures:
            mn = f"{m}_Month"
            if m in base.columns and mn not in base.columns:
                base[mn] = base[m].dt.to_period("M").dt.to_timestamp()

        mtd_from = mtd_to = coh_from = coh_to = None
        mtd_grain = coh_grain = "Month"

        if mode in ("MTD","Both"):
            st.caption("Create-Date window (MTD)")
            mtd_from, mtd_to, mtd_grain = date_preset_row(name, base["Create Date"], f"{name}_mtd", default_grain="Month")
        if mode in ("Cohort","Both"):
            st.caption("Measure-Date window (Cohort)")
            series = base[measures[0]] if measures else base["Create Date"]
            coh_from, coh_to, coh_grain = date_preset_row(name, series, f"{name}_coh", default_grain="Month")

        with st.expander(f"[{name}] Splits & Leaderboards", expanded=False):
            sc1, sc2, sc3 = st.columns([3,2,2])
            split_dims = sc1.multiselect(f"[{name}] Split by", ["JetLearn Deal Source","Country","Student/Academic Counsellor"],
                                         default=[], key=f"{name}_split")
            top_ctry = sc2.checkbox(f"[{name}] Top 5 Countries", value=True, key=f"{name}_top_ctry")
            top_src  = sc3.checkbox(f"[{name}] Top 3 Deal Sources", value=True, key=f"{name}_top_src")
            top_csl  = st.checkbox(f"[{name}] Top 5 Counsellors", value=False, key=f"{name}_top_csl")
            pair     = st.checkbox(f"[{name}] Country × Deal Source (Top 10)", value=False, key=f"{name}_pair")

        return dict(
            name=name, base=base, measures=measures, mode=mode,
            mtd_from=mtd_from, mtd_to=mtd_to, mtd_grain=mtd_grain,
            coh_from=coh_from, coh_to=coh_to, coh_grain=coh_grain,
            split_dims=split_dims, top_ctry=top_ctry, top_src=top_src, top_csl=top_csl, pair=pair,
            pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
            cty_all=cty_all, cty_sel=cty_sel, csl_all=csl_all, csl_sel=csl_sel
        )

    def compute_outputs(meta):
        base=meta["base"]; measures=meta["measures"] or []
        mode=meta["mode"]
        mtd_from, mtd_to, mtd_grain = meta["mtd_from"], meta["mtd_to"], meta["mtd_grain"]
        coh_from, coh_to, coh_grain = meta["coh_from"], meta["coh_to"], meta["coh_grain"]
        split_dims=meta["split_dims"]
        top_ctry, top_src, top_csl, pair = meta["top_ctry"], meta["top_src"], meta["top_csl"], meta["pair"]

        metrics_rows, tables, charts = [], {}, {}

        # MTD
        if mode in ("MTD","Both") and mtd_from and mtd_to and measures:
            in_cre = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
            sub = base[in_cre].copy()
            flags=[]
            for m in measures:
                if m not in sub.columns: continue
                flg=f"__MTD__{m}"
                sub[flg] = ((sub[m].notna()) & (sub[f"{m}_Month"]==sub["Create_Month"])).astype(int)
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

                if top_ctry and "Country" in sub.columns:
                    g=sub.groupby("Country", dropna=False)[flags].sum().reset_index()
                    g=g.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                    tables["Top 5 Countries — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)

                if top_src and "JetLearn Deal Source" in sub.columns:
                    g=sub.groupby("JetLearn Deal Source", dropna=False)[flags].sum().reset_index()
                    g=g.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                    tables["Top 3 Deal Sources — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(3)

                if top_csl and "Student/Academic Counsellor" in sub.columns:
                    g=sub.groupby("Student/Academic Counsellor", dropna=False)[flags].sum().reset_index()
                    g=g.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                    tables["Top 5 Counsellors — MTD"]=g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)

                if pair and {"Country","JetLearn Deal Source"}.issubset(sub.columns):
                    both=sub.groupby(["Country","JetLearn Deal Source"], dropna=False)[flags].sum().reset_index()
                    both=both.rename(columns={f:f"MTD: {m}" for f,m in zip(flags,measures)})
                    tables["Top Country × Deal Source — MTD"]=both.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(10)

                trend=sub.copy()
                trend["Bucket"]=group_label_from_series(trend["Create Date"], mtd_grain)
                t=trend.groupby("Bucket")[flags].sum().reset_index()
                t=t.rename(columns={f:m for f,m in zip(flags,measures)})
                long=t.melt(id_vars="Bucket", var_name="Measure", value_name="Count")
                charts["MTD Trend"]=alt_line(long,"Bucket:O","Count:Q",color="Measure:N",tooltip=["Bucket","Measure","Count"])

        # Cohort
        if mode in ("Cohort","Both") and coh_from and coh_to and measures:
            tmp=base.copy(); ch_flags=[]
            for m in measures:
                if m not in tmp.columns: continue
                flg=f"__COH__{m}"
                tmp[flg]=tmp[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both").astype(int)
                ch_flags.append(flg)
                metrics_rows.append({"Scope":"Cohort","Metric":f"Count on '{m}'","Window":f"{coh_from} → {coh_to}","Value":int(tmp[flg].sum())})
            in_cre_coh = base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
            metrics_rows.append({"Scope":"Cohort","Metric":"Create Count in Cohort window","Window":f"{coh_from} → {coh_to}","Value":int(in_cre_coh.sum())})

            if ch_flags:
                if split_dims:
                    tmp["_CreateInCohort"]=in_cre_coh.astype(int)
                    grp2=tmp.groupby(split_dims, dropna=False)[ch_flags+["_CreateInCohort"]].sum().reset_index()
                    rename_map2={"_CreateInCohort":"Create Count in Cohort window"}
                    for f,m in zip(ch_flags,measures): rename_map2[f]=f"Cohort: {m}"
                    grp2=grp2.rename(columns=rename_map2).sort_values(by=f"Cohort: {measures[0]}", ascending=False)
                    tables[f"Cohort split by {', '.join(split_dims)}"]=grp2

                if top_ctry and "Country" in base.columns:
                    g=tmp.groupby("Country", dropna=False)[ch_flags].sum().reset_index()
                    g=g.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                    tables["Top 5 Countries — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)

                if top_src and "JetLearn Deal Source" in base.columns:
                    g=tmp.groupby("JetLearn Deal Source", dropna=False)[ch_flags].sum().reset_index()
                    g=g.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                    tables["Top 3 Deal Sources — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(3)

                if top_csl and "Student/Academic Counsellor" in base.columns:
                    g=tmp.groupby("Student/Academic Counsellor", dropna=False)[ch_flags].sum().reset_index()
                    g=g.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                    tables["Top 5 Counsellors — Cohort"]=g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)

                if pair and {"Country","JetLearn Deal Source"}.issubset(base.columns):
                    both2=tmp.groupby(["Country","JetLearn Deal Source"], dropna=False)[ch_flags].sum().reset_index()
                    both2=both2.rename(columns={f:f"Cohort: {m}" for f,m in zip(ch_flags,measures)})
                    tables["Top Country × Deal Source — Cohort"]=both2.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(10)

                frames=[]
                for m in measures:
                    mask=base[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
                    sel=base.loc[mask,[m]].copy()
                    if sel.empty: continue
                    sel["Bucket"]=group_label_from_series(sel[m], coh_grain)
                    t=sel.groupby("Bucket")[m].count().reset_index(name="Count")
                    t["Measure"]=m
                    frames.append(t)
                if frames:
                    trend=pd.concat(frames, ignore_index=True)
                    charts["Cohort Trend"]=alt_line(trend,"Bucket:O","Count:Q",color="Measure:N",tooltip=["Bucket","Measure","Count"])

        return metrics_rows, tables, charts

    def caption_from(meta):
        return (f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
                f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(coerce_list(meta['pipe_sel'])) or 'None'} · "
                f"Deal Source: {'All' if meta['src_all'] else ', '.join(coerce_list(meta['src_sel'])) or 'None'} · "
                f"Country: {'All' if meta['cty_all'] else ', '.join(coerce_list(meta['cty_sel'])) or 'None'} · "
                f"Counsellor: {'All' if meta['csl_all'] else ', '.join(coerce_list(meta['csl_sel'])) or 'None'}")

    show_b = st.toggle("Enable Scenario B (compare)", value=False)
    left_col, right_col = st.columns(2) if show_b else (st.container(), None)
    with (left_col if show_b else st.container()):
        metaA = scenario_controls("A", df, date_like_cols)
    if show_b:
        with right_col:
            metaB = scenario_controls("B", df, date_like_cols)

    with st.spinner("Calculating…"):
        metricsA, tablesA, chartsA = compute_outputs(metaA)
        if show_b:
            metricsB, tablesB, chartsB = compute_outputs(metaB)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    if show_b:
        tA, tB, tC = st.tabs(["📋 Scenario A", "📋 Scenario B", "🧠 Compare"])
    else:
        tA, = st.tabs(["📋 Scenario A"])

    with tA:
        st.markdown("<div class='section-title'>📌 KPI Overview — A</div>", unsafe_allow_html=True)
        dfA=pd.DataFrame(metricsA)
        if dfA.empty: st.info("No KPIs yet — adjust filters.")
        else:
            cols=st.columns(4)
            for i,row in dfA.iterrows():
                with cols[i%4]:
                    st.markdown(f"""
<div class="kpi">
  <div class="label">{row['Scope']} — {row['Metric']}</div>
  <div class="value">{int(row['Value']):,}</div>
  <div class="delta">{row['Window']}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — A</div>", unsafe_allow_html=True)
        if not tablesA: st.info("No tables — enable splits/leaderboards.")
        else:
            for name,frame in tablesA.items():
                st.subheader(name)
                st.dataframe(frame, use_container_width=True)
                st.download_button("Download CSV — "+name, to_csv_bytes(frame),
                                   file_name=f"A_{name.replace(' ','_')}.csv", mime="text/csv")
        st.markdown("<div class='section-title'>📈 Trends — A</div>", unsafe_allow_html=True)
        if "MTD Trend" in chartsA: st.altair_chart(chartsA["MTD Trend"], use_container_width=True)
        if "Cohort Trend" in chartsA: st.altair_chart(chartsA["Cohort Trend"], use_container_width=True)
        st.caption("**Scenario A** — " + caption_from(metaA))

    if show_b:
        with tB:
            st.markdown("<div class='section-title'>📌 KPI Overview — B</div>", unsafe_allow_html=True)
            dfB=pd.DataFrame(metricsB)
            if dfB.empty: st.info("No KPIs yet — adjust filters.")
            else:
                cols=st.columns(4)
                for i,row in dfB.iterrows():
                    with cols[i%4]:
                        st.markdown(f"""
<div class="kpi">
  <div class="label">{row['Scope']} — {row['Metric']}</div>
  <div class="value">{int(row['Value']):,}</div>
  <div class="delta">{row['Window']}</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — B</div>", unsafe_allow_html=True)
            if not tablesB: st.info("No tables — enable splits/leaderboards.")
            else:
                for name,frame in tablesB.items():
                    st.subheader(name)
                    st.dataframe(frame, use_container_width=True)
                    st.download_button("Download CSV — "+name, to_csv_bytes(frame),
                                       file_name=f"B_{name.replace(' ','_')}.csv", mime="text/csv")
            st.markdown("<div class='section-title'>📈 Trends — B</div>", unsafe_allow_html=True)
            if "MTD Trend" in chartsB: st.altair_chart(chartsB["MTD Trend"], use_container_width=True)
            if "Cohort Trend" in chartsB: st.altair_chart(chartsB["Cohort Trend"], use_container_width=True)
            st.caption("**Scenario B** — " + caption_from(metaB))

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.caption("Excluded globally: 1.2 Invalid Deal")

# =========================================================
# ===========  PREDICTABILITY (M0 + Carryover) ============
# =========================================================
with tab_predict:

    # ---- Detect payment column ----
    def detect_payment_col(cols):
        for c in cols:
            cl=c.lower()
            if "payment" in cl and "received" in cl and "date" in cl:
                return c
        for c in cols:
            cl=c.lower()
            if "payment" in cl and "date" in cl:
                return c
        return None

    # ---- Month helpers ----
    def month_start(dt: date) -> pd.Timestamp:
        return pd.Timestamp(dt).to_period("M").to_timestamp()

    def month_days(ts: pd.Timestamp) -> int:
        m0 = ts.to_period("M").to_timestamp()
        m1 = (m0 + pd.offsets.MonthBegin(1))
        return int((m1 - m0).days)

    # ---- Create & Payment coercion ----
    PAY_COL = detect_payment_col(df.columns)
    if PAY_COL is None:
        st.error("Couldn't find a payment date column in historical data. Add one like 'Payment Received Date'.")
        st.stop()

    df_pred_base = df.copy()
    df_pred_base["Payment Received Date"] = pd.to_datetime(df_pred_base[PAY_COL], errors="coerce", dayfirst=True)
    df_pred_base["Payment_Month"] = df_pred_base["Payment Received Date"].dt.to_period("M").dt.to_timestamp()

    # ---- Controls ----
    st.markdown("### Forecast controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        # target month defaults to this month
        today = pd.Timestamp.today().normalize()
        cm_default = pd.Timestamp(year=today.year, month=today.month, day=1)
        # allow choosing from a window around history
        mmin = min(df_pred_base["Create_Month"].dropna().min(), df_pred_base["Payment_Month"].dropna().min())
        mmax = max(df_pred_base["Create_Month"].dropna().max(), df_pred_base["Payment_Month"].dropna().max())
        options = list(pd.date_range(start=(mmin - pd.offsets.MonthBegin(0)), end=(mmax + pd.offsets.MonthBegin(6)), freq="MS"))
        target_month = st.selectbox("Target month", options=options, index=options.index(cm_default) if cm_default in options else len(options)-1, format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))
        target_month = pd.Timestamp(target_month)
    with c2:
        split_by = st.selectbox("Split forecast by", ["None","JetLearn Deal Source","Student/Academic Counsellor","Country","Pipeline"])
    with c3:
        lookback_split = st.slider("Split lookback (months)", 3, 12, 6, 1)

    st.caption("This forecast = **M0 (same-month)** + **Carryover (lagged)** into the target month.")

    # ---------- Build cohort matrix (historical) ----------
    # Create cohorts
    hist = df_pred_base.copy()
    hist["CM"] = hist["Create_Month"]
    hist["PM"] = hist["Payment_Month"]
    # Monthly creates
    creates_by_CM = hist.groupby("CM")["Create Date"].count().rename("Creates_CM")
    # Payments attributed to creation-month (cohort payments)
    cohort_paid = hist.dropna(subset=["PM"]).groupby(["CM","PM"])["Payment Received Date"].count().rename("Paid")
    cohort = cohort_paid.reset_index()
    cohort["lag_k"] = ((cohort["PM"].dt.year - cohort["CM"].dt.year) * 12 + (cohort["PM"].dt.month - cohort["CM"].dt.month)).astype(int)
    cohort = cohort[cohort["lag_k"] >= 0]

    # Derive global lag distribution P(lag=k) based on creates
    # For stability, compute:
    #   M0_rate = sum_paid_k0 / sum_creates
    #   For k>=1: lag_k_share = sum_paid_k / sum_creates
    # Smoothing via small epsilon
    epsilon = 1e-6
    # Sum payments by lag
    paid_by_k = cohort.groupby("lag_k")["Paid"].sum()
    total_creates = creates_by_CM.sum()
    # If total_creates is zero, avoid division
    if total_creates <= 0:
        st.error("No creates in historical data.")
        st.stop()

    lag_prob = (paid_by_k / max(total_creates, 1.0)).reindex(range(0, 18), fill_value=0.0)  # up to 18 months lag
    # Normalize to <=1 (it can be <1 if some historical creates haven't converted yet)
    lag_prob = lag_prob.clip(lower=0.0)
    # Extract M0 rate
    M0_rate = float(lag_prob.get(0, 0.0))

    # ---------- Create day-of-month profile (for extrapolating creates) ----------
    def create_day_profile(df_hist: pd.DataFrame, month_of_year: int) -> pd.Series:
        d = df_hist.copy()
        d = d.dropna(subset=["Create Date"])
        d["d"] = d["Create Date"].dt.day
        d["moy"] = d["Create Date"].dt.month
        pool = d[d["moy"]==month_of_year]
        if pool.empty: pool = d
        cnt = pool["d"].value_counts().sort_index()
        # Determine maximum days in any month in history for index range
        max_days = 31
        idx = pd.Index(range(1, max_days+1), name="day")
        cnt = cnt.reindex(idx, fill_value=0)
        total = cnt.sum()
        if total == 0:
            return pd.Series(1.0/max_days, index=idx, name="prop")
        return (cnt/total).rename("prop")

    # ---------- Helper: EB split proportions ----------
    def eb_smooth_props(counts_by_cat: pd.Series, prior_props: pd.Series, prior_strength: float = 5.0):
        counts = counts_by_cat.astype(float)
        total = counts.sum()
        if total <= 0:
            pp = prior_props.fillna(0).clip(0,1)
            return (pp / pp.sum()) if pp.sum()>0 else pp
        cats = counts.index
        prior = prior_props.reindex(cats).fillna(0.0)
        smoothed = (counts + prior_strength * prior) / (total + prior_strength)
        s = smoothed.sum()
        return smoothed / s if s>0 else smoothed

    def historical_split_props(df_paid: pd.DataFrame, split_col: str, lookback_months: int = 6):
        if split_col not in df_paid.columns:
            return pd.Series(dtype=float)
        dfp = df_paid.copy()
        dfp["PaymentMonth"] = dfp["Payment Received Date"].dt.to_period("M").dt.to_timestamp()
        global_counts = dfp.groupby(split_col)["Payment Received Date"].count()
        gp_total = global_counts.sum()
        prior_props = (global_counts / gp_total) if gp_total>0 else global_counts
        months = sorted(dfp["PaymentMonth"].dropna().unique())
        take = months[-lookback_months:] if len(months)>=lookback_months else months
        recent = dfp[dfp["PaymentMonth"].isin(take)]
        recent_counts = recent.groupby(split_col)["Payment Received Date"].count().sort_values(ascending=False)
        return eb_smooth_props(recent_counts, prior_props, prior_strength=5.0)

    # ---------- Compute forecast components ----------
    # 1) Estimate full-month creates for target month (using partial file if provided; else infer from historical pattern)
    tm_moy = int(target_month.month)
    create_profile = create_day_profile(df_pred_base, tm_moy)  # proportions by day (1..31)
    days_in_target = month_days(target_month)

    # observed creates in target month so far (from partial/current file if provided)
    def observed_creates_so_far(df_current: pd.DataFrame, target_month_ts: pd.Timestamp) -> tuple[int,int,float]:
        """Returns (obs_creates, last_day_observed, cumulative_prop_through_last_day)."""
        if df_current is None or "Create Date" not in df_current.columns:
            return 0, 0, 0.0
        cur = df_current.copy()
        cur["CM"] = cur["Create Date"].dt.to_period("M").dt.to_timestamp()
        cur_tm = cur[cur["CM"]==target_month_ts]
        if cur_tm.empty:
            return 0, 0, 0.0
        cur_tm["day"] = cur_tm["Create Date"].dt.day
        last_day = int(cur_tm["day"].max())
        obs = int(len(cur_tm))
        # cap at days_in_target
        last_day = min(last_day, days_in_target)
        cum_prop = float(create_profile.loc[1:last_day].sum()) if last_day>=1 else 0.0
        if cum_prop <= 0.0:
            # fallback: linear proportion of days
            cum_prop = max(last_day / max(days_in_target,1), 1e-6)
        return obs, last_day, cum_prop

    obs_creates, last_obs_day, cum_prop = observed_creates_so_far(df_curr, target_month)
    if obs_creates>0:
        est_full_creates_tm = obs_creates / max(cum_prop, 1e-6)
    else:
        # fallback: use historical average creates for that month-of-year
        hist_cm = df_pred_base[df_pred_base["Create_Month"].notna()]
        hist_cm["moy"] = hist_cm["Create_Month"].dt.month
        moy_avg_creates = hist_cm.groupby("moy")["Create Date"].count() / hist_cm["Create_Month"].nunique()
        est_full_creates_tm = float(moy_avg_creates.get(tm_moy, hist_cm["Create Date"].count() / max(hist_cm["Create_Month"].nunique(),1)))

    # 2) M0 (same-month) expected
    M0_expected = float(est_full_creates_tm * M0_rate)

    # 3) Carryover (lagged) expected
    # For each prior month j, use creates in j from history and apply lag_prob[k] where k = diff months to target
    prior_months = creates_by_CM.index[creates_by_CM.index < target_month]
    carry = 0.0
    for j in prior_months:
        k = (target_month.year - j.year)*12 + (target_month.month - j.month)
        if k <= 0: 
            continue
        p = float(lag_prob.get(k, 0.0))
        c_j = float(creates_by_CM.get(j, 0.0))
        carry += c_j * p
    Carry_expected = float(carry)

    total_forecast = float(M0_expected + Carry_expected)

    # ---------- KPIs ----------
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"<div class='kpi'><div class='label'>M0 (same-month) — {target_month:%b %Y}</div><div class='value'>{int(round(M0_expected)):,}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><div class='label'>Carryover (lagged) — {target_month:%b %Y}</div><div class='value'>{int(round(Carry_expected)):,}</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi'><div class='label'>Total Forecast — {target_month:%b %Y}</div><div class='value'>{int(round(total_forecast)):,}</div></div>", unsafe_allow_html=True)

    # ---------- Chart: history & implied forecast ----------
    # Build monthly actual payment history
    paid_hist = hist.dropna(subset=["Payment Received Date"]).copy()
    monthly_paid = paid_hist["Payment_Month"].value_counts().rename_axis("Month").sort_index().rename("y").reset_index()
    if not monthly_paid.empty:
        monthly_paid["MonthStr"] = pd.to_datetime(monthly_paid["Month"]).dt.strftime("%Y-%m")

    # Create a small DF with target forecast
    fut_df = pd.DataFrame({"Month":[target_month], "yhat":[total_forecast]})
    fut_df["MonthStr"] = fut_df["Month"].dt.strftime("%Y-%m")

    if not monthly_paid.empty:
        hist_line = alt_line(monthly_paid, "MonthStr:O", "y:Q", tooltip=["MonthStr","y"]).encode(color=alt.value("#0ea5e9"))
        fut_line  = alt_line(fut_df,       "MonthStr:O", "yhat:Q", tooltip=["MonthStr","yhat"]).encode(color=alt.value("#ef4444"))
        st.altair_chart(alt.layer(hist_line, fut_line).resolve_scale(y='shared'), use_container_width=True)
    else:
        st.info("No historical payments to plot.")

    # ---------- Split of forecast ----------
    st.markdown("### Split of forecast (optional)")
    if split_by == "None":
        st.info("No split selected.")
    else:
        paid_subset = paid_hist.copy()
        props = historical_split_props(paid_subset, split_by, lookback_months=lookback_split)
        if props.empty or props.sum() == 0:
            st.warning("Not enough data to compute split proportions; showing uniform split across observed categories.")
            cats = paid_subset[split_by].dropna().astype(str).value_counts().index
            if len(cats) > 0:
                props = pd.Series(1/len(cats), index=cats)
        if len(props) > 0:
            split_table = props.rename("Prop").reset_index().rename(columns={"index":split_by})
            split_table["Forecast"] = (total_forecast * split_table["Prop"]).round(0)
            st.dataframe(split_table.sort_values("Forecast", ascending=False), use_container_width=True)
            st.download_button("Download split CSV", split_table.to_csv(index=False).encode("utf-8"),
                               file_name="forecast_split.csv", mime="text/csv")
            ch = alt.Chart(split_table).mark_bar().encode(
                x=alt.X(f"{split_by}:N", title=None),
                y=alt.Y("Forecast:Q", title=None),
                tooltip=[split_by,"Forecast","Prop"]
            )
            st.altair_chart(ch, use_container_width=True)

    # ---------- Day-of-month distribution for target month ----------
    st.markdown("### Day-of-month distribution (payments)")
    def payment_day_profile(df_paid: pd.DataFrame, target_month_ts: pd.Timestamp):
        dp = df_paid.copy()
        dp["d"] = dp["Payment Received Date"].dt.day
        dp["moy"] = dp["Payment Received Date"].dt.month
        pool = dp[dp["moy"] == target_month_ts.month]
        if pool.empty: pool = dp
        cnt = pool["d"].value_counts().sort_index()
        days = month_days(target_month_ts)
        idx = pd.Index(range(1, days+1), name="day")
        cnt = cnt.reindex(idx, fill_value=0)
        total = cnt.sum()
        if total == 0:
            return pd.Series(1.0/days, index=idx, name="prop")
        return (cnt/total).rename("prop")

    pay_profile = payment_day_profile(paid_hist, target_month)
    dom = pay_profile.reset_index().rename(columns={"day":"Day","prop":"Prop"})
    dom["Forecast"] = (total_forecast * dom["Prop"]).round(0)
    st.dataframe(dom, use_container_width=True)
    ch2 = alt.Chart(dom).mark_bar().encode(
        x=alt.X("Day:O", title=None), y=alt.Y("Forecast:Q", title=None),
        tooltip=["Day","Forecast","Prop"]
    )
    st.altair_chart(ch2, use_container_width=True)

    # ---------- Quick backtest (walk-forward on payments) ----------
    st.markdown("### Backtest (walk-forward, payments landing per month)")
    # Build monthly payments series
    mp = monthly_paid.copy()
    if mp.empty or len(mp) < 8:
        st.info("Not enough months to backtest.")
    else:
        mp = mp.sort_values("Month")
        months_list = list(pd.to_datetime(mp["Month"]))
        y_act = mp["y"].astype(float).values
        # Recompute creates_by_CM on the same window to ensure alignment
        cbm = creates_by_CM.reindex(pd.to_datetime(sorted(creates_by_CM.index))).fillna(0.0)

        recs=[]
        window = st.slider("Backtest training window (months)", 6, max(12, len(months_list)-2), min(12, len(months_list)-2), 1)
        for i in range(window, len(months_list)):
            T = months_list[i]  # predict this month
            # Use history up to i-1 to compute lag_prob and M0
            hist_cut = hist[(hist["PM"] < T) | (hist["PM"].isna())]  # payments strictly before T
            # recompute creates & cohort on the cut
            creates_cut = hist_cut.groupby("CM")["Create Date"].count().rename("Creates_CM")
            cohort_cut = hist_cut.dropna(subset=["PM"]).groupby(["CM","PM"])["Payment Received Date"].count().rename("Paid").reset_index()
            cohort_cut["lag_k"] = ((cohort_cut["PM"].dt.year - cohort_cut["CM"].dt.year)*12 + (cohort_cut["PM"].dt.month - cohort_cut["CM"].dt.month)).astype(int)
            cohort_cut = cohort_cut[cohort_cut["lag_k"]>=0]

            total_creates_cut = creates_cut.sum()
            if total_creates_cut <= 0:
                continue
            paid_by_k_cut = cohort_cut.groupby("lag_k")["Paid"].sum()
            lag_prob_cut = (paid_by_k_cut / max(total_creates_cut,1.0)).reindex(range(0,18), fill_value=0.0).clip(lower=0.0)
            M0_cut = float(lag_prob_cut.get(0,0.0))

            # Estimate creates in T (we don't have partial here; use moy average on training window)
            moy_T = T.month
            hist_cm_cut = hist_cut[hist_cut["CM"].notna()].copy()
            hist_cm_cut["moy"] = hist_cm_cut["CM"].dt.month
            if hist_cm_cut["CM"].nunique() > 0:
                moy_avg_creates_cut = hist_cm_cut.groupby("moy")["Create Date"].count() / hist_cm_cut["CM"].nunique()
                est_creates_T = float(moy_avg_creates_cut.get(moy_T, hist_cm_cut["Create Date"].count()/max(hist_cm_cut["CM"].nunique(),1)))
            else:
                est_creates_T = 0.0

            M0_pred = est_creates_T * M0_cut

            # Carryover into T from prior months
            prior_js = creates_cut.index[creates_cut.index < T]
            carry_pred = 0.0
            for j in prior_js:
                k = (T.year - j.year)*12 + (T.month - j.month)
                if k<=0: continue
                p = float(lag_prob_cut.get(k,0.0))
                c_j = float(creates_cut.get(j,0.0))
                carry_pred += c_j * p

            pred_total = float(M0_pred + carry_pred)
            # actual
            actual_row = mp[mp["Month"]==T]
            if actual_row.empty: 
                continue
            actual = float(actual_row["y"].values[0])
            mae = abs(pred_total - actual)
            mape = (mae / actual * 100) if actual>0 else np.nan
            recs.append({"Month":T, "Pred":pred_total, "Actual":actual, "MAE":mae, "MAPE":mape})

        bt = pd.DataFrame(recs)
        if not bt.empty:
            c1,c2 = st.columns(2)
            c1.metric("Avg MAE", f"{bt['MAE'].mean():.1f}")
            c2.metric("Avg MAPE", f"{bt['MAPE'].dropna().mean():.1f}%")
            bt_plot = bt.assign(M=bt["Month"].dt.strftime("%Y-%m"))
            line_bt = alt.layer(
                alt_line(bt_plot, "M:O", "Actual:Q").encode(color=alt.value("#10b981")),
                alt_line(bt_plot, "M:O", "Pred:Q").encode(color=alt.value("#ef4444")),
            ).resolve_scale(y='shared')
            st.altair_chart(line_bt, use_container_width=True)
            st.dataframe(bt.assign(Month=bt["Month"].dt.strftime("%Y-%m")).drop(columns=["Month"]), use_container_width=True)
        else:
            st.info("Backtest window too small after filtering or insufficient data variety. Adjust the slider.")

