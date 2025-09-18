# app.py — JetLearn Insights (MTD/Cohort) — Predictability removed

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# --------------------- Page & Style ---------------------
st.set_page_config(page_title="JetLearn Insights", layout="wide", page_icon="📊")
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
.popcap { font-size:.78rem; color:#64748b; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]
REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country","Student/Academic Counsellor","Deal Stage","Create Date"]

# --------------------- Utilities ---------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try: return pd.read_csv(file_or_path, encoding=enc)
        except Exception: pass
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
    # Preference: Payment Received Date first if present
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
    st.markdown('<div class="head"><div class="title">📊 JetLearn — Insights</div></div>', unsafe_allow_html=True)

# --------------------- Data Source ---------------------
ds = st.expander("Data source", expanded=True)
with ds:
    c1, c2, c3 = st.columns([3,2,2])
    with c1:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with c2:
        path = st.text_input("…or CSV path", value="Master_sheet_DB_10percent.csv")
    with c3:
        exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal'", value=True)

try:
    if uploaded is not None:
        df = robust_read_csv(BytesIO(uploaded.getvalue()))
    else:
        df = robust_read_csv(path)
    st.success("Data loaded ✅")
except Exception as e:
    st.error(str(e)); st.stop()

df.columns = [c.strip() for c in df.columns]
missing=[c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")
    st.stop()

if exclude_invalid:
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"] = df["Create Date"].dt.to_period("M")

# Coerce all other date-like cols for Insights
date_like_cols = detect_measure_date_columns(df)

# --------------------- Single Tab: Insights only ---------------------
tab_insights, = st.tabs(["📋 Insights (MTD/Cohort)"])

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
        options = sorted([v for v in df[colname].dropna().astype(str).unique()])
        all_key = f"{key_prefix}_all"
        ms_key  = f"{key_prefix}_ms"
        # UI container: popover (if available) else expander
        header = f"{label}: " + summary_label(options, True)
        ctx = st.popover(header) if hasattr(st, "popover") else st.expander(header, expanded=False)
        with ctx:
            c1,c2 = st.columns([1,3])
            all_flag = c1.checkbox("All", value=True, key=all_key)
            disabled = st.session_state.get(all_key, True)
            _sel = st.multiselect(label, options=options,
                                  default=options, key=ms_key,
                                  placeholder=f"Type to search {label.lower()}…",
                                  label_visibility="collapsed",
                                  disabled=disabled)
        # Effective values
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
        )
        base = df[mask].copy()

        st.markdown("##### Measures & Windows")
        mcol1,mcol2 = st.columns([3,2])
        with mcol1:
            measures = st.multiselect(f"[{name}] Measure date(s)",
                                      options=date_like_cols,
                                      default=date_like_cols[:1],
                                      key=f"{name}_measures")
        with mcol2:
            mode = st.radio(f"[{name}] Mode", ["MTD","Cohort","Both"], horizontal=True, key=f"{name}_mode")

        # Prepare month cols for measures
        for m in measures:
            mn = f"{m}_Month"
            if m in base.columns and mn not in base.columns:
                base[mn] = base[m].dt.to_period("M")

        # Windows
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
                sub[flg] = ((sub[m].notna()) & (sub[f"{m}_Month"]==sub["Create_Month"]).astype(bool)).astype(int)
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

                # Trend
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

                # Trend (measure-date buckets)
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

    # Scenario A + optional B
    show_b = st.toggle("Enable Scenario B (compare)", value=False)
    left_col, right_col = st.columns(2) if show_b else (st.container(), None)
    with (left_col if show_b else st.container()):
        metaA = scenario_controls("A", df, date_like_cols)
    if show_b:
        with right_col:
            metaB = scenario_controls("B", df, date_like_cols)

    # Compute and render
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
# ====================== NEW TAB: Predictability of Enrollment ======================
# Paste this anywhere AFTER your data (df) is loaded & cleaned.

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, timedelta
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier

# ---------- Config & Safe Palette ----------
try:
    PALETTE  # use existing if present
except NameError:
    PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# ---------- Column expectations ----------
PE_REQUIRED_BASE = ["Create Date", "Country", "JetLearn Deal Source"]
PE_REQUIRED_EXTRA = [
    "Payment Received Date",
    "Number of Sales Activity",
    "Number of Call Connected",
    "Last Date of Sales Activity",
    "Last Date of Call Connected",
]
# "Age" is optional; if missing, we compute age_in_days as (as_of - Create Date)

# ---------- Helpers (namespaced) ----------
def pe_to_ts_month(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d, errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()

def pe_month_start(d: date) -> pd.Timestamp:
    return pd.Timestamp(d).to_period("M").to_timestamp()

def pe_month_add(ts: pd.Timestamp, k: int) -> pd.Timestamp:
    return (ts.to_period("M") + k).to_timestamp()

def pe_asof_guard_train_mask(create_month: pd.Series, as_of: date) -> pd.Series:
    """Include only rows with Create_Month strictly before the as_of month."""
    cutoff = pe_month_start(as_of)
    return create_month < cutoff

def pe_make_feature_frame(raw: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Derive model features from raw df for the chosen as_of date."""
    dfX = raw.copy()

    # Ensure needed date cols are parsed
    for c in ["Create Date", "Payment Received Date", "Last Date of Sales Activity", "Last Date of Call Connected"]:
        if c in dfX.columns:
            dfX[c] = pd.to_datetime(dfX[c], errors="coerce", dayfirst=True)

    # Create_Month for leakage guard
    dfX["Create_Month"] = pe_to_ts_month(dfX["Create Date"])

    # Age features
    asof_ts = pd.Timestamp(as_of)
    if "Age" in dfX.columns:
        # keep provided Age, and also compute robust age_in_days from dates (used if Age missing/invalid)
        dfX["age_in_days"] = (asof_ts - dfX["Create Date"]).dt.days.clip(lower=0)
        # if Age looks numeric, use it as extra signal
        with np.errstate(all="ignore"):
            dfX["Age_num"] = pd.to_numeric(dfX["Age"], errors="coerce")
    else:
        dfX["Age_num"] = np.nan
        dfX["age_in_days"] = (asof_ts - dfX["Create Date"]).dt.days.clip(lower=0)

    # Recency features
    dfX["days_since_last_activity"] = (asof_ts - dfX["Last Date of Sales Activity"]).dt.days
    dfX["days_since_last_call"]     = (asof_ts - dfX["Last Date of Call Connected"]).dt.days
    dfX["days_since_last_activity"] = dfX["days_since_last_activity"].fillna(999).clip(lower=0, upper=9999)
    dfX["days_since_last_call"]     = dfX["days_since_last_call"].fillna(999).clip(lower=0, upper=9999)

    # Create-date calendar features
    dfX["create_dow"] = dfX["Create Date"].dt.dayofweek
    dfX["create_dom"] = dfX["Create Date"].dt.day
    dfX["create_moy"] = dfX["Create Date"].dt.month

    # Activity integers
    for c in ["Number of Sales Activity", "Number of Call Connected"]:
        if c in dfX.columns:
            dfX[c] = pd.to_numeric(dfX[c], errors="coerce").fillna(0).clip(lower=0)

    return dfX

def pe_build_targets(dfX: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Create four binary targets w.r.t. as_of: Today, Tomorrow, ThisMonth, NextMonth."""
    out = dfX.copy()
    pay = out["Payment Received Date"]

    asof_ts = pd.Timestamp(as_of)
    tomorrow = asof_ts + pd.Timedelta(days=1)
    this_m0  = pe_month_start(as_of)
    next_m0  = pe_month_add(this_m0, 1)
    next_m1  = pe_month_add(this_m0, 2) - pd.Timedelta(seconds=1)  # end of next month

    out["y_today"]     = (pay.dt.date == asof_ts.date()).astype(int)
    out["y_tomorrow"]  = (pay.dt.date == tomorrow.date()).astype(int)
    out["y_thismonth"] = ((pay >= asof_ts.floor("D")) & (pay < next_m0)).astype(int)
    out["y_nextmonth"] = ((pay >= next_m0) & (pay < next_m1)).astype(int)

    # "Open as of" (still not paid by as_of) -> used for scoring expected counts
    out["open_as_of"] = pay.isna() | (pay > asof_ts)
    out["created_by_asof"] = out["Create Date"] <= asof_ts
    out["score_mask"] = out["open_as_of"] & out["created_by_asof"]

    return out

def pe_feature_target_splits(dfT: pd.DataFrame, as_of: date):
    """Return X_train, y_train dicts for four horizons + rows_to_score mask."""
    # Exclude current month entirely from training
    train_mask = pe_asof_guard_train_mask(dfT["Create_Month"], as_of)
    X_all = dfT.copy()

    # Selected feature columns
    num_cols = [
        "age_in_days", "Age_num",
        "days_since_last_activity", "days_since_last_call",
        "Number of Sales Activity", "Number of Call Connected",
        "create_dow", "create_dom", "create_moy"
    ]
    cat_cols = ["Country", "JetLearn Deal Source"]

    # Keep only available feature columns
    num_cols = [c for c in num_cols if c in X_all.columns]
    cat_cols = [c for c in cat_cols if c in X_all.columns]

    feats = num_cols + cat_cols
    X_train = X_all.loc[train_mask, feats]
    rows_to_score = X_all["score_mask"]

    ys = {
        "today":     X_all.loc[train_mask, "y_today"],
        "tomorrow":  X_all.loc[train_mask, "y_tomorrow"],
        "thismonth": X_all.loc[train_mask, "y_thismonth"],
        "nextmonth": X_all.loc[train_mask, "y_nextmonth"],
    }

    return feats, num_cols, cat_cols, X_train, ys, rows_to_score

def pe_model_pipeline(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num_pass", "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat_ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    pre = ColumnTransformer(transformers)
    # Small, fast, robust classifier
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def pe_fit_models(X_train, ys, num_cols, cat_cols):
    models = {}
    for key in ["today", "tomorrow", "thismonth", "nextmonth"]:
        y = ys[key]
        # skip if all zeros or NaNs
        if (pd.Series(y).fillna(0).sum() == 0) or (len(pd.Series(y).dropna()) == 0):
            models[key] = None
            continue
        pipe = pe_model_pipeline(num_cols, cat_cols)
        pipe.fit(X_train, y)
        models[key] = pipe
    return models

def pe_predict_counts(models, df_all, feats, rows_to_score, as_of: date):
    """Sum probabilities over open deals to get expected count per horizon."""
    res = {}
    X_score = df_all.loc[rows_to_score, feats]
    for key, model in models.items():
        if model is None or X_score.empty:
            res[key] = 0.0
        else:
            proba = model.predict_proba(X_score)[:, 1]
            res[key] = float(np.sum(proba))
    # Actuals (for today/tomorrow/this/next month) — only meaningful for backtests
    asof_ts = pd.Timestamp(as_of)
    tomorrow = asof_ts + pd.Timedelta(days=1)
    this_m0  = pe_month_start(as_of)
    next_m0  = pe_month_add(this_m0, 1)
    next_m1  = pe_month_add(this_m0, 2) - pd.Timedelta(seconds=1)
    pay = df_all["Payment Received Date"]

    actuals = {
        "today":     int(((pay.dt.date == asof_ts.date())).sum()),
        "tomorrow":  int(((pay.dt.date == tomorrow.date())).sum()),
        "thismonth": int(((pay >= asof_ts.floor("D")) & (pay < next_m0)).sum()),
        "nextmonth": int(((pay >= next_m0) & (pay < next_m1)).sum()),
    }
    return res, actuals

def pe_backtest_last_full_month(df_all: pd.DataFrame, base_as_of: date, models_ready=False):
    """Backtest on the last fully completed month."""
    asof_m0 = pe_month_start(base_as_of)
    test_m0 = pe_month_add(asof_m0, -1)  # last full month
    test_m1 = pe_month_add(test_m0, 1) - pd.Timedelta(seconds=1)

    # We simulate running on the last day of test month
    asof_bt = (test_m1.to_pydatetime().date())

    # Recompute features/targets at backtest as-of, and refit (train excludes test month)
    dfT = pe_make_feature_frame(df_all, asof_bt)
    dfT = pe_build_targets(dfT, asof_bt)
    feats, num_cols, cat_cols, X_train, ys, rows_to_score = pe_feature_target_splits(dfT, asof_bt)
    models = pe_fit_models(X_train, ys, num_cols, cat_cols)
    preds, actuals = pe_predict_counts(models, dfT, feats, rows_to_score, asof_bt)

    # Daily error within the month for "thismonth" horizon:
    # For each day d in [test_m0, test_m1], predict remaining-in-month at d, compare cumulative actuals from d..month_end.
    # (Approximation; still leakage-safe since training excludes the month.)
    daily_rows = []
    days = pd.date_range(test_m0, test_m1, freq="D")
    pay = df_all["Payment Received Date"]
    for d in days:
        as_of_d = d.date()
        dfTd = pe_make_feature_frame(df_all, as_of_d)
        dfTd = pe_build_targets(dfTd, as_of_d)
        featsd, numd, catd, Xtd, ysd, rowsd = pe_feature_target_splits(dfTd, as_of_d)
        md = pe_fit_models(Xtd, ysd, numd, catd)
        p_d, act_d = pe_predict_counts(md, dfTd, featsd, rowsd, as_of_d)
        daily_rows.append({"date": d.date(), "pred_thismonth": p_d["thismonth"]})

    daily_df = pd.DataFrame(daily_rows)
    # Actual remaining-in-month from each day (cumulative from day to month end)
    daily_df["actual_thismonth"] = [
        int(((pay >= pd.Timestamp(d)) & (pay < pe_month_add(test_m0, 1))).sum())
        for d in pd.to_datetime(daily_df["date"])
    ]
    # Errors
    daily_df["ae"] = (daily_df["pred_thismonth"] - daily_df["actual_thismonth"]).abs()
    mae = float(daily_df["ae"].mean())
    mape = float((daily_df["ae"] / daily_df["actual_thismonth"].replace(0, np.nan)).dropna().mean()) if (daily_df["actual_thismonth"]>0).any() else np.nan

    # AUC & Brier for "thismonth" (deal-level)
    try:
        feats_b, num_b, cat_b, X_b, ys_b, _ = pe_feature_target_splits(dfT, asof_bt)
        mdl_b = pe_fit_models(X_b, ys_b, num_b, cat_b)["thismonth"]
        auc = np.nan
        brier = np.nan
        if mdl_b is not None:
            proba_b = mdl_b.predict_proba(X_b)[:,1]
            y_b = ys_b["thismonth"]
            if len(np.unique(y_b)) > 1:
                auc = float(roc_auc_score(y_b, proba_b))
            brier = float(brier_score_loss(y_b, proba_b))
    except Exception:
        auc, brier = np.nan, np.nan

    return {
        "asof_bt": asof_bt,
        "preds": preds,
        "actuals": actuals,
        "daily_df": daily_df,
        "mae": mae, "mape": mape, "auc": auc, "brier": brier
    }

# ---------- UI: Tab ----------
tab_pred_enroll, = st.tabs(["🔮 Predictability of Enrollment"])

with tab_pred_enroll:
    st.markdown("#### Train a fast ML model to estimate enrollments **Today / Tomorrow / This Month / Next Month**")
    # Column checks
    missing = [c for c in PE_REQUIRED_BASE + PE_REQUIRED_EXTRA if c not in df.columns]
    if missing:
        st.warning(f"Missing recommended columns for Predictability: {missing}. "
                   f"We will proceed where possible (using fallbacks for dates/features).")

    # As-of date
    as_of = st.date_input("As-of date", value=pd.Timestamp.today().date(), help="Training will exclude this entire month to avoid leakage.")
    run_bt = st.checkbox("Backtest on last full month (show accuracy)", value=False)
    run_btn = st.button("Train & Predict", type="primary")

    if run_btn:
        with st.spinner("Training models and generating predictions…"):
            # 1) Build feature set & targets for chosen as-of
            dfT = pe_make_feature_frame(df, as_of)
            dfT = pe_build_targets(dfT, as_of)

            # 2) Train (excluding current month)
            feats, num_cols, cat_cols, X_train, ys, rows_to_score = pe_feature_target_splits(dfT, as_of)
            models = pe_fit_models(X_train, ys, num_cols, cat_cols)

            # 3) Predict expected counts by horizon (sum of probabilities over open deals)
            pred_counts, actual_counts = pe_predict_counts(models, dfT, feats, rows_to_score, as_of)

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(f"Predicted Today ({pd.Timestamp(as_of):%d %b})", f"{int(round(pred_counts['today'])):,}")
            k2.metric(f"Predicted Tomorrow ({(pd.Timestamp(as_of)+pd.Timedelta(days=1)):%d %b})", f"{int(round(pred_counts['tomorrow'])):,}")
            k3.metric(f"Predicted This Month ({pe_month_start(as_of):%b %Y})", f"{int(round(pred_counts['thismonth'])):,}")
            k4.metric(f"Predicted Next Month ({pe_month_add(pe_month_start(as_of),1):%b %Y})", f"{int(round(pred_counts['nextmonth'])):,}")

            # Deal-level scores table (download)
            score_idx = dfT[rows_to_score].index
            out = pd.DataFrame({
                "Deal Index": score_idx,
                "Country": dfT.loc[score_idx, "Country"] if "Country" in dfT.columns else "",
                "JetLearn Deal Source": dfT.loc[score_idx, "JetLearn Deal Source"] if "JetLearn Deal Source" in dfT.columns else "",
            })
            for key, mdl in models.items():
                if mdl is None or score_idx.empty:
                    out[f"p_{key}"] = 0.0
                else:
                    out[f"p_{key}"] = mdl.predict_proba(dfT.loc[score_idx, feats])[:,1]
            st.markdown("**Open deals as of selected date (with predicted probabilities)**")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download Predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="predictability_open_deals.csv", mime="text/csv")

            # Optional backtest
            if run_bt:
                bt = pe_backtest_last_full_month(df, as_of)
                st.markdown("---")
                st.markdown(f"##### Backtest (last full month) — As-of simulated at **{pd.Timestamp(bt['asof_bt']):%d %b %Y}**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted This Month", f"{int(round(bt['preds']['thismonth'])):,}")
                m2.metric("Actual This Month", f"{bt['actuals']['thismonth']:,}")
                m3.metric("MAE (daily, this month)", f"{bt['mae']:.2f}")
                m4.metric("MAPE (daily, this month)", f"{(bt['mape']*100):.1f}%" if pd.notna(bt['mape']) else "—")

                # AUC/Brier for deal-level classification (this month)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("AUC (This Month, deal-level)", f"{bt['auc']:.3f}" if pd.notna(bt['auc']) else "—")
                with c2:
                    st.metric("Brier Score (This Month, deal-level)", f"{bt['brier']:.3f}" if pd.notna(bt['brier']) else "—")

                # Chart: Pred vs Actual by day (remaining in month from each day)
                if not bt["daily_df"].empty:
                    dd = bt["daily_df"].copy()
                    dd["date"] = pd.to_datetime(dd["date"])
                    long = dd.melt(id_vars="date", value_vars=["pred_thismonth","actual_thismonth"],
                                   var_name="Series", value_name="Count")
                    ch = alt.Chart(long).mark_line(point=True).encode(
                        x=alt.X("date:T", title=None),
                        y=alt.Y("Count:Q", title=None),
                        color=alt.Color("Series:N", scale=alt.Scale(range=PALETTE)),
                        tooltip=["date:T","Series:N","Count:Q"]
                    ).properties(height=280)
                    st.altair_chart(ch, use_container_width=True)

    st.caption("Training excludes the entire current month to prevent leakage. "
               "Expected counts are computed by summing predicted probabilities over open deals as of the selected date.")
# ==================== END: Predictability of Enrollment ====================
  
