# app.py  — JetLearn Insights (MTD/Cohort) + Predictability in one Streamlit app

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
    st.markdown('<div class="head"><div class="title">📊 JetLearn — Insights & Predictability</div></div>', unsafe_allow_html=True)

# --------------------- Data Source (shared by both tabs) ---------------------
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

# --------------------- Tabs ---------------------
tab_insights, tab_predict = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictability"])

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

# =========================================================
# ================  PREDICTABILITY (Standalone) ===========
# =========================================================
with tab_predict:
    # ---- Helpers specific to predictability ----
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

    def month_start(dt: date) -> pd.Timestamp:
        return pd.Timestamp(dt).to_period("M").to_timestamp()

    def month_days(dt: date) -> int:
        m0 = month_start(dt)
        m1 = (m0 + pd.offsets.MonthBegin(1))
        return int((m1 - m0).days)

    def make_month_table_from_dates(dates: pd.Series) -> pd.DataFrame:
        m = pd.to_datetime(dates, errors="coerce").dt.to_period("M").dt.to_timestamp()
        monthly = m.value_counts().rename_axis("Month").sort_index().rename("y").reset_index()
        return monthly

    def add_calendar_features(months: pd.DatetimeIndex) -> pd.DataFrame:
        dfc = pd.DataFrame({"Month": months})
        dfc["t"] = np.arange(len(dfc))  # trend
        dfc["moy"] = dfc["Month"].dt.month
        for m in range(1,12):  # drop December to avoid dummy trap
            dfc[f"m_{m}"] = (dfc["moy"]==m).astype(int)
        return dfc.drop(columns=["moy"])

    def fit_poisson_or_none(hist_months: pd.DatetimeIndex, y: np.ndarray, alpha=0.5):
        try:
            from sklearn.linear_model import PoissonRegressor
            X = add_calendar_features(hist_months).drop(columns=["Month"])
            if len(X) < 12:
                return None
            model = PoissonRegressor(alpha=alpha, max_iter=1000)
            model.fit(X, y)
            return model
        except Exception:
            return None

    def forecast_months(hist_months: pd.DatetimeIndex, y: np.ndarray, horizon: int, alpha=0.5):
        model = fit_poisson_or_none(hist_months, y, alpha=alpha)
        last = hist_months[-1]
        fut = pd.date_range(start=last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        if model is not None:
            Xf = add_calendar_features(fut).drop(columns=["Month"])
            yhat = np.maximum(0.0, model.predict(Xf))
        else:
            hist = pd.DataFrame({"Month": hist_months, "y": y})
            hist["moy"] = hist["Month"].dt.month
            moy_avg = hist.groupby("moy")["y"].mean()
            yhat = np.array([moy_avg.get(m, np.mean(y)) for m in fut.month], dtype=float)
        return pd.DataFrame({"Month": fut, "yhat": yhat})

    def forecast_this_and_next_month(hist_months: pd.DatetimeIndex, y: np.ndarray, alpha=0.5):
        today = pd.Timestamp.today().normalize()
        cm = pd.Timestamp(year=today.year, month=today.month, day=1)
        mask = hist_months < cm
        hm, hy = (hist_months[mask], y[mask]) if mask.sum()>=6 else (hist_months, y)
        model = fit_poisson_or_none(hm, hy, alpha=alpha)
        months_to_predict = pd.DatetimeIndex([cm, cm + pd.offsets.MonthBegin(1)])
        if model is not None:
            Xf = add_calendar_features(months_to_predict).drop(columns=["Month"])
            yhat = np.maximum(0.0, model.predict(Xf))
        else:
            hist = pd.DataFrame({"Month": hm, "y": hy})
            hist["moy"] = hist["Month"].dt.month
            moy_avg = hist.groupby("moy")["y"].mean()
            yhat = np.array([moy_avg.get(m, np.mean(hy)) for m in months_to_predict.month], dtype=float)
        return {"this_month": float(yhat[0]), "next_month": float(yhat[1])}

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
        dfp["Month"] = dfp["PaymentMonth"]
        global_counts = dfp.groupby(split_col)["Payment Received Date"].count()
        gp_total = global_counts.sum()
        prior_props = (global_counts / gp_total) if gp_total>0 else global_counts
        months = sorted(dfp["Month"].dropna().unique())
        take = months[-lookback_months:] if len(months)>=lookback_months else months
        recent = dfp[dfp["Month"].isin(take)]
        recent_counts = recent.groupby(split_col)["Payment Received Date"].count().sort_values(ascending=False)
        return eb_smooth_props(recent_counts, prior_props, prior_strength=5.0)

    def day_of_month_profile(df_paid: pd.DataFrame, target_month: pd.Timestamp):
        if df_paid.empty:
            return None
        dfp = df_paid.copy()
        dfp["d"] = dfp["Payment Received Date"].dt.day
        dfp["moy"] = dfp["Payment Received Date"].dt.month
        moy = int(target_month.month)
        pool = dfp[dfp["moy"] == moy]
        if pool.empty: pool = dfp
        cnt = pool["d"].value_counts().sort_index()
        days_in_target = month_days(target_month)
        idx = pd.Index(range(1, days_in_target+1), name="day")
        cnt = cnt.reindex(idx, fill_value=0)
        total = cnt.sum()
        if total == 0:
            return pd.Series(1.0/days_in_target, index=idx, name="prop")
        return (cnt/total).rename("prop")

    # ---- Detect payments and prepare ----
    PAYMENT_COL = None
    PAYMENT_COL = detect_payment_col(df.columns)
    if PAYMENT_COL is None:
        st.error("Couldn't find a payment date column. Add one like 'Payment Received Date'.")
        st.stop()

    df_paid = df.copy()
    df_paid["Payment Received Date"] = pd.to_datetime(df_paid[PAYMENT_COL], errors="coerce", dayfirst=True)
    paid = df_paid[df_paid["Payment Received Date"].notna()].copy()
    if paid.empty:
        st.error("No non-empty payment dates found.")
        st.stop()
    paid["PaymentMonth"] = paid["Payment Received Date"].dt.to_period("M").dt.to_timestamp()

    # ---- Controls ----
    st.markdown("### Forecast controls")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        horizon = st.slider("Forecast horizon (months)", 1, 12, 6, 1)
    with cc2:
        alpha = st.slider("Poisson regularization (alpha)", 0.0, 2.0, 0.5, 0.1)
    with cc3:
        split_by = st.selectbox("Split forecast by", ["None","JetLearn Deal Source","Student/Academic Counsellor","Country","Pipeline"])
    with cc4:
        lookback_split = st.slider("Split lookback (months)", 1, 12, 6, 1)

    # ---- Monthly history & forecast ----
    monthly = make_month_table_from_dates(paid["Payment Received Date"]).sort_values("Month")
    hist_months = pd.to_datetime(monthly["Month"]); y = monthly["y"].astype(float).values
    fut = forecast_months(hist_months, y, horizon=horizon, alpha=alpha)
    fut["MonthStr"] = fut["Month"].dt.strftime("%Y-%m")
    monthly["MonthStr"] = pd.to_datetime(monthly["Month"]).dt.strftime("%Y-%m")

    st.markdown("### History + Forecast")
    hist_line = alt_line(monthly, "MonthStr:O", "y:Q", tooltip=["MonthStr","y"]).encode(color=alt.value("#0ea5e9"))
    fut_line  = alt_line(fut,     "MonthStr:O", "yhat:Q", tooltip=["MonthStr","yhat"]).encode(color=alt.value("#ef4444"))
    st.altair_chart(alt.layer(hist_line, fut_line).resolve_scale(y='shared'), use_container_width=True)

    # ---- KPIs Today / Tomorrow / This Month / Next Month ----
    k = forecast_this_and_next_month(hist_months, y, alpha=alpha)
    this_month_forecast = k["this_month"]; next_month_forecast = k["next_month"]
    today_dt = pd.Timestamp.today().date()
    cm_start = month_start(today_dt)
    dom_prop = day_of_month_profile(paid, cm_start)
    def safe_take_prop(prop_series, day_idx):
        if prop_series is None: return 0.0
        return float(prop_series.reindex(range(1, len(prop_series)+1)).get(day_idx, 0.0))
    today_n  = int(round(this_month_forecast * safe_take_prop(dom_prop, today_dt.day)))
    tom_dt   = today_dt + timedelta(days=1)
    tom_n    = int(round(this_month_forecast * safe_take_prop(dom_prop, tom_dt.day)))

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f"<div class='kpi'><div class='label'>Today ({today_dt:%d %b})</div><div class='value'>{today_n:,}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi'><div class='label'>Tomorrow ({tom_dt:%d %b})</div><div class='value'>{tom_n:,}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi'><div class='label'>This Month ({cm_start:%b %Y})</div><div class='value'>{int(round(this_month_forecast)):,}</div></div>", unsafe_allow_html=True)
    with k4:
        nm = (cm_start + pd.offsets.MonthBegin(1))
        st.markdown(f"<div class='kpi'><div class='label'>Next Month ({nm:%b %Y})</div><div class='value'>{int(round(next_month_forecast)):,}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    # ---- Inspect a forecast month & Split ----
    st.markdown("### Inspect a forecast month")
    tm1, tm2 = st.columns([1.2,3])
    with tm1:
        options = list((pd.date_range(start=cm_start, periods=max(horizon, 2), freq="MS")).to_pydatetime())
        default_index = 1
        target_month = st.selectbox("Pick a month", options=options, index=default_index, format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))

    target_month_ts = pd.Timestamp(target_month)
    if target_month_ts == cm_start:
        pred_total = this_month_forecast
    elif target_month_ts == (cm_start + pd.offsets.MonthBegin(1)):
        pred_total = next_month_forecast
    else:
        row = fut[fut["Month"]==target_month_ts]
        pred_total = float(row["yhat"].iloc[0]) if not row.empty else float(next_month_forecast)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"<div class='kpi'><div class='label'>Forecast — {target_month_ts:%b %Y}</div><div class='value'>{int(round(pred_total)):,}</div></div>", unsafe_allow_html=True)

    st.markdown("### Split of forecast (category proportions)")
    if split_by == "None":
        st.info("No split selected.")
    else:
        props = historical_split_props(paid, split_by, lookback_months=lookback_split)
        if props.empty or props.sum() == 0:
            st.warning("Not enough data to compute split proportions; showing uniform split across observed categories.")
            cats = paid[split_by].dropna().astype(str).value_counts().index
            if len(cats) == 0:
                st.info("No categories available for split.")
            else:
                props = pd.Series(1/len(cats), index=cats)

        if len(props) > 0:
            split_table = props.rename("Prop").reset_index().rename(columns={"index":split_by})
            split_table["Forecast"] = (pred_total * split_table["Prop"]).round(0)
            st.dataframe(split_table.sort_values("Forecast", ascending=False), use_container_width=True)
            st.download_button("Download split CSV", split_table.to_csv(index=False).encode("utf-8"),
                               file_name="forecast_split.csv", mime="text/csv")
            ch = alt.Chart(split_table).mark_bar().encode(
                x=alt.X(f"{split_by}:N", title=None),
                y=alt.Y("Forecast:Q", title=None),
                tooltip=[split_by,"Forecast","Prop"]
            )
            st.altair_chart(ch, use_container_width=True)

    # ---- Day-of-month distribution for chosen month ----
    st.markdown("### Day-of-month distribution")
    prop_dom = day_of_month_profile(paid, target_month_ts)
    if prop_dom is None:
        st.info("Not enough data for daily profile.")
    else:
        dom = prop_dom.reset_index().rename(columns={"day":"Day","prop":"Prop"})
        dom["Forecast"] = (pred_total * dom["Prop"]).round(0)
        st.dataframe(dom, use_container_width=True)
        ch2 = alt.Chart(dom).mark_bar().encode(
            x=alt.X("Day:O", title=None), y=alt.Y("Forecast:Q", title=None),
            tooltip=["Day","Forecast","Prop"]
        )
        st.altair_chart(ch2, use_container_width=True)

    # ---- Quick backtest (walk-forward) ----
    st.markdown("### Backtest (walk-forward, quick)")
    bt_window = st.slider("Training window (months)", 6, 24, 12, 1)
    hist = monthly.copy()
    hist["Month"] = pd.to_datetime(hist["Month"])
    hist = hist.sort_values("Month")

    if len(hist) <= bt_window + 2:
        st.info("Not enough history to backtest with the chosen window.")
    else:
        tests = min(12, len(hist) - bt_window - 1)
        recs = []
        for i in range(bt_window, bt_window + tests):
            train = hist.iloc[:i].copy()
            test  = hist.iloc[i:i+1].copy()
            m_train = train["Month"].values
            y_train = train["y"].astype(float).values
            m_test  = test["Month"].values[0]
            model = fit_poisson_or_none(pd.to_datetime(m_train), y_train, alpha=alpha)
            if model is not None:
                Xf = add_calendar_features(pd.DatetimeIndex([m_test])).drop(columns=["Month"])
                pred = float(max(0.0, model.predict(Xf)[0]))
            else:
                train["moy"] = pd.to_datetime(train["Month"]).dt.month
                moy_avg = train.groupby("moy")["y"].mean()
                pred = float(moy_avg.get(pd.Timestamp(m_test).month, train["y"].mean()))
            actual = float(test["y"].values[0])
            mae = abs(pred - actual)
            mape = (mae / actual * 100) if actual>0 else np.nan
            recs.append({"Month": m_test, "Pred":pred, "Actual":actual, "MAE":mae, "MAPE":mape})
        bt = pd.DataFrame(recs)
        if not bt.empty:
            c1,c2 = st.columns(2)
            c1.metric("Avg MAE", f"{bt['MAE'].mean():.1f}")
            c2.metric("Avg MAPE", f"{bt['MAPE'].dropna().mean():.1f}%")
            line_bt = alt.layer(
                alt_line(bt.assign(M=bt["Month"].dt.strftime("%Y-%m")), "M:O", "Actual:Q").encode(color=alt.value("#10b981")),
                alt_line(bt.assign(M=bt["Month"].dt.strftime("%Y-%m")), "M:O", "Pred:Q").encode(color=alt.value("#ef4444")),
            ).resolve_scale(y='shared')
            st.altair_chart(line_bt, use_container_width=True)
            st.dataframe(bt.assign(Month=bt["Month"].dt.strftime("%Y-%m")).drop(columns=["Month"]), use_container_width=True)
