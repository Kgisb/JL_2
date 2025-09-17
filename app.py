# app.py — JetLearn Insights (MTD/Cohort) + ML Predictability (Payment Count, leak-free + inflow)
# Run: streamlit run app.py

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
tab_insights, tab_predict = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictability (ML)"])

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

    def unified_multifilter(label, df_, colname, key_prefix):
        options = sorted([v for v in df_[colname].dropna().astype(str).unique()])
        all_key = f"{key_prefix}_all"
        ms_key  = f"{key_prefix}_ms"
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

    def scenario_controls(name: str, df_: pd.DataFrame, date_like_cols_):
        st.markdown(f"**Scenario {name}** <span class='badge'>independent</span>", unsafe_allow_html=True)
        pipe_all, pipe_sel = unified_multifilter("Pipeline", df_, "Pipeline", f"{name}_pipe")
        src_all,  src_sel  = unified_multifilter("Deal Source", df_, "JetLearn Deal Source", f"{name}_src")
        cty_all,  cty_sel  = unified_multifilter("Country", df_, "Country", f"{name}_cty")
        csl_all,  csl_sel  = unified_multifilter("Counsellor", df_, "Student/Academic Counsellor", f"{name}_csl")

        mask = (
            in_filter(df_["Pipeline"], pipe_all, pipe_sel) &
            in_filter(df_["JetLearn Deal Source"], src_all, src_sel) &
            in_filter(df_["Country"], cty_all, cty_sel) &
            in_filter(df_["Student/Academic Counsellor"], csl_all, csl_sel)
        )
        base = df_[mask].copy()

        st.markdown("##### Measures & Windows")
        mcol1,mcol2 = st.columns([3,2])
        with mcol1:
            measures = st.multiselect(f"[{name}] Measure date(s)",
                                      options=date_like_cols_,
                                      default=date_like_cols_[:1],
                                      key=f"{name}_measures")
        with mcol2:
            mode = st.radio(f"[{name}] Mode", ["MTD","Cohort","Both"], horizontal=True, key=f"{name}_mode")

        for m in measures:
            mn = f"{m}_Month"
            if m in base.columns and mn not in base.columns:
                base[mn] = base[m].dt.to_period("M")

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
        tA, tB, _ = st.tabs(["📋 Scenario A", "📋 Scenario B", "🧠 Compare"])
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
# ==============  PREDICTABILITY (ML, simple) =============
# =========================================================
with tab_predict:
    st.markdown("### 🔮 Predictability (ML) — Payment Count (leak-free & includes new-deal inflow)")
    st.caption("Trains on **all history excluding the running month** (Asia/Kolkata). Target = count of 'Payment Received Date'.")

    # ---------- Column detection ----------
    colmap = {c.lower(): c for c in df.columns}
    def pick_exact(name_like): return colmap.get(name_like.lower())

    COL_CREATE   = pick_exact("create date")
    COL_COUNTRY  = pick_exact("country")
    COL_SOURCE   = pick_exact("jetlearn deal source")
    COL_CSL      = pick_exact("student/academic counsellor")

    # detect payment date col by keywords
    COL_PAY = None
    for c in df.columns:
        cl = c.lower()
        if "payment" in cl and "received" in cl and "date" in cl:
            COL_PAY = c; break
    if COL_PAY is None:
        for c in df.columns:
            cl = c.lower()
            if "payment" in cl and "date" in cl:
                COL_PAY = c; break

    COL_TIMES = next((c for c in df.columns if "number of times contacted" in c.lower()), None)
    COL_SALES = next((c for c in df.columns if "number of sales activities" in c.lower()), None)
    COL_LAST_ACT = next((c for c in df.columns if "last activity date" in c.lower()), None)
    COL_LAST_CNT = next((c for c in df.columns if "last contacted" in c.lower()), None)

    need = [("Create Date",COL_CREATE), ("Payment Received Date",COL_PAY)]
    miss = [n for n,c in need if c is None]
    if miss:
        st.error(f"Missing required column(s): {', '.join(miss)}")
        st.stop()

    # ---------- Parse & prep ----------
    X = df.copy()
    X[COL_CREATE] = pd.to_datetime(X[COL_CREATE], errors="coerce", dayfirst=True)
    X[COL_PAY]    = pd.to_datetime(X[COL_PAY],    errors="coerce", dayfirst=True)
    if COL_LAST_ACT: X[COL_LAST_ACT] = pd.to_datetime(X[COL_LAST_ACT], errors="coerce", dayfirst=True)
    if COL_LAST_CNT: X[COL_LAST_CNT] = pd.to_datetime(X[COL_LAST_CNT], errors="coerce", dayfirst=True)
    for num_col in [COL_TIMES, COL_SALES]:
        if num_col: X[num_col] = pd.to_numeric(X[num_col], errors="coerce").fillna(0)

    # ---------- Leak-free cutoff (Asia/Kolkata) ----------
    tz = "Asia/Kolkata"
    now_ist = pd.Timestamp.now(tz=tz)
    cm_start = pd.Timestamp(year=now_ist.year, month=now_ist.month, day=1, tz=tz).tz_convert(None)

    # ---------- Breakdowns UI ----------
    group_options = []
    if COL_CSL: group_options.append("Student/Academic Counsellor")
    if COL_SOURCE: group_options.append("JetLearn Deal Source")
    if COL_COUNTRY: group_options.append("Country")
    group_options += ["Day", "Day of Week"]
    group_by = st.multiselect("Breakdowns (group by)", options=group_options,
                              default=[x for x in ["JetLearn Deal Source","Country"] if x in group_options])

    # ---------- Build training: positives + sampled negatives ----------
    NEG_PER_POS = 5
    MAX_AGE_DAYS = 150

    def build_training_simple(X_: pd.DataFrame) -> pd.DataFrame:
        D = X_[X_[COL_CREATE].notna()].copy().reset_index(drop=True)
        if D.empty: return pd.DataFrame()
        D["deal_id"] = np.arange(len(D))

        # Choose columns safely
        base_cols = ["deal_id", COL_CREATE, COL_COUNTRY, COL_SOURCE, COL_CSL, COL_TIMES, COL_SALES, COL_LAST_ACT, COL_LAST_CNT]
        base_cols = [c for c in base_cols if c is not None]

        # positives: payments < current month
        pos_cols = base_cols + [COL_PAY]
        pos = D[D[COL_PAY].notna() & (D[COL_PAY] < cm_start)][pos_cols].copy()
        pos.rename(columns={COL_PAY: "day"}, inplace=True)
        pos["y"] = 1.0

        rng = np.random.default_rng(42)
        neg_rows = []

        # negatives from paid deals pre-pay
        for _, r in pos.iterrows():
            d0 = pd.to_datetime(r[COL_CREATE]).normalize()
            dp = pd.to_datetime(r["day"]).normalize()
            d1 = min(dp - pd.Timedelta(days=1), cm_start - pd.Timedelta(days=1))
            if d1 < d0: continue
            span = (d1.date() - d0.date()).days + 1
            take = min(span, NEG_PER_POS)
            offs = rng.choice(span, size=take, replace=False)
            for o in offs:
                row = r.to_dict()
                row["day"] = (d0 + pd.Timedelta(days=int(o)))
                row["y"] = 0.0
                neg_rows.append(row)

        # negatives from unpaid-as-of-cutoff
        unpaid = D[D[COL_PAY].isna() & (D[COL_CREATE] < cm_start)][base_cols].copy()
        for _, r in unpaid.iterrows():
            d0 = pd.to_datetime(r[COL_CREATE]).normalize()
            d1 = cm_start - pd.Timedelta(days=1)
            if d1 < d0: continue
            span = (d1.date() - d0.date()).days + 1
            take = min(span, NEG_PER_POS)
            offs = rng.choice(span, size=take, replace=False)
            for o in offs:
                row = r.to_dict()
                row["day"] = (d0 + pd.Timedelta(days=int(o)))
                row["y"] = 0.0
                neg_rows.append(row)

        neg = pd.DataFrame(neg_rows) if neg_rows else pd.DataFrame(columns=pos.columns)
        train = pd.concat([pos, neg], ignore_index=True)
        if train.empty: return train

        # features
        train["day"] = pd.to_datetime(train["day"]).dt.normalize()
        train["age"] = (train["day"] - pd.to_datetime(train[COL_CREATE]).dt.normalize()).dt.days.clip(lower=0, upper=MAX_AGE_DAYS).astype(int)
        train["moy"] = train["day"].dt.month.astype(int)
        train["dow"] = train["day"].dt.dayofweek.astype(int)
        train["dom"] = train["day"].dt.day.astype(int)

        def recency(series_name):
            if series_name is None or series_name not in train.columns: return 365
            d = pd.to_datetime(train[series_name], errors="coerce")
            return (train["day"] - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        train["rec_act"] = recency(COL_LAST_ACT)
        train["rec_cnt"] = recency(COL_LAST_CNT)

        # normalize numeric feature names used by the model
        train["times"] = pd.to_numeric(train[COL_TIMES], errors="coerce").fillna(0) if COL_TIMES in train.columns else 0
        train["sales"] = pd.to_numeric(train[COL_SALES], errors="coerce").fillna(0) if COL_SALES in train.columns else 0

        keep = ["deal_id","day","y","age","moy","dow","dom","rec_act","rec_cnt","times","sales"]
        for c in [COL_COUNTRY, COL_SOURCE, COL_CSL]:
            if c is not None and c in train.columns: keep.append(c)
        return train[keep].copy()

    with st.spinner("Preparing & training…"):
        train = build_training_simple(X)
        if train.empty:
            st.info("Not enough historical data to train."); st.stop()

        # model
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import HistGradientBoostingClassifier

        num_cols = [c for c in ["age","moy","dow","dom","rec_act","rec_cnt","times","sales"] if c in train.columns]
        cat_cols = [c for c in [COL_COUNTRY, COL_SOURCE, COL_CSL] if c is not None and c in train.columns]

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        pre = ColumnTransformer(
            transformers=[
                ("num","passthrough", num_cols),
                ("cat", ohe, cat_cols)
            ],
            remainder="drop"
        )
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_leaf_nodes=31,
            max_iter=250,          # correct param (not n_estimators)
            early_stopping=True,
            random_state=42
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(train.drop(columns=["y","deal_id","day"], errors="ignore"), train["y"])

    # ---------- Score existing pipeline (today → end of next month) ----------
    today = pd.Timestamp.now(tz=tz).tz_convert(None).normalize()
    end_this = today.to_period("M").to_timestamp("M")
    end_next = (today.to_period("M") + 1).to_timestamp("M")
    dates = pd.date_range(start=today, end=end_next, freq="D")

    deals = X.copy()
    deals["deal_id"] = np.arange(len(deals))
    deals["Create"] = pd.to_datetime(deals[COL_CREATE]).dt.normalize()
    deals["Pay"]    = pd.to_datetime(deals[COL_PAY]).dt.normalize()

    # make sure placeholder numeric cols exist BEFORE selecting
    if COL_TIMES is None: deals["times"] = 0
    if COL_SALES is None: deals["sales"] = 0

    sel_cols = ["deal_id","Create","Pay"]
    for c in [COL_COUNTRY, COL_SOURCE, COL_CSL, COL_TIMES if COL_TIMES else "times", COL_SALES if COL_SALES else "sales", COL_LAST_ACT, COL_LAST_CNT]:
        if c is not None and c in deals.columns: sel_cols.append(c)

    base_deals = deals[sel_cols].copy()

    cart = base_deals.assign(key=1).merge(pd.DataFrame({"day": dates, "key":1}), on="key").drop(columns=["key"])
    cart = cart[(cart["Create"] <= cart["day"]) & (cart["Pay"].isna() | (cart["Pay"] >= cart["day"]))].copy()

    # features for scoring
    cart["age"] = (cart["day"] - cart["Create"]).dt.days.clip(lower=0, upper=MAX_AGE_DAYS).astype(int)
    cart["moy"] = cart["day"].dt.month.astype(int)
    cart["dow"] = cart["day"].dt.dayofweek.astype(int)
    cart["dom"] = cart["day"].dt.day.astype(int)

    def recency_s(col):
        if col is None or col not in cart.columns: return 365
        d = pd.to_datetime(cart[col], errors="coerce")
        return (cart["day"] - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
    cart["rec_act"] = recency_s(COL_LAST_ACT)
    cart["rec_cnt"] = recency_s(COL_LAST_CNT)

    # normalize numeric names
    if COL_TIMES and COL_TIMES in cart.columns:
        cart["times"] = pd.to_numeric(cart[COL_TIMES], errors="coerce").fillna(0)
    else:
        cart["times"] = pd.to_numeric(cart.get("times", 0), errors="coerce").fillna(0)
    if COL_SALES and COL_SALES in cart.columns:
        cart["sales"] = pd.to_numeric(cart[COL_SALES], errors="coerce").fillna(0)
    else:
        cart["sales"] = pd.to_numeric(cart.get("sales", 0), errors="coerce").fillna(0)

    # predict probability → expected count per deal-day
    drop_cols = ["deal_id","Create","Pay","day"]
    Xscore = cart.drop(columns=[c for c in drop_cols if c in cart.columns], errors="ignore")
    proba = pipe.predict_proba(Xscore)[:,1]
    cart["p"] = proba

    cart["Day"] = cart["day"].dt.date.astype(str)
    cart["Day of Week"] = cart["day"].dt.day_name()

    # ---------- New-deal inflow (from last full month DOW create-rate, with M0/M1 spill) ----------
    def month_code(ts): return ts.dt.year * 12 + ts.dt.month

    gcols_full = [c for c in [COL_SOURCE, COL_COUNTRY, COL_CSL] if c is not None]

    dfC = X.copy()
    dfC["Create_M"] = dfC[COL_CREATE].dt.to_period("M").dt.to_timestamp()
    dfC["Pay_M"]    = dfC[COL_PAY].dt.to_period("M").dt.to_timestamp()
    dfC_hist = dfC[dfC["Create_M"].notna() & (dfC["Create_M"] < cm_start)].copy()

    for c in gcols_full:
        dfC_hist[c] = dfC_hist[c].fillna("NA").astype(str)

    cmcode = month_code(dfC_hist["Create_M"])
    pmcode = month_code(dfC_hist["Pay_M"])
    dfC_hist["Lag"] = (pmcode - cmcode)

    global_trials = dfC_hist.groupby(gcols_full)["Create_M"].count().rename("trials")
    succ0 = dfC_hist[dfC_hist["Lag"]==0].groupby(gcols_full)["Lag"].count().rename("succ0")
    succ1 = dfC_hist[dfC_hist["Lag"]==1].groupby(gcols_full)["Lag"].count().rename("succ1")
    rates = pd.concat([global_trials, succ0, succ1], axis=1).fillna(0)

    g_trials = int(rates["trials"].sum()) if not rates.empty else 0
    g_r0 = (rates["succ0"].sum()/g_trials) if g_trials>0 else 0.0
    g_r1 = (rates["succ1"].sum()/g_trials) if g_trials>0 else 0.0
    alpha = 20.0
    if not rates.empty:
        rates["r0"] = (rates["succ0"] + alpha*g_r0) / (rates["trials"] + alpha)
        rates["r1"] = (rates["succ1"] + alpha*g_r1) / (rates["trials"] + alpha)
        rates = rates.reset_index()
    else:
        rates = pd.DataFrame(columns=gcols_full+["trials","succ0","succ1","r0","r1"])

    lm_start = (cm_start - pd.offsets.MonthBegin(1))
    lm_end   = (cm_start - pd.Timedelta(days=1))
    lastm = X[(X[COL_CREATE] >= lm_start) & (X[COL_CREATE] <= lm_end)].copy()
    for c in gcols_full:
        lastm[c] = lastm[c].fillna("NA").astype(str)
    lastm["dow"] = lastm[COL_CREATE].dt.dayofweek

    days_last_month = pd.date_range(lm_start, lm_end, freq="D")
    dow_counts = pd.Series(days_last_month.dayofweek).value_counts().to_dict()
    for d in range(7): dow_counts.setdefault(d, 0)

    if lastm.empty:
        creates_dow = pd.DataFrame(columns=gcols_full+["dow","creates","rate_per_day"])
    else:
        creates_dow = lastm.groupby(gcols_full + ["dow"])[COL_CREATE].count().rename("creates").reset_index()
        creates_dow["rate_per_day"] = creates_dow.apply(lambda r: (r["creates"] / max(1, dow_counts[int(r["dow"])])), axis=1)

    # functions to get expected creates per day for a date range
    def expected_creates_for_month(day_range):
        if creates_dow.empty:
            return pd.DataFrame(columns=gcols_full+["day","exp_creates"])
        recs=[]
        rows = creates_dow.to_dict("records")
        for d in day_range:
            dw = int(d.dayofweek)
            for r in rows:
                if int(r["dow"]) == dw:
                    recs.append(tuple([*(r[c] for c in gcols_full), d, float(r["rate_per_day"])]))
        if not recs:
            return pd.DataFrame(columns=gcols_full+["day","exp_creates"])
        E = pd.DataFrame(recs, columns=gcols_full+["day","exp_creates"])
        return E.groupby(gcols_full+["day"], dropna=False)["exp_creates"].sum().reset_index()

    future_days = pd.date_range(start=today, end=end_next, freq="D")
    days_remaining_this = pd.date_range(start=today, end=end_this, freq="D")
    days_next_month = pd.date_range(start=end_this + pd.Timedelta(days=1), end=end_next, freq="D")
    n_rem = max(1, len(days_remaining_this))
    n_next = max(1, len(days_next_month))

    exp_this = expected_creates_for_month(days_remaining_this)
    exp_next = expected_creates_for_month(days_next_month)

    if rates.empty:
        # no history → no inflow
        inflow = pd.DataFrame(columns=gcols_full+["day","p"])
    else:
        rates_key = rates.set_index(gcols_full)[["r0","r1"]].to_dict(orient="index")

        # aggregate totals per group
        tot_this = exp_this.groupby(gcols_full)["exp_creates"].sum().rename("tot_creates").reset_index() if not exp_this.empty else pd.DataFrame(columns=gcols_full+["tot_creates"])
        tot_next = exp_next.groupby(gcols_full)["exp_creates"].sum().rename("tot_creates").reset_index() if not exp_next.empty else pd.DataFrame(columns=gcols_full+["tot_creates"])

        inflow_rows = []
        # This month: M0 from this month's remaining creates, spread uniformly
        for _, r in tot_this.iterrows():
            gkey = tuple(r[c] for c in gcols_full)
            rr = rates_key.get(gkey, {"r0":g_r0, "r1":g_r1})
            m0_total = float(r["tot_creates"]) * float(rr["r0"])
            per_day = m0_total / n_rem
            for d in days_remaining_this:
                inflow_rows.append((*gkey, d, per_day))
        # Next month: M1 from this month's creates + M0 from next month's creates
        for _, r in tot_this.iterrows():
            gkey = tuple(r[c] for c in gcols_full)
            rr = rates_key.get(gkey, {"r0":g_r0, "r1":g_r1})
            m1_total = float(r["tot_creates"]) * float(rr["r1"])
            per_day = m1_total / n_next
            for d in days_next_month:
                inflow_rows.append((*gkey, d, per_day))
        for _, r in tot_next.iterrows():
            gkey = tuple(r[c] for c in gcols_full)
            rr = rates_key.get(gkey, {"r0":g_r0, "r1":g_r1})
            m0n_total = float(r["tot_creates"]) * float(rr["r0"])
            per_day = m0n_total / n_next
            for d in days_next_month:
                inflow_rows.append((*gkey, d, per_day))

        inflow = pd.DataFrame(inflow_rows, columns=gcols_full+["day","p"])
        if inflow.empty:
            inflow = pd.DataFrame(columns=gcols_full+["day","p"])

    # Conform group labels
    for c in gcols_full:
        if c not in cart.columns: cart[c] = "NA"
        if c not in inflow.columns: inflow[c] = "NA"
        cart[c] = cart[c].astype(str)
        inflow[c] = inflow[c].astype(str)

    inflow["Day"] = inflow["day"].dt.date.astype(str)
    inflow["Day of Week"] = inflow["day"].dt.day_name()

    base_cols = ["day","p","Day","Day of Week"] + gcols_full
    cart_all = pd.concat([
        cart[base_cols].copy(),
        inflow[base_cols].copy()
    ], ignore_index=True)

    # ---------- Summaries & Breakdowns ----------
    keys = []
    rename_map = {}
    if "Student/Academic Counsellor" in group_by and COL_CSL: keys.append(COL_CSL); rename_map[COL_CSL]="Counsellor"
    if "JetLearn Deal Source" in group_by and COL_SOURCE: keys.append(COL_SOURCE); rename_map[COL_SOURCE]="Deal Source"
    if "Country" in group_by and COL_COUNTRY: keys.append(COL_COUNTRY); rename_map[COL_COUNTRY]="Country"
    if "Day" in group_by: keys.append("Day")
    if "Day of Week" in group_by: keys.append("Day of Week")

    def summarize(mask, label):
        sub = cart_all.loc[mask]
        if not keys:
            return pd.DataFrame({label:[int(round(sub['p'].sum()))]})
        g = sub.groupby(keys, dropna=False)["p"].sum().reset_index()
        g.rename(columns=rename_map, inplace=True)
        g[label] = g["p"].round(0).astype(int)
        return g.drop(columns=["p"])

    today_mask = cart_all["day"] == today
    tom_mask   = cart_all["day"] == (today + pd.Timedelta(days=1))
    month_mask = (cart_all["day"] >= today) & (cart_all["day"] <= end_this)
    next_mask  = (cart_all["day"] > end_this) & (cart_all["day"] <= end_next)

    g_today = summarize(today_mask, "Today")
    g_tom   = summarize(tom_mask, "Tomorrow")
    g_this  = summarize(month_mask, "This Month")
    g_next  = summarize(next_mask, "Next Month")

    def smart_merge(a,b):
        common = [c for c in a.columns if c in b.columns and c not in ["Today","Tomorrow","This Month","Next Month"]]
        return a.merge(b, on=common, how="outer")

    out = g_today
    for part in [g_tom, g_this, g_next]:
        out = smart_merge(out, part)
    for c in ["Today","Tomorrow","This Month","Next Month"]:
        if c in out.columns: out[c] = out[c].fillna(0).astype(int)

    # ---------- Display ----------
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Today", f"{int(out['Today'].sum()) if 'Today' in out.columns else 0:,}")
    k2.metric("Tomorrow", f"{int(out['Tomorrow'].sum()) if 'Tomorrow' in out.columns else 0:,}")
    k3.metric("This Month", f"{int(out['This Month'].sum()) if 'This Month' in out.columns else 0:,}")
    k4.metric("Next Month", f"{int(out['Next Month'].sum()) if 'Next Month' in out.columns else 0:,}")

    st.markdown("#### Forecast breakdown")
    order_cols = [c for c in ["Today","Tomorrow","This Month","Next Month"] if c in out.columns]
    st.dataframe(out.sort_values(by=[c for c in ["Today","This Month","Next Month"] if c in out.columns], ascending=False),
                 use_container_width=True)
    st.download_button("Download — ML Predictability CSV",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="ml_predictability_payment_count_inflow.csv",
                       mime="text/csv")

    # Optional day-wise bars if 'Day' included
    if "Day" in group_by:
        try:
            long = out.melt(id_vars=[c for c in out.columns if c not in ["Today","Tomorrow","This Month","Next Month"]],
                            value_vars=order_cols, var_name="Bucket", value_name="Count")
            st.altair_chart(
                alt.Chart(long).mark_bar().encode(
                    x=alt.X("Day:N", sort=None, title=None),
                    y=alt.Y("Count:Q", title=None),
                    color="Bucket:N",
                    tooltip=list(long.columns)
                ).properties(height=260),
                use_container_width=True
            )
        except Exception:
            pass

    st.caption("Counts = predicted payments from **existing open deals** + expected payments from **new deals** "
               "(last-month DOW create rates with smoothed M0/M1 cohort rates).")
