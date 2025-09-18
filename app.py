# app.py — JetLearn Insights (MTD/Cohort) + Predictability of Enrollment
# Works on Python 3.11–3.13+. No top-level sklearn imports (safe fallback).
# Streamlit APIs use width='stretch' (no use_container_width deprecation).

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
.warn { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; padding:6px 10px; border-radius:10px; display:inline-block; }
.ok { background:#ecfeff; border:1px solid #a5f3fc; color:#155e75; padding:6px 10px; border-radius:10px; display:inline-block; }
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

def month_start(d: date) -> pd.Timestamp:
    return pd.Timestamp(d).to_period("M").to_timestamp()

def month_add(ts: pd.Timestamp, k: int) -> pd.Timestamp:
    return (ts.to_period("M") + k).to_timestamp()

# --------------------- Header ---------------------
with st.container():
    st.markdown('<div class="head"><div class="title">📊 JetLearn — Insights & Predictability</div></div>', unsafe_allow_html=True)

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

# --------------------- Tabs ---------------------
tab_insights, tab_pred = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictability of Enrollment"])

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
            _ = c1.checkbox("All", value=True, key=all_key)
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

    def scenario_controls(name: str, df_, date_like_cols_):
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
            pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=True, src_sel=src_sel,
            cty_all=True, cty_sel=cty_sel, csl_all=True, csl_sel=csl_sel
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
  <div class="delta'>{row['Window']}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — A</div>", unsafe_allow_html=True)
        if not tablesA: st.info("No tables — enable splits/leaderboards.")
        else:
            for name,frame in tablesA.items():
                st.subheader(name)
                st.dataframe(frame, width='stretch')
                st.download_button("Download CSV — "+name, to_csv_bytes(frame),
                                   file_name=f"A_{name.replace(' ','_')}.csv", mime="text/csv")
        st.markdown("<div class='section-title'>📈 Trends — A</div>", unsafe_allow_html=True)
        if "MTD Trend" in chartsA: st.altair_chart(chartsA["MTD Trend"], width='stretch')
        if "Cohort Trend" in chartsA: st.altair_chart(chartsA["Cohort Trend"], width='stretch')
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
  <div class="label'>{row['Scope']} — {row['Metric']}</div>
  <div class="value'>{int(row['Value']):,}</div>
  <div class="delta'>{row['Window']}</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — B</div>", unsafe_allow_html=True)
            if not tablesB: st.info("No tables — enable splits/leaderboards.")
            else:
                for name,frame in tablesB.items():
                    st.subheader(name)
                    st.dataframe(frame, width='stretch')
                    st.download_button("Download CSV — "+name, to_csv_bytes(frame),
                                       file_name=f"B_{name.replace(' ','_')}.csv", mime="text/csv")
            st.markdown("<div class='section-title'>📈 Trends — B</div>", unsafe_allow_html=True)
            if "MTD Trend" in chartsB: st.altair_chart(chartsB["MTD Trend"], width='stretch')
            if "Cohort Trend" in chartsB: st.altair_chart(chartsB["Cohort Trend"], width='stretch')
            st.caption("**Scenario B** — " + caption_from(metaB))

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.caption("Excluded globally: 1.2 Invalid Deal")

# =========================================================
# =========  🔮 PREDICTABILITY OF ENROLLMENT TAB ==========
# =========================================================
with tab_pred:
    st.markdown("#### Predict enrollments **Today / Tomorrow / This Month / Next Month**")

    # Optional ML deps (guarded import so app never crashes)
    SKLEARN_OK = True
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from sklearn.ensemble import GradientBoostingClassifier
    except Exception:
        SKLEARN_OK = False

    # Required / recommended columns
    PE_REQUIRED_BASE = ["Create Date","Country","JetLearn Deal Source"]
    PE_REQUIRED_EXTRA = [
        "Payment Received Date",
        "Number of Sales Activity",
        "Number of Call Connected",
        "Last Date of Sales Activity",
        "Last Date of Call Connected",
    ]
    missing_cols = [c for c in PE_REQUIRED_BASE if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for predictability tab: {missing_cols}")
        st.stop()

    # Parse key dates for this tab
    for c in ["Payment Received Date", "Last Date of Sales Activity", "Last Date of Call Connected"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Controls
    as_of = st.date_input("As-of date", value=pd.Timestamp.today().date(),
                          help="Training excludes the entire as-of month to avoid leakage.")
    run_bt = st.checkbox("Backtest on last full month (show accuracy)", value=False)
    run_btn = st.button("Train / Estimate", type="primary")

    st.markdown(
        ("<span class='ok'>Using scikit-learn ML models</span>"
         if SKLEARN_OK else
         "<span class='warn'>scikit-learn not found — using statistics-based fallback (EB cohort)</span>"),
        unsafe_allow_html=True
    )

    # --------- helpers ----------
    def pe_to_ts_month(d: pd.Series) -> pd.Series:
        return pd.to_datetime(d, errors="coerce", dayfirst=True).dt.to_period("M").to_timestamp()

    def pe_month_start(d: date) -> pd.Timestamp:
        return pd.Timestamp(d).to_period("M").to_timestamp()

    def pe_month_add(ts: pd.Timestamp, k: int) -> pd.Timestamp:
        return (ts.to_period("M") + k).to_timestamp()

    def build_feature_frame(raw: pd.DataFrame, as_of_: date) -> pd.DataFrame:
        X = raw.copy()
        asof_ts = pd.Timestamp(as_of_)
        for c in ["Create Date","Payment Received Date","Last Date of Sales Activity","Last Date of Call Connected"]:
            if c in X.columns:
                X[c] = pd.to_datetime(X[c], errors="coerce", dayfirst=True)
        X["Create_Month"] = pe_to_ts_month(X["Create Date"])
        # Age features
        if "Age" in X.columns:
            with np.errstate(all="ignore"):
                X["Age_num"] = pd.to_numeric(X["Age"], errors="coerce")
        else:
            X["Age_num"] = np.nan
        X["age_in_days"] = (asof_ts - X["Create Date"]).dt.days.clip(lower=0)
        # Recency
        X["days_since_last_activity"] = (asof_ts - X["Last Date of Sales Activity"]).dt.days
        X["days_since_last_call"]     = (asof_ts - X["Last Date of Call Connected"]).dt.days
        X["days_since_last_activity"] = X["days_since_last_activity"].fillna(999).clip(lower=0, upper=9999)
        X["days_since_last_call"]     = X["days_since_last_call"].fillna(999).clip(lower=0, upper=9999)
        # Calendar
        X["create_dow"] = X["Create Date"].dt.dayofweek
        X["create_dom"] = X["Create Date"].dt.day
        X["create_moy"] = X["Create Date"].dt.month
        # Numeric counts
        for c in ["Number of Sales Activity","Number of Call Connected"]:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).clip(lower=0)
        # Targets & masks
        pay = X["Payment Received Date"] if "Payment Received Date" in X.columns else pd.Series(pd.NaT, index=X.index)
        tomorrow = asof_ts + pd.Timedelta(days=1)
        this_m0  = pe_month_start(as_of_)
        next_m0  = pe_month_add(this_m0, 1)
        next_m1  = pe_month_add(this_m0, 2) - pd.Timedelta(seconds=1)
        X["y_today"]     = (pay.dt.date == asof_ts.date()).astype(int)
        X["y_tomorrow"]  = (pay.dt.date == tomorrow.date()).astype(int)
        X["y_thismonth"] = ((pay >= asof_ts.floor("D")) & (pay < next_m0)).astype(int)
        X["y_nextmonth"] = ((pay >= next_m0) & (pay < next_m1)).astype(int)
        X["open_as_of"] = pay.isna() | (pay > asof_ts)
        X["created_by_asof"] = X["Create Date"] <= asof_ts
        X["score_mask"] = X["open_as_of"] & X["created_by_asof"]
        return X

    # --------- ML path ----------
    def train_predict_ML(X: pd.DataFrame, as_of_: date):
        cutoff = pe_month_start(as_of_)
        train_mask = X["Create_Month"] < cutoff
        feats_num = ["age_in_days","Age_num","days_since_last_activity","days_since_last_call",
                     "Number of Sales Activity","Number of Call Connected","create_dow","create_dom","create_moy"]
        feats_num = [c for c in feats_num if c in X.columns]
        feats_cat = [c for c in ["Country","JetLearn Deal Source"] if c in X.columns]
        feats = feats_num + feats_cat

        def pipe():
            transf=[]
            if feats_num: transf.append(("num","passthrough",feats_num))
            if feats_cat: transf.append(("cat",OneHotEncoder(handle_unknown="ignore"),feats_cat))
            pre=ColumnTransformer(transf)
            clf=GradientBoostingClassifier(random_state=42)
            return Pipeline([("pre",pre),("clf",clf)])

        ys = {
            "today": X.loc[train_mask, "y_today"],
            "tomorrow": X.loc[train_mask, "y_tomorrow"],
            "thismonth": X.loc[train_mask, "y_thismonth"],
            "nextmonth": X.loc[train_mask, "y_nextmonth"],
        }
        models={}
        for k in ys:
            y=ys[k]
            if len(y.dropna().unique())<2 or y.fillna(0).sum()==0:
                models[k]=None
                continue
            model=pipe()
            model.fit(X.loc[train_mask, feats], y)
            models[k]=model

        score_idx = X.index[X["score_mask"]]
        preds={}
        if len(score_idx)==0:
            preds = {k:0.0 for k in ["today","tomorrow","thismonth","nextmonth"]}
        else:
            Xs = X.loc[score_idx, feats]
            for k,m in models.items():
                preds[k]= float(np.sum(m.predict_proba(Xs)[:,1])) if m is not None else 0.0

        out = pd.DataFrame({"Deal Index": score_idx})
        for k,m in models.items():
            if m is None or len(score_idx)==0:
                out[f"p_{k}"]=0.0
            else:
                out[f"p_{k}"]=m.predict_proba(X.loc[score_idx, feats])[:,1]
        if "Country" in X.columns: out["Country"]=X.loc[score_idx,"Country"].astype(str)
        if "JetLearn Deal Source" in X.columns: out["JetLearn Deal Source"]=X.loc[score_idx,"JetLearn Deal Source"].astype(str)

        pay = X["Payment Received Date"]
        asof_ts = pd.Timestamp(as_of_)
        tomorrow = asof_ts + pd.Timedelta(days=1)
        this_m0  = pe_month_start(as_of_)
        next_m0  = pe_month_add(this_m0, 1)
        next_m1  = pe_month_add(this_m0, 2) - pd.Timedelta(seconds=1)
        actuals = {
            "today":     int((pay.dt.date == asof_ts.date()).sum()),
            "tomorrow":  int((pay.dt.date == tomorrow.date()).sum()),
            "thismonth": int(((pay >= asof_ts.floor("D")) & (pay < next_m0)).sum()),
            "nextmonth": int(((pay >= next_m0) & (pay < next_m1)).sum()),
        }
        return preds, out, actuals

    # --------- EB fallback (no sklearn) ----------
    def train_predict_EB(X: pd.DataFrame, as_of_: date):
        asof_ts = pd.Timestamp(as_of_)
        this_m0 = pe_month_start(as_of_)
        prev_m0 = month_add(this_m0, -1)
        next_m0 = month_add(this_m0, 1)
        next_m1 = month_add(this_m0, 2)

        mask_train = X["Create_Month"] < this_m0
        T = X.loc[mask_train].copy()

        grp = ["JetLearn Deal Source","Country"]
        for g in grp:
            if g not in T.columns:
                T[g]=""

        T["PaymentMonth"] = X.loc[T.index,"Payment Received Date"].dt.to_period("M").dt.to_timestamp()
        T["Lag"] = (T["PaymentMonth"].dt.year*12 + T["PaymentMonth"].dt.month) - (T["Create_Month"].dt.year*12 + T["Create_Month"].dt.month)
        r0 = T[T["Lag"]==0].groupby(grp)["Lag"].count().rename("succ0")
        r1 = T[T["Lag"]==1].groupby(grp)["Lag"].count().rename("succ1")
        trials = T.groupby(grp)["Create Date"].count().rename("trials")
        base = pd.concat([r0,r1,trials], axis=1).fillna(0.0)

        g_trials = float(trials.sum())
        g_s0 = float(r0.sum()); g_s1 = float(r1.sum())
        g_r0 = (g_s0 / g_trials) if g_trials else 0.0
        g_r1 = (g_s1 / g_trials) if g_trials else 0.0
        prior_strength = 25.0
        base["r0"] = (base["succ0"] + prior_strength*g_r0) / (base["trials"] + prior_strength)
        base["r1"] = (base["succ1"] + prior_strength*g_r1) / (base["trials"] + prior_strength)

        score_mask = (X["Payment Received Date"].isna() | (X["Payment Received Date"] > asof_ts)) & (X["Create Date"] <= asof_ts)
        score = X.loc[score_mask].copy()
        score["CM"] = pe_to_ts_month(score["Create Date"])

        cur_cre = score[score["CM"]==this_m0].groupby(grp)["Create Date"].count().rename("C_cur")
        prv_cre = score[score["CM"]==prev_m0].groupby(grp)["Create Date"].count().rename("C_prev")
        both = pd.concat([base[["r0","r1"]], cur_cre, prv_cre], axis=1).fillna(0.0)
        both["Forecast_thismonth"] = both["r0"]*both["C_cur"] + both["r1"]*both["C_prev"]

        # Day-of-month profile for today/tomorrow
        paid_hist = X.loc[X["Payment Received Date"].notna()].copy()
        paid_hist["d"] = paid_hist["Payment Received Date"].dt.day
        paid_hist["moy"] = paid_hist["Payment Received Date"].dt.month
        moy = month_start(as_of_).month
        pool = paid_hist[paid_hist["moy"]==moy]
        if pool.empty: pool = paid_hist
        days_in_m = int((next_m0 - this_m0).days)
        idx = pd.Index(range(1, days_in_m+1), name="day")
        cnt = pool["d"].value_counts().reindex(idx, fill_value=0).astype(float)
        prior = pd.Series(1.0/days_in_m, index=idx)
        day_prof = (cnt + 5.0*prior) / (cnt.sum() + 5.0)
        day_prof = day_prof / day_prof.sum()

        dom_today = asof_ts.day
        dom_tom = (asof_ts + pd.Timedelta(days=1)).day if asof_ts < (next_m0 - pd.Timedelta(days=1)) else dom_today
        p_today = float(day_prof.reindex(idx).get(dom_today, 0.0))
        p_tom = float(day_prof.reindex(idx).get(dom_tom, 0.0))

        pred_this = float(both["Forecast_thismonth"].sum())
        pred_today = pred_this * p_today
        pred_tom   = pred_this * p_tom
        next_forecast = float((both["r1"] * both["C_cur"]).sum())

        out = score[[*grp]].copy().reset_index(drop=True)
        join = both.reset_index()[grp+["r0","r1"]].rename(columns={"r0":"p_thismonth_proxy","r1":"p_nextmonth_proxy"})
        out = out.merge(join, on=grp, how="left").fillna(0.0)
        out["p_today"]=p_today
        out["p_tomorrow"]=p_tom
        out = out.rename(columns={"p_thismonth_proxy":"p_thismonth","p_nextmonth_proxy":"p_nextmonth"})

        pay = X["Payment Received Date"]
        actuals = {
            "today":     int((pay.dt.date == asof_ts.date()).sum()),
            "tomorrow":  int((pay.dt.date == (asof_ts + pd.Timedelta(days=1)).date()).sum()),
            "thismonth": int(((pay >= asof_ts.floor("D")) & (pay < next_m0)).sum()),
            "nextmonth": int(((pay >= next_m0) & (pay < next_m1)).sum()),
        }
        preds = {"today": pred_today, "tomorrow": pred_tom, "thismonth": pred_this, "nextmonth": next_forecast}
        return preds, out, actuals

    # Backtest util (daily error on last full month)
    def simple_backtest(X_all: pd.DataFrame, estimator_fn, as_of_: date):
        asof_m0 = pe_month_start(as_of_)
        test_m0 = month_add(asof_m0, -1)
        test_m1 = month_add(test_m0, 1) - pd.Timedelta(seconds=1)
        days = pd.date_range(test_m0, test_m1, freq="D")
        rows=[]
        for d in days:
            Xd = build_feature_frame(df, d.date())
            preds, *_ = estimator_fn(Xd, d.date())
            rows.append({"date": d.date(), "pred_thismonth": preds["thismonth"]})
        dd = pd.DataFrame(rows)
        pay = X_all["Payment Received Date"]
        dd["actual_thismonth"] = [
            int(((pay >= pd.Timestamp(d)) & (pay < month_add(test_m0, 1))).sum())
            for d in pd.to_datetime(dd["date"])
        ]
        dd["ae"] = (dd["pred_thismonth"] - dd["actual_thismonth"]).abs()
        mae = float(dd["ae"].mean())
        mape = float((dd["ae"] / dd["actual_thismonth"].replace(0, np.nan)).dropna().mean()) if (dd["actual_thismonth"]>0).any() else np.nan
        return dd, mae, mape, test_m1.date()

    if run_btn:
        with st.spinner("Building features and generating predictions…"):
            X = build_feature_frame(df, as_of)
            if SKLEARN_OK:
                try:
                    preds, deal_out, actuals = train_predict_ML(X, as_of)
                except Exception as e:
                    st.warning(f"ML path error ({e}). Falling back to EB…")
                    preds, deal_out, actuals = train_predict_EB(X, as_of)
            else:
                preds, deal_out, actuals = train_predict_EB(X, as_of)

            k1,k2,k3,k4 = st.columns(4)
            k1.metric(f"Predicted Today ({pd.Timestamp(as_of):%d %b})", f"{int(round(preds['today'])):,}")
            k2.metric(f"Predicted Tomorrow ({(pd.Timestamp(as_of)+pd.Timedelta(days=1)):%d %b})", f"{int(round(preds['tomorrow'])):,}")
            k3.metric(f"Predicted This Month ({month_start(as_of):%b %Y})", f"{int(round(preds['thismonth'])):,}")
            k4.metric(f"Predicted Next Month ({month_add(month_start(as_of),1):%b %Y})", f"{int(round(preds['nextmonth'])):,}")

            st.markdown("**Open deals (with predicted probabilities / proxies)**")
            if not deal_out.empty:
                st.dataframe(deal_out, width='stretch')
                st.download_button("Download Predictions (CSV)", deal_out.to_csv(index=False).encode("utf-8"),
                                   file_name="predictability_open_deals.csv", mime="text/csv")
            else:
                st.info("No open deals to score for the selected date.")

            if run_bt:
                with st.spinner("Running backtest on last full month…"):
                    estimator = (lambda Xd, d: train_predict_ML(Xd, d)[0:3]) if SKLEARN_OK else (lambda Xd, d: train_predict_EB(Xd, d))
                    daily_df, mae, mape, asof_bt = simple_backtest(df, estimator, as_of)
                    st.markdown(f"##### Backtest (last full month) — simulated as-of **{pd.Timestamp(asof_bt):%d %b %Y}**")
                    m1, m2 = st.columns(2)
                    m1.metric("MAE (daily, this month)", f"{mae:.2f}")
                    m2.metric("MAPE (daily, this month)", f"{(mape*100):.1f}%" if pd.notna(mape) else "—")
                    if not daily_df.empty:
                        long = daily_df.melt(id_vars="date", value_vars=["pred_thismonth","actual_thismonth"],
                                             var_name="Series", value_name="Count")
                        ch = alt.Chart(long).mark_line(point=True).encode(
                            x=alt.X("date:T", title=None),
                            y=alt.Y("Count:Q", title=None),
                            color=alt.Color("Series:N", scale=alt.Scale(range=PALETTE)),
                            tooltip=["date:T","Series:N","Count:Q"]
                        ).properties(height=280)
                        st.altair_chart(ch, width='stretch')

    st.caption("Notes: Current month is excluded from training. If scikit-learn is unavailable, "
               "the tab uses an empirical-Bayes cohort estimator with day-of-month allocation for Today/Tomorrow.")
