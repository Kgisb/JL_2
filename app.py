# app.py — JetLearn Insights (fast) + Predictivity of Enrollment (independent, fast)
# - No auto CSV read at startup (instant boot)
# - Caching for load/preprocess/compute/ML
# - Fast mode sampling for ML
# - use_container_width=True (no width='stretch')

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
.head { position:sticky; top:0; z-index:50; display:flex; gap:10px; align-items:center;
  padding:10px 12px; background:#0b1220; color:#fff; border-radius:12px; margin-bottom:10px; }
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

# --------------------- Fast, cached I/O ---------------------
@st.cache_data(show_spinner=False)
def robust_read_csv_fast(obj):
    # Try utf-8 first; fallback to your robust strategy
    try:
        return pd.read_csv(obj, low_memory=False)
    except Exception:
        for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
            try:
                return pd.read_csv(obj, encoding=enc, low_memory=False)
            except Exception:
                pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

@st.cache_data(show_spinner=False)
def preprocess_df(df_raw: pd.DataFrame, exclude_invalid: bool):
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if exclude_invalid and "Deal Stage" in df.columns:
        df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

    # Parse only relevant dates
    for c in ["Create Date","Payment Received Date","Last Date of Sales Activity","Last Date of Call Connected"]:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    if "Create Date" in df.columns:
        df["Create_Month"] = df["Create Date"].dt.to_period("M")
    return df

# --------------------- Helpers ---------------------
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

def to_csv_bytes(df: pd.DataFrame)->bytes: return df.to_csv(index=False).encode("utf-8")

def group_label_from_series(s: pd.Series, grain: str):
    if grain=="Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain=="Week":
        iso=pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str)+"-W"+iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

def detect_measure_date_columns_fast(df: pd.DataFrame):
    # Prefer common names; don’t scan the whole frame
    preferred = [c for c in [
        "Payment Received Date",
        "Enrollment Date", "First Demo Date", "Second Demo Date"
    ] if c in df.columns]
    date_like = []
    cols = preferred if preferred else [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    for col in cols:
        if col == "Create Date": 
            continue
        if not np.issubdtype(df[col].dtype, np.datetime64):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum()>0:
                df[col] = parsed
                date_like.append(col)
        else:
            date_like.append(col)
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c!="Payment Received Date"]
    # Cap list length to avoid very wide UIs
    return list(dict.fromkeys(date_like))[:6]

def month_start(d: date) -> pd.Timestamp:
    return pd.Timestamp(d).to_period("M").to_timestamp()

def month_add(ts: pd.Timestamp, k: int) -> pd.Timestamp:
    return (ts.to_period("M") + k).to_timestamp()

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
        path = st.text_input("…or CSV path", value="", placeholder="Enter a CSV file path (optional)")
    with c3:
        exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal'", value=True)

# Load lazily
if uploaded is not None:
    df_raw = robust_read_csv_fast(BytesIO(uploaded.getvalue()))
elif path.strip():
    df_raw = robust_read_csv_fast(path.strip())
else:
    st.info("Upload a CSV or enter a path to begin.")
    st.stop()

st.success("Data loaded ✅")

df = preprocess_df(df_raw, exclude_invalid=exclude_invalid)

# Validate required columns for Insights
missing=[c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")
    st.stop()

# --------------------- TABS ---------------------
tab_insights, tab_predict = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictivity of Enrollment"])

# =========================================================
# ===============  INSIGHTS (MTD / Cohort)  ===============
# =========================================================
with tab_insights:
    date_like_cols = detect_measure_date_columns_fast(df.copy())
    if not date_like_cols:
        st.error("No usable date-like columns (other than Create Date) found. Add a column like 'Payment Received Date'.")
        st.stop()

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

    @st.cache_data(show_spinner=False)
    def compute_outputs_cached(base, measures, mode,
                               mtd_from, mtd_to, mtd_grain,
                               coh_from, coh_to, coh_grain,
                               split_dims, top_ctry, top_src, top_csl, pair):
        metrics_rows, tables, charts = [], {}, {}
        # MTD
        if mode in ("MTD","Both") and mtd_from and mtd_to and measures:
            in_cre = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
            sub = base[in_cre].copy()
            flags=[]
            for m in measures:
                if m not in sub.columns: continue
                mn = f"{m}_Month"
                if mn not in sub.columns: sub[mn] = sub[m].dt.to_period("M")
                flg=f"__MTD__{m}"
                sub[flg] = ((sub[m].notna()) & (sub[mn]==sub["Create_Month"]).astype(bool)).astype(int)
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
            split_dims=split_dims, top_ctry=top_ctry, top_src=top_src, top_csl=top_csl, pair=pair
        )

    show_b = st.toggle("Enable Scenario B (compare)", value=False)
    left_col, right_col = st.columns(2) if show_b else (st.container(), None)
    with (left_col if show_b else st.container()):
        metaA = scenario_controls("A", df, date_like_cols)
    if show_b:
        with right_col:
            metaB = scenario_controls("B", df, date_like_cols)

    with st.spinner("Calculating…"):
        metricsA, tablesA, chartsA = compute_outputs_cached(**metaA)
        if show_b:
            metricsB, tablesB, chartsB = compute_outputs_cached(**metaB)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    if show_b:
        tA, tB, _ = st.tabs(["📋 Scenario A", "📋 Scenario B", "🧠 Compare"])
    else:
        tA, = st.tabs(["📋 Scenario A"])

    def render_block(metrics, tables, charts, cap):
        st.markdown("<div class='section-title'>📌 KPI Overview</div>", unsafe_allow_html=True)
        dfK=pd.DataFrame(metrics)
        if dfK.empty: st.info("No KPIs yet — adjust filters.")
        else:
            cols=st.columns(4)
            for i,row in dfK.iterrows():
                with cols[i%4]:
                    st.markdown(f"""
<div class="kpi">
  <div class="label">{row['Scope']} — {row['Metric']}</div>
  <div class="value">{int(row['Value']):,}</div>
  <div class="delta">{row['Window']}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧩 Splits & Leaderboards</div>", unsafe_allow_html=True)
        if not tables: st.info("No tables — enable splits/leaderboards.")
        else:
            for name,frame in tables.items():
                st.subheader(name)
                st.dataframe(frame, use_container_width=True)
                st.download_button("Download CSV — "+name, to_csv_bytes(frame),
                                   file_name=f"{name.replace(' ','_')}.csv", mime="text/csv")
        st.markdown("<div class='section-title'>📈 Trends</div>", unsafe_allow_html=True)
        if "MTD Trend" in charts: st.altair_chart(charts["MTD Trend"], use_container_width=True)
        if "Cohort Trend" in charts: st.altair_chart(charts["Cohort Trend"], use_container_width=True)
        st.caption(cap)

    with tA:
        render_block(metricsA, tablesA, chartsA, "Scenario A")
    if show_b:
        with tB:
            render_block(metricsB, tablesB, chartsB, "Scenario B")

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.caption("Excluded globally: 1.2 Invalid Deal")

# =========================================================
# =========  🔮 PREDICTIVITY OF ENROLLMENT (INDEPENDENT) ==
# =========================================================
with tab_predict:
    st.markdown("### Predictivity of Enrollment — This Month & Next Month")
    st.caption("Trains on past data (excluding the current month). Independent of Insights settings.")

    # Optional sklearn imports (guarded)
    _SK_OK = True
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception:
        _SK_OK = False

    # Column checks (minimal)
    for need in ["Create Date","Country","JetLearn Deal Source"]:
        if need not in df.columns:
            st.error(f"Missing column required for Predictivity: {need}")
            st.stop()

    # Local copy
    _df = df.copy()

    def _mstart(d):  return pd.Timestamp(d).to_period("M").to_timestamp()
    def _madd(ts,k): return (ts.to_period("M")+k).to_timestamp()
    def _to_month(s): return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.to_period("M").to_timestamp()

    @st.cache_data(show_spinner=False)
    def build_features_cached(df_in: pd.DataFrame, as_of_iso: str):
        as_of = pd.to_datetime(as_of_iso).date()
        X = df_in.copy()
        for c in ["Create Date","Payment Received Date","Last Date of Sales Activity","Last Date of Call Connected"]:
            if c in X.columns and not np.issubdtype(X[c].dtype, np.datetime64):
                X[c] = pd.to_datetime(X[c], errors="coerce", dayfirst=True)
        X["Create_Month"] = _to_month(X["Create Date"])

        for c in ["Number of Sales Activity","Number of Call Connected"]:
            if c in X.columns: X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).clip(lower=0)
            else: X[c] = 0
        X["Age_num"] = pd.to_numeric(X["Age"], errors="coerce") if "Age" in X.columns else np.nan

        asof_ts = pd.Timestamp(as_of)
        X["age_in_days"] = (asof_ts - X["Create Date"]).dt.days.clip(lower=0)
        X["days_since_last_activity"] = (asof_ts - X.get("Last Date of Sales Activity")).dt.days if "Last Date of Sales Activity" in X.columns else np.nan
        X["days_since_last_call"]     = (asof_ts - X.get("Last Date of Call Connected")).dt.days if "Last Date of Call Connected" in X.columns else np.nan
        X["days_since_last_activity"] = pd.to_numeric(X["days_since_last_activity"], errors="coerce").fillna(999).clip(lower=0, upper=9999)
        X["days_since_last_call"]     = pd.to_numeric(X["days_since_last_call"], errors="coerce").fillna(999).clip(lower=0, upper=9999)

        X["create_dow"] = X["Create Date"].dt.dayofweek
        X["create_dom"] = X["Create Date"].dt.day
        X["create_moy"] = X["Create Date"].dt.month

        pay = X.get("Payment Received Date")
        this_m0 = _mstart(as_of)
        next_m0 = _madd(this_m0, 1)
        X["y_thismonth"] = ((pay >= this_m0) & (pay < next_m0)).astype(int) if pay is not None else 0
        X["y_nextmonth"] = ((pay >= next_m0) & (pay < _madd(this_m0, 2))).astype(int) if pay is not None else 0

        X["open_as_of"] = (pay.isna() | (pay > pd.Timestamp(as_of))) if pay is not None else True
        X["created_by_asof"] = X["Create Date"] <= pd.Timestamp(as_of)
        X["score_mask"] = X["open_as_of"] & X["created_by_asof"]
        return X

    @st.cache_data(show_spinner=False)
    def predict_cached(X: pd.DataFrame, as_of_iso: str, use_ml: bool, sample_n: int|None):
        as_of = pd.to_datetime(as_of_iso).date()
        # Exclude current month from training
        cutoff = _mstart(as_of)
        train_mask = X["Create_Month"] < cutoff
        feats_num = ["age_in_days","Age_num","days_since_last_activity","days_since_last_call",
                     "Number of Sales Activity","Number of Call Connected","create_dow","create_dom","create_moy"]
        feats_num = [c for c in feats_num if c in X.columns]
        feats_cat = [c for c in ["Country","JetLearn Deal Source"] if c in X.columns]
        use_cols = feats_num + feats_cat

        # Optional sampling for speed
        train_idx = X.index[train_mask]
        if sample_n and len(train_idx) > sample_n:
            train_idx = np.random.RandomState(42).choice(train_idx, size=sample_n, replace=False)

        def _eb_fallback():
            # Simple cohort r0/r1 by (Source, Country)
            this_m0 = _mstart(as_of); prev_m0 = _madd(this_m0,-1); next_m0 = _madd(this_m0,1)
            H = X[X["Create_Month"] < this_m0].copy()
            grp = ["JetLearn Deal Source","Country"]
            for g in grp:
                if g not in H.columns: H[g] = ""
            H["PaymentMonth"] = X.loc[H.index, "Payment Received Date"].dt.to_period("M").to_timestamp() if "Payment Received Date" in X.columns else pd.NaT
            codeC = H["Create_Month"].dt.year*12 + H["Create_Month"].dt.month
            codeP = H["PaymentMonth"].dt.year*12 + H["PaymentMonth"].dt.month
            H["Lag"] = codeP - codeC

            r0 = H[H["Lag"]==0].groupby(grp)["Lag"].count().rename("succ0")
            r1 = H[H["Lag"]==1].groupby(grp)["Lag"].count().rename("succ1")
            trials = H.groupby(grp)["Create Date"].count().rename("trials")
            base = pd.concat([r0,r1,trials], axis=1).fillna(0.0)

            g_trials = trials.sum()
            g_r0 = (r0.sum()/g_trials) if g_trials else 0.0
            g_r1 = (r1.sum()/g_trials) if g_trials else 0.0
            prior_strength = 25.0
            base["r0"] = (base["succ0"] + prior_strength*g_r0) / (base["trials"] + prior_strength)
            base["r1"] = (base["succ1"] + prior_strength*g_r1) / (base["trials"] + prior_strength)

            score_mask = (X.get("Payment Received Date").isna() | (X.get("Payment Received Date") > pd.Timestamp(as_of))) if "Payment Received Date" in X.columns else True
            score_mask &= (X["Create Date"] <= pd.Timestamp(as_of))
            S = X.loc[score_mask].copy(); S["CM"] = _to_month(S["Create Date"])

            cur_cre = S[S["CM"] == this_m0].groupby(grp)["Create Date"].count().rename("C_cur")
            prev_cre = S[S["CM"] == prev_m0].groupby(grp)["Create Date"].count().rename("C_prev")
            both = pd.concat([base[["r0","r1"]], cur_cre, prev_cre], axis=1).fillna(0.0)

            this_pred = float((both["r0"]*both["C_cur"] + both["r1"]*both["C_prev"]).sum())
            next_pred = float((both["r1"]*both["C_cur"]).sum())
            return {"thismonth": this_pred, "nextmonth": next_pred}, None

            # (Per-deal table omitted in EB for speed)

        if not use_ml:
            return _eb_fallback()

        try:
            from sklearn.metrics import roc_auc_score
            pre = []
            if feats_num: pre.append(("num","passthrough",feats_num))
            if feats_cat: pre.append(("cat",OneHotEncoder(handle_unknown="ignore"),feats_cat))
            pre = ColumnTransformer(pre) if pre else "passthrough"
            clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=200, random_state=42)
            pipe = Pipeline([("pre", pre), ("clf", clf)])

            totals = {}
            models={}
            for key, ycol in {"thismonth":"y_thismonth","nextmonth":"y_nextmonth"}.items():
                y = X.loc[train_idx, ycol]
                if (y.sum() == 0) or (len(np.unique(y)) < 2):
                    models[key]=None
                    totals[key]=0.0
                    continue
                pipe_k = Pipeline([("pre", pre), ("clf", clf)])
                pipe_k.fit(X.loc[train_idx, use_cols], y)
                models[key]=pipe_k

            idx = X.index[X["score_mask"]]
            totals["thismonth"]=0.0; totals["nextmonth"]=0.0
            if len(idx)>0:
                Xs = X.loc[idx, use_cols]
                for key, m in models.items():
                    if m is None:
                        continue
                    probs = m.predict_proba(Xs)[:,1]
                    totals[key] = float(np.sum(probs))
            return totals, None
        except Exception:
            return _eb_fallback()

    as_of = st.date_input("As-of date", value=pd.Timestamp.today().date(),
                          help="Training excludes the current month to avoid leakage. Prediction for THIS & NEXT month.")
    fast_mode = st.toggle("Fast mode (sample training data for speed)", value=True,
                          help="On: sample up to N rows from training set for quicker ML.")
    sample_n = st.number_input("Fast mode sample size (rows)", min_value=1000, max_value=200000, value=50000, step=5000, disabled=not fast_mode)

    st.caption(("Using **ML (HistGradientBoosting)**" if _SK_OK else "scikit-learn not found — using **cohort fallback**")
               + ". Independent of the Insights tab.")

    if st.button("Run Predictivity", type="primary"):
        with st.spinner("Preparing features…"):
            X = build_features_cached(_df, str(as_of))
        with st.spinner("Training / Predicting…"):
            totals, _ = predict_cached(X, str(as_of), use_ml=_SK_OK, sample_n=int(sample_n) if fast_mode else None)

        this_m0 = month_start(as_of)
        next_m0 = month_add(this_m0, 1)
        k1, k2 = st.columns(2)
        k1.metric(f"Predicted This Month ({this_m0:%b %Y})", f"{int(round(totals.get('thismonth',0))):,}")
        k2.metric(f"Predicted Next Month ({next_m0:%b %Y})", f"{int(round(totals.get('nextmonth',0))):,}")
