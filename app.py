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
    st.error(f"Missing required columns: {missing}
Available: {list(df.columns)}")
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
# ================  PREDICTABILITY (Cohort-based) =========
# =========================================================
with tab_predict:
    # ----------------------------
    # Utilities specific to cohort forecast
    # ----------------------------
    def detect_payment_col(cols):
        for c in cols:
            cl = c.lower()
            if "payment" in cl and "received" in cl and "date" in cl:
                return c
        for c in cols:
            cl = c.lower()
            if "payment" in cl and "date" in cl:
                return c
        return None

    def to_month(dtser: pd.Series) -> pd.Series:
        return pd.to_datetime(dtser, errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()

    def month_add(ts: pd.Timestamp, k: int) -> pd.Timestamp:
        return (ts.to_period("M") + k).to_timestamp()

    def month_start(d: date) -> pd.Timestamp:
        return pd.Timestamp(d).to_period("M").to_timestamp()

    def month_days(ts: pd.Timestamp) -> int:
        m0 = ts
        m1 = (m0 + pd.offsets.MonthBegin(1))
        return int((m1 - m0).days)

    def safe_div(a, b):
        return (a / b) if b else 0.0

    # Empirical-Bayes smoothing for binomial rate
    def eb_rate(success: float, trials: float, prior_rate: float, prior_strength: float) -> float:
        return (success + prior_strength * prior_rate) / (trials + prior_strength) if (trials + prior_strength) > 0 else prior_rate

    # Day-of-month profile with EB smoothing and seasonal pooling (same month-of-year preferred)
    def day_of_month_profile(df_paid: pd.DataFrame, target_month: pd.Timestamp, group_mask: pd.Series, prior_strength: float = 5.0) -> pd.Series:
        dfp = df_paid.loc[group_mask].copy()
        if dfp.empty:
            days = month_days(target_month)
            return pd.Series(1.0 / days, index=pd.Index(range(1, days + 1), name="day"))
        dfp["d"] = dfp["Payment Received Date"].dt.day
        dfp["moy"] = dfp["Payment Received Date"].dt.month
        moy = int(target_month.month)
        pool = dfp[dfp["moy"] == moy]
        if pool.empty:
            pool = dfp
        days = month_days(target_month)
        idx = pd.Index(range(1, days + 1), name="day")
        cnt = pool["d"].value_counts().reindex(idx, fill_value=0).astype(float)
        total = cnt.sum()
        prior = pd.Series(1.0 / days, index=idx)
        smoothed = (cnt + prior_strength * prior) / (total + prior_strength)
        return smoothed / smoothed.sum()

    # Hour-of-day profile (0..23) with EB smoothing; used to split Today's prediction by hour
    def hour_of_day_profile(df_paid: pd.DataFrame, group_mask: pd.Series, prior_strength: float = 10.0) -> pd.Series:
        dfp = df_paid.loc[group_mask].copy()
        if dfp.empty or dfp["Payment Received Date"].isna().all():
            return pd.Series(1/24, index=pd.Index(range(24), name="hour"))
        hrs = dfp["Payment Received Date"].dt.hour
        idx = pd.Index(range(24), name="hour")
        cnt = hrs.value_counts().reindex(idx, fill_value=0).astype(float)
        total = cnt.sum()
        prior = pd.Series(1/24, index=idx)
        smoothed = (cnt + prior_strength * prior) / (total + prior_strength)
        return smoothed / smoothed.sum()

    # Backoff helpers for priors
    def compute_priors(df_cre: pd.DataFrame, df_paid_cohort: pd.DataFrame):
        """
        Returns dictionaries with prior rates for:
        - global
        - by Deal Source
        - by (Deal Source, Country)
        Keys:
          priors_global = {"r0": float, "r1": float, "n": int}
          priors_src[(src)] = {"r0": float, "r1": float, "n": int}
          priors_src_cty[(src, cty)] = {"r0": float, "r1": float, "n": int}
        """
        g_trials = len(df_cre)
        g_succ0 = int(((df_paid_cohort["Lag"] == 0)).sum())
        g_succ1 = int(((df_paid_cohort["Lag"] == 1)).sum())
        priors_global = {"r0": safe_div(g_succ0, g_trials), "r1": safe_div(g_succ1, g_trials), "n": g_trials}

        priors_src = {}
        if "JetLearn Deal Source" in df_cre.columns:
            for src, grp in df_cre.groupby("JetLearn Deal Source", dropna=False):
                gidx = grp.index
                sub = df_paid_cohort.loc[gidx]
                trials = len(grp)
                succ0 = int(((sub["Lag"] == 0)).sum())
                succ1 = int(((sub["Lag"] == 1)).sum())
                priors_src[src] = {"r0": safe_div(succ0, trials), "r1": safe_div(succ1, trials), "n": trials}

        priors_src_cty = {}
        if {"JetLearn Deal Source", "Country"}.issubset(df_cre.columns):
            for (src, cty), grp in df_cre.groupby(["JetLearn Deal Source", "Country"], dropna=False):
                gidx = grp.index
                sub = df_paid_cohort.loc[gidx]
                trials = len(grp)
                succ0 = int(((sub["Lag"] == 0)).sum())
                succ1 = int(((sub["Lag"] == 1)).sum())
                priors_src_cty[(src, cty)] = {"r0": safe_div(succ0, trials), "r1": safe_div(succ1, trials), "n": trials}

        return priors_global, priors_src, priors_src_cty

    def estimate_group_rates(df_cre: pd.DataFrame, df_paid_cohort: pd.DataFrame,
                             lookback_months: int, prior_strength_base: float = 25.0,
                             min_trials_for_local: int = 30) -> pd.DataFrame:
        """
        Compute EB-smoothed M0/M1 rates per finest group (Source×Country×Counsellor) with backoff priors.
        """
        last_month = df_cre["Create_Month"].max()
        if pd.isna(last_month):
            return pd.DataFrame(columns=["JetLearn Deal Source","Country","Student/Academic Counsellor","r0","r1","trials"])  # empty
        first_lb = month_add(last_month, -lookback_months + 1)
        mask_lb = (df_cre["Create_Month"] >= first_lb) & (df_cre["Create_Month"] <= last_month)
        cre_lb = df_cre.loc[mask_lb]
        paid_lb = df_paid_cohort.loc[cre_lb.index]

        pg, psrc, psrccty = compute_priors(cre_lb, paid_lb)

        recs = []
        grp_cols = ["JetLearn Deal Source", "Country", "Student/Academic Counsellor"]
        for keys, grp in cre_lb.groupby(grp_cols, dropna=False):
            gidx = grp.index
            sub = paid_lb.loc[gidx]
            trials = len(grp)
            succ0 = int(((sub["Lag"] == 0)).sum())
            succ1 = int(((sub["Lag"] == 1)).sum())

            src = keys[0]
            cty = keys[1]
            prior_r0, prior_r1, prior_n = pg["r0"], pg["r1"], max(pg["n"], 1)
            if (src, cty) in psrccty and psrccty[(src, cty)]["n"] >= 10:
                pr = psrccty[(src, cty)]
                prior_r0, prior_r1, prior_n = pr["r0"], pr["r1"], pr["n"]
            elif src in psrc and psrc[src]["n"] >= 10:
                pr = psrc[src]
                prior_r0, prior_r1, prior_n = pr["r0"], pr["r1"], pr["n"]

            prior_strength = prior_strength_base * (1.0 if trials < min_trials_for_local else 0.4)

            r0 = eb_rate(succ0, trials, prior_r0, prior_strength)
            r1 = eb_rate(succ1, trials, prior_r1, prior_strength)

            recs.append({
                "JetLearn Deal Source": keys[0],
                "Country": keys[1],
                "Student/Academic Counsellor": keys[2],
                "r0": float(r0),
                "r1": float(r1),
                "trials": int(trials)
            })

        rates = pd.DataFrame(recs)
        return rates

    def forecast_month_groupwise(df: pd.DataFrame, df_paid: pd.DataFrame, rates: pd.DataFrame,
                                 target_month: pd.Timestamp) -> pd.DataFrame:
        prev_month = month_add(target_month, -1)
        grp_cols = ["JetLearn Deal Source", "Country", "Student/Academic Counsellor"]
        dfc = df.copy()
        dfc["Create_Month"] = to_month(dfc["Create Date"])  

        cur_cre = dfc[dfc["Create_Month"] == target_month].groupby(grp_cols, dropna=False)["Create Date"].count().rename("C_cur")
        prev_cre = dfc[dfc["Create_Month"] == prev_month].groupby(grp_cols, dropna=False)["Create Date"].count().rename("C_prev")
        base = pd.concat([cur_cre, prev_cre], axis=1).fillna(0)

        out = base.reset_index().merge(rates, on=grp_cols, how="left")
        out[["r0","r1"]] = out[["r0","r1"]].fillna(out[["r0","r1"]].median().fillna(0))
        out["Forecast"] = out["r0"] * out["C_cur"] + out["r1"] * out["C_prev"]
        out["Forecast"] = out["Forecast"].clip(lower=0)
        out["Month"] = target_month
        return out[grp_cols + ["C_cur","C_prev","r0","r1","Forecast","Month"]]

    # ----------------------------
    # Detect payments and prep cohort frame
    # ----------------------------
    PAYMENT_COL = detect_payment_col(df.columns)
    if PAYMENT_COL is None:
        st.error("Couldn't find a payment date column. Add one like 'Payment Received Date'.")
        st.stop()

    dfX = df.copy()
    dfX["Create Date"] = pd.to_datetime(dfX["Create Date"], errors="coerce", dayfirst=True)
    dfX["Create_Month"] = to_month(dfX["Create Date"]) 
    dfX["Payment Received Date"] = pd.to_datetime(dfX[PAYMENT_COL], errors="coerce", dayfirst=True)
    paid = dfX[dfX["Payment Received Date"].notna()].copy()
    paid["PaymentMonth"] = to_month(paid["Payment Received Date"])  

    if paid.empty:
        st.error("No non-empty payment dates found.")
        st.stop()

    # Cohort merge: align each deal with its payment month (if any) and compute Lag
    df_cohort = dfX[["Create Date","Create_Month","JetLearn Deal Source","Country","Student/Academic Counsellor"]].copy()
    df_cohort["PaymentMonth"] = to_month(dfX["Payment Received Date"])  
    cm_code = df_cohort["Create_Month"].dt.year * 12 + df_cohort["Create_Month"].dt.month
    pm_code = df_cohort["PaymentMonth"].dt.year * 12 + df_cohort["PaymentMonth"].dt.month
    df_cohort["Lag"] = (pm_code - cm_code)

    # ----------------------------
    # Controls
    # ----------------------------
    st.markdown("### Cohort-based Predictability")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lookback = st.slider("Lookback window (months)", 3, 12, 6, 1)
    with c2:
        min_trials = st.slider("Min. creates for local rates", 10, 100, 30, 5)
    with c3:
        prior_strength = st.slider("Prior strength", 5, 100, 25, 5)
    with c4:
        group_view = st.selectbox("View level", [
            "Deal Source",
            "Country",
            "Counsellor",
            "Deal Source × Country",
            "Deal Source × Country × Counsellor"
        ])

    today_dt = pd.Timestamp.today().date()
    cm = month_start(today_dt)
    nm = month_add(cm, 1)

    # ----------------------------
    # Estimate rates (r0/r1) with EB
    # ----------------------------
    with st.spinner("Estimating conversion rates by cohort…"):
        rates = estimate_group_rates(
            df_cre=df_cohort,
            df_paid_cohort=df_cohort,
            lookback_months=lookback,
            prior_strength_base=float(prior_strength),
            min_trials_for_local=int(min_trials)
        )

    # ----------------------------
    # Forecast this & next month at finest level
    # ----------------------------
    fc_this = forecast_month_groupwise(dfX, paid, rates, cm)
    fc_next = forecast_month_groupwise(dfX, paid, rates, nm)

    def aggregate_view(df_fc: pd.DataFrame, view: str) -> pd.DataFrame:
        if view == "Deal Source":
            keys = ["JetLearn Deal Source"]
        elif view == "Country":
            keys = ["Country"]
        elif view == "Counsellor":
            keys = ["Student/Academic Counsellor"]
        elif view == "Deal Source × Country":
            keys = ["JetLearn Deal Source","Country"]
        else:
            keys = ["JetLearn Deal Source","Country","Student/Academic Counsellor"]
        agg = df_fc.groupby(keys, dropna=False)["Forecast"].sum().reset_index()
        return agg

    v_this = aggregate_view(fc_this, group_view).rename(columns={"Forecast":"This Month"})
    v_next = aggregate_view(fc_next, group_view).rename(columns={"Forecast":"Next Month"})
    merged_months = v_this.merge(v_next, on=v_this.columns[:-1].tolist(), how="outer").fillna(0)

    # ----------------------------
    # Today / Tomorrow using day-of-month allocation per group
    # ----------------------------
    st.markdown("### Today & Tomorrow (day/time aware)")
    with st.expander("Daily & hourly allocation settings", expanded=False):
        daily_prior_strength = st.slider("Daily profile prior strength", 1, 20, 5, 1)
        hourly_prior_strength = st.slider("Hourly profile prior strength", 5, 50, 10, 5)

    dom_today = today_dt.day
    dom_tom = (today_dt + timedelta(days=1)).day

    def mask_for_row(row: pd.Series) -> pd.Series:
        m = pd.Series(True, index=paid.index)
        if "JetLearn Deal Source" in row.index and not pd.isna(row["JetLearn Deal Source"]):
            m &= (paid["JetLearn Deal Source"].astype(str) == str(row["JetLearn Deal Source"]))
        if "Country" in row.index and not pd.isna(row.get("Country", np.nan)):
            m &= (paid["Country"].astype(str) == str(row["Country"]))
        if "Student/Academic Counsellor" in row.index and not pd.isna(row.get("Student/Academic Counsellor", np.nan)):
            m &= (paid["Student/Academic Counsellor"].astype(str) == str(row["Student/Academic Counsellor"]))
        return m

    vt = v_this.copy()
    vt["Today"] = 0.0
    vt["Tomorrow"] = 0.0

    for i, row in vt.iterrows():
        m = mask_for_row(row)
        dom_profile = day_of_month_profile(paid, cm, m, prior_strength=daily_prior_strength)
        p_today = float(dom_profile.reindex(range(1, month_days(cm) + 1)).get(dom_today, 0.0))
        p_tom = float(dom_profile.reindex(range(1, month_days(cm) + 1)).get(dom_tom, 0.0))
        vt.at[i, "Today"] = row["This Month"] * p_today
        vt.at[i, "Tomorrow"] = row["This Month"] * p_tom

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>Today ({pd.Timestamp(today_dt):%d %b})</div>
          <div class='value'>{int(round(vt['Today'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>Tomorrow ({pd.Timestamp(today_dt + timedelta(days=1)):%d %b})</div>
          <div class='value'>{int(round(vt['Tomorrow'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>This Month ({cm:%b %Y})</div>
          <div class='value'>{int(round(v_this['This Month'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>Next Month ({nm:%b %Y})</div>
          <div class='value'>{int(round(v_next['Next Month'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    st.markdown("#### Month-level Forecast (cohort-based)")
    st.dataframe(merged_months.sort_values("This Month", ascending=False), use_container_width=True)
    st.download_button("Download — Month Forecast CSV",
                       merged_months.to_csv(index=False).encode("utf-8"),
                       file_name="cohort_month_forecast.csv", mime="text/csv")

    st.markdown("#### Today & Tomorrow (by group)")
    st.dataframe(vt.sort_values("Today", ascending=False), use_container_width=True)
    st.download_button("Download — Today/Tomorrow CSV",
                       vt.to_csv(index=False).encode("utf-8"),
                       file_name="cohort_today_tomorrow.csv", mime="text/csv")

    st.markdown("#### Optional: Today's hourly breakdown (top 10 groups by Today forecast)")
    top_today = vt.sort_values("Today", ascending=False).head(10)
    if not top_today.empty:
        for _, row in top_today.iterrows():
            m = mask_for_row(row)
            hprof = hour_of_day_profile(paid, m, prior_strength=hourly_prior_strength)
            hour_df = hprof.rename("Prop").reset_index()
            hour_df["Forecast"] = (row["Today"] * hour_df["Prop"]).round(1)
            title_bits = []
            if "JetLearn Deal Source" in row.index and not pd.isna(row["JetLearn Deal Source"]):
                title_bits.append(str(row["JetLearn Deal Source"]))
            if "Country" in row.index and not pd.isna(row.get("Country", np.nan)):
                title_bits.append(str(row["Country"]))
            if "Student/Academic Counsellor" in row.index and not pd.isna(row.get("Student/Academic Counsellor", np.nan)):
                title_bits.append(str(row["Student/Academic Counsellor"]))
            st.caption(" — ".join(title_bits) or "Group")
            ch = alt.Chart(hour_df).mark_bar().encode(
                x=alt.X("hour:O", title=None),
                y=alt.Y("Forecast:Q", title=None),
                tooltip=["hour","Forecast","Prop"]
            ).properties(height=160)
            st.altair_chart(ch, use_container_width=True)

    st.caption("Model: Cohort-based (M0/M1) using EB-smoothed rates by Deal Source × Country × Counsellor; daily/hourly allocation from historical payment patterns of the same month-of-year. Excludes '1.2 Invalid Deal'.")
