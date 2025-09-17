# app.py — JetLearn Insights (MTD/Cohort) + Data-Specific ML Predictability (Payment Count)
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

def best_parse(series):
    s1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    s2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return s1 if s1.notna().sum() >= s2.notna().sum() else s2

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

def to_csv_bytes(df: pd.DataFrame)->bytes: return df.to_csv(index=False).encode("utf-8")

def group_label_from_series(s: pd.Series, grain: str):
    if grain=="Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain=="Week":
        iso=pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str)+"-W"+iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

def pick_by_alias(df, possible_names):
    lower = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower:
            return lower[name.lower()]
    for c in df.columns:
        cl = c.lower()
        if any(name.lower() in cl for name in possible_names):
            return c
    return None

def cap_categories(df, col, top_k=50):
    """Map infrequent categories to '__OTHER__' to keep OHE small & robust."""
    if col is None or col not in df.columns:
        return df, None
    s = df[col].astype(str).fillna("NA")
    vc = s.value_counts()
    keep = set(vc.head(top_k).index)
    new = col + "_capped"
    df[new] = s.where(s.isin(keep), "__OTHER__")
    return df, new

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

# --- Core columns (auto + UI fallback) ---
colmap = {c.lower(): c for c in df.columns}
def pick_exact(name_like): return colmap.get(name_like.lower())

COL_CREATE   = pick_exact("Create Date") or pick_by_alias(df, ["Create Date","Created Date","Deal Create Date"])
# payment date detector
COL_PAY = None
for c in df.columns:
    cl=c.lower()
    if "payment" in cl and "date" in cl: COL_PAY=c; break

if COL_CREATE is None:
    st.warning("Create Date not detected. Pick it below.")
    COL_CREATE = st.selectbox("Create Date column", options=df.columns)
if COL_PAY is None:
    st.warning("Payment Received Date not detected. Pick it below.")
    COL_PAY = st.selectbox("Payment Date column", options=df.columns)

COL_COUNTRY  = pick_exact("Country") or pick_by_alias(df, ["Country","Country Name"])
COL_SOURCE   = pick_exact("JetLearn Deal Source") or pick_by_alias(df, ["JetLearn Deal Source","Deal Source","Source"])
COL_CSL      = pick_exact("Student/Academic Counsellor") or pick_by_alias(df, ["Student/Academic Counsellor","Student/Academic Counselor","Academic Counsellor","Academic Counselor","Counsellor","Counselor"])
COL_TIMES    = pick_by_alias(df, ["Number of times contacted","Times Contacted","Times Contacted Count"])
COL_SALES    = pick_by_alias(df, ["Number of Sales Activities","Sales Activities"])
COL_LAST_ACT = pick_by_alias(df, ["Last Activity Date","Last Activity"])
COL_LAST_CNT = pick_by_alias(df, ["Last Contacted","Last Contacted Date"])

# Optional house-keeping
if exclude_invalid and "Deal Stage" in df.columns:
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

# Robust date parsing
df[COL_CREATE] = best_parse(df[COL_CREATE])
df[COL_PAY]    = best_parse(df[COL_PAY])
if COL_LAST_ACT and COL_LAST_ACT in df.columns:
    df[COL_LAST_ACT] = best_parse(df[COL_LAST_ACT])
if COL_LAST_CNT and COL_LAST_CNT in df.columns:
    df[COL_LAST_CNT] = best_parse(df[COL_LAST_CNT])

# Numeric coercion
if COL_TIMES and COL_TIMES in df.columns: df[COL_TIMES] = pd.to_numeric(df[COL_TIMES], errors="coerce").fillna(0)
if COL_SALES and COL_SALES in df.columns: df[COL_SALES] = pd.to_numeric(df[COL_SALES], errors="coerce").fillna(0)

# For Insights convenience
df["Create_Month"] = df[COL_CREATE].dt.to_period("M")
date_like_cols = detect_measure_date_columns(df)

# Quick integrity view
with st.expander("🔎 Data integrity check", expanded=False):
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Create Date non-null:", int(df[COL_CREATE].notna().sum()))
    st.write("Payment Date non-null:", int(df[COL_PAY].notna().sum()))

# --------------------- Tabs ---------------------
tab_insights, tab_predict = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictability (Data-Specific ML)"])

# =========================================================
# ===============  INSIGHTS (MTD / Cohort)  ===============
# (kept concise; unchanged logic from previous version)
# =========================================================
with tab_insights:
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
        if colname not in df_.columns:
            st.caption(f"({label} column missing)")
            return True, []
        options = sorted([v for v in df_[colname].dropna().astype(str).unique()])
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
        all_flag = bool(st.session_state.get(all_key, True)) if len(options)>0 else True
        selected = [v for v in coerce_list(st.session_state.get(ms_key, options)) if v in options] if len(options)>0 else []
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

        mask = pd.Series(True, index=df_.index)
        if "Pipeline" in df_.columns: mask &= in_filter(df_["Pipeline"], pipe_all, pipe_sel)
        if "JetLearn Deal Source" in df_.columns: mask &= in_filter(df_["JetLearn Deal Source"], src_all, src_sel)
        if "Country" in df_.columns: mask &= in_filter(df_["Country"], cty_all, cty_sel)
        if "Student/Academic Counsellor" in df_.columns: mask &= in_filter(df_["Student/Academic Counsellor"], csl_all, csl_sel)
        base = df_[mask].copy()

        st.markdown("##### Measures & Windows")
        mcol1,mcol2 = st.columns([3,2])
        with mcol1:
            measures = st.multiselect(f"[{name}] Measure date(s)",
                                      options=date_like_cols_,
                                      default=[date_like_cols_[0]] if date_like_cols_ else [],
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
            mtd_from, mtd_to, mtd_grain = date_preset_row(name, base["Create Date"] if "Create Date" in base.columns else base[COL_CREATE], f"{name}_mtd", default_grain="Month")
        if mode in ("Cohort","Both"):
            st.caption("Measure-Date window (Cohort)")
            series = base[measures[0]] if measures else base[COL_CREATE]
            coh_from, coh_to, coh_grain = date_preset_row(name, series, f"{name}_coh", default_grain="Month")

        with st.expander(f"[{name}] Splits & Leaderboards", expanded=False):
            sc1, sc2, sc3 = st.columns([3,2,2])
            split_dims = [d for d in ["JetLearn Deal Source","Country","Student/Academic Counsellor"] if d in base.columns]
            split_dims = sc1.multiselect(f"[{name}] Split by", split_dims, default=[], key=f"{name}_split")
            top_ctry = sc2.checkbox(f"[{name}] Top 5 Countries", value=True, key=f"{name}_top_cty")
            top_src  = sc3.checkbox(f"[{name}] Top 3 Deal Sources", value=True, key=f"{name}_top_src")
            top_csl  = st.checkbox(f"[{name}] Top 5 Counsellors", value=False, key=f"{name}_top_csl")
            pair     = st.checkbox(f"[{name}] Country × Deal Source (Top 10)", value=False, key=f"{name}_pair")

        return dict(
            name=name, base=base, measures=measures, mode=mode,
            mtd_from=mtd_from, mtd_to=mtd_to, mtd_grain=mtd_grain,
            coh_from=coh_from, coh_to=coh_to, coh_grain=coh_grain,
            split_dims=split_dims, top_ctry=top_ctry, top_src=top_src, top_csl=top_csl, pair=pair
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
            in_cre = base[COL_CREATE].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
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

                trend=sub.copy()
                trend["Bucket"]=group_label_from_series(trend[COL_CREATE], mtd_grain)
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
            in_cre_coh = base[COL_CREATE].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
            metrics_rows.append({"Scope":"Cohort","Metric":"Create Count in Cohort window","Window":f"{coh_from} → {coh_to}","Value":int(in_cre_coh.sum())})

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

    metaA = scenario_controls("A", df, date_like_cols)
    with st.spinner("Calculating…"):
        metricsA, tablesA, chartsA = compute_outputs(metaA)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
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

# =========================================================
# =====  PREDICTABILITY (Data-Specific ML, robust)   ======
# =========================================================
with tab_predict:
    st.markdown("### 🔮 Predictability — Payment Count (data-specific, leak-free, chunked scoring)")
    st.caption("Training uses **all history before the running month** (Asia/Kolkata). You can choose which features to use — the model adapts to the columns in your CSV.")

    # ---------- Leak-free cutoff (Asia/Kolkata) ----------
    tz = "Asia/Kolkata"
    now_ist = pd.Timestamp.now(tz=tz)
    cm_start = pd.Timestamp(year=now_ist.year, month=now_ist.month, day=1, tz=tz).tz_convert(None)
    today = pd.Timestamp.now(tz=tz).tz_convert(None).normalize()
    end_this = today.to_period("M").to_timestamp("M")
    end_next = (today.to_period("M") + 1).to_timestamp("M")

    # ---------- Feature selection (DATA-SPECIFIC) ----------
    st.markdown("#### Choose features present in this file")
    available_cats = { "Country": COL_COUNTRY, "JetLearn Deal Source": COL_SOURCE, "Student/Academic Counsellor": COL_CSL }
    available_nums = { "Number of times contacted": COL_TIMES, "Number of Sales Activities": COL_SALES }
    available_dates= { "Last Activity Date": COL_LAST_ACT, "Last Contacted": COL_LAST_CNT }

    use_cat = st.multiselect("Categorical", [k for k,v in available_cats.items() if v is not None],
                             default=[k for k,v in available_cats.items() if v is not None])
    use_num = st.multiselect("Numeric", [k for k,v in available_nums.items() if v is not None],
                             default=[k for k,v in available_nums.items() if v is not None])
    use_dt  = st.multiselect("Date/Recency features", [k for k,v in available_dates.items() if v is not None],
                             default=[k for k,v in available_dates.items() if v is not None])

    include_inflow = st.toggle("Include new-deal inflow (M0/M1)", value=True)
    fast_mode = st.toggle("Fast mode (only score until end of this month)", value=False)
    if fast_mode:
        end_next = end_this

    # ---------- Prepare training set (positives + sampled negatives) ----------
    NEG_PER_POS = 5
    MAX_AGE_DAYS = 150

    X = df.copy()
    X["Create"] = pd.to_datetime(X[COL_CREATE]).dt.normalize()
    X["Pay"]    = pd.to_datetime(X[COL_PAY]).dt.normalize()

    # cap categories to keep OHE compact
    cap_map = {}
    for label in use_cat:
        base_col = available_cats[label]
        X, capped = cap_categories(X, base_col, top_k=50)
        cap_map[label] = capped

    # build training rows
    def build_training(X_: pd.DataFrame) -> pd.DataFrame:
        D = X_[X_["Create"].notna()].copy().reset_index(drop=True)
        if D.empty: return pd.DataFrame()
        D["deal_id"] = np.arange(len(D))

        base_cols = ["deal_id","Create"]
        # include chosen categorical (capped)
        for label in use_cat:
            if cap_map.get(label): base_cols.append(cap_map[label])
        # include chosen numeric
        for label in use_num:
            col = available_nums[label]
            if col and col in D.columns: base_cols.append(col)
        # include last recency dates (raw; converted later)
        for label in use_dt:
            col = available_dates[label]
            if col and col in D.columns: base_cols.append(col)

        # positives: first payments strictly before current month
        pos_cols = base_cols + ["Pay"]
        pos = D[D["Pay"].notna() & (D["Pay"] < cm_start)][pos_cols].copy()
        pos.rename(columns={"Pay":"day"}, inplace=True)
        pos["y"] = 1.0

        rng = np.random.default_rng(42)
        neg_rows = []

        # negatives from paid deals pre-pay
        for _, r in pos.iterrows():
            d0 = pd.to_datetime(r["Create"]).normalize()
            dp = pd.to_datetime(r["day"]).normalize()
            d1 = min(dp - pd.Timedelta(days=1), cm_start - pd.Timedelta(days=1))
            if d1 < d0: continue
            span = (d1.date() - d0.date()).days + 1
            take = min(span, NEG_PER_POS)
            offs = rng.choice(span, size=take, replace=False)
            rr = r.to_dict()
            for o in offs:
                row = dict(rr)
                row["day"] = (d0 + pd.Timedelta(days=int(o)))
                row["y"] = 0.0
                neg_rows.append(row)

        # negatives from unpaid deals as of cutoff
        unpaid = D[D["Pay"].isna() & (D["Create"] < cm_start)][base_cols].copy()
        for _, r in unpaid.iterrows():
            d0 = pd.to_datetime(r["Create"]).normalize()
            d1 = cm_start - pd.Timedelta(days=1)
            if d1 < d0: continue
            span = (d1.date() - d0.date()).days + 1
            take = min(span, NEG_PER_POS)
            offs = rng.choice(span, size=take, replace=False)
            rr = r.to_dict()
            for o in offs:
                row = dict(rr)
                row["day"] = (d0 + pd.Timedelta(days=int(o)))
                row["y"] = 0.0
                neg_rows.append(row)

        neg = pd.DataFrame(neg_rows) if neg_rows else pd.DataFrame(columns=pos.columns)
        train = pd.concat([pos, neg], ignore_index=True)
        return train

    with st.spinner("Preparing & training…"):
        train = build_training(X)
        if train.empty:
            st.info("Not enough historical data to train."); st.stop()

        # feature engineering for training rows
        train["day"] = pd.to_datetime(train["day"]).dt.normalize()
        train["age"] = (train["day"] - pd.to_datetime(train["Create"]).dt.normalize()).dt.days.clip(lower=0, upper=MAX_AGE_DAYS).astype(int)
        train["moy"] = train["day"].dt.month.astype(int)
        train["dow"] = train["day"].dt.dayofweek.astype(int)
        train["dom"] = train["day"].dt.day.astype(int)

        # recencies
        def recency(series_name):
            if series_name is None or series_name not in train.columns: return 365
            d = pd.to_datetime(train[series_name], errors="coerce")
            return (train["day"] - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        # add selected recencies
        if "Last Activity Date" in use_dt and available_dates["Last Activity Date"] in train.columns:
            train["rec_act"] = recency(available_dates["Last Activity Date"])
        else:
            train["rec_act"] = 365
        if "Last Contacted" in use_dt and available_dates["Last Contacted"] in train.columns:
            train["rec_cnt"] = recency(available_dates["Last Contacted"])
        else:
            train["rec_cnt"] = 365

        # numeric counts
        if "Number of times contacted" in use_num and available_nums["Number of times contacted"] in train.columns:
            train["times"] = pd.to_numeric(train[available_nums["Number of times contacted"]], errors="coerce").fillna(0)
        else:
            train["times"] = 0
        if "Number of Sales Activities" in use_num and available_nums["Number of Sales Activities"] in train.columns:
            train["sales"] = pd.to_numeric(train[available_nums["Number of Sales Activities"]], errors="coerce").fillna(0)
        else:
            train["sales"] = 0

        # build lists of features present
        num_cols = ["age","moy","dow","dom","rec_act","rec_cnt","times","sales"]
        num_cols = [c for c in num_cols if c in train.columns]
        cat_cols = []
        for label in use_cat:
            col = cap_map.get(label)
            if col and col in train.columns:
                cat_cols.append(col)

        # model: HGBT + OHE
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import HistGradientBoostingClassifier

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

        # one-class guard
        pos_count = int((train["y"] == 1).sum())
        neg_count = int((train["y"] == 0).sum())
        use_baseline = (pos_count == 0 or neg_count == 0)
        if use_baseline:
            st.warning("Not enough pre-current-month payments to train; using a baseline rate.")
        else:
            pipe.fit(train[num_cols + cat_cols], train["y"])

    # ---------- Scoring (chunked per day; memory safe) ----------
    # active deals only (unpaid as of a given day)
    deals = X.copy()
    deals["deal_id"] = np.arange(len(deals))

    # optional day range
    date_range = pd.date_range(start=today, end=end_next, freq="D")

    def make_features_for_day(day, subset):
        F = subset.copy()
        F["age"] = (day - F["Create"]).dt.days.clip(lower=0, upper=MAX_AGE_DAYS).astype(int)
        F["moy"] = day.month
        F["dow"] = day.dayofweek
        F["dom"] = day.day
        # recencies
        if "Last Activity Date" in use_dt and available_dates["Last Activity Date"] in F.columns:
            d = pd.to_datetime(F[available_dates["Last Activity Date"]], errors="coerce")
            F["rec_act"] = (day - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        else:
            F["rec_act"] = 365
        if "Last Contacted" in use_dt and available_dates["Last Contacted"] in F.columns:
            d = pd.to_datetime(F[available_dates["Last Contacted"]], errors="coerce")
            F["rec_cnt"] = (day - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        else:
            F["rec_cnt"] = 365
        # numeric
        if "Number of times contacted" in use_num and available_nums["Number of times contacted"] in F.columns:
            F["times"] = pd.to_numeric(F[available_nums["Number of times contacted"]], errors="coerce").fillna(0)
        else:
            F["times"] = 0
        if "Number of Sales Activities" in use_num and available_nums["Number of Sales Activities"] in F.columns:
            F["sales"] = pd.to_numeric(F[available_nums["Number of Sales Activities"]], errors="coerce").fillna(0)
        else:
            F["sales"] = 0
        # ensure capped cat columns exist
        for label in use_cat:
            col = cap_map.get(label)
            if col and col not in F.columns:
                F[col] = "__OTHER__"
        return F

    # Grouping options (data-specific)
    group_opts = []
    if COL_CSL: group_opts.append("Student/Academic Counsellor")
    if COL_SOURCE: group_opts.append("JetLearn Deal Source")
    if COL_COUNTRY: group_opts.append("Country")
    group_opts += ["Day", "Day of Week"]
    group_by = st.multiselect("Breakdowns (group by)", options=group_opts,
                              default=[x for x in ["JetLearn Deal Source","Country","Student/Academic Counsellor","Day"] if x in group_opts])

    # map grouping to actual columns (using capped cats where appropriate)
    g_cols = []
    rename_map = {}
    if "Student/Academic Counsellor" in group_by and COL_CSL:
        col = cap_map.get("Student/Academic Counsellor") or COL_CSL
        g_cols.append(col); rename_map[col] = "Counsellor"
    if "JetLearn Deal Source" in group_by and COL_SOURCE:
        col = cap_map.get("JetLearn Deal Source") or COL_SOURCE
        g_cols.append(col); rename_map[col] = "Deal Source"
    if "Country" in group_by and COL_COUNTRY:
        col = cap_map.get("Country") or COL_COUNTRY
        g_cols.append(col); rename_map[col] = "Country"

    # chunk scoring per day
    rows = []
    for day in date_range:
        # active = created by day and not already paid before day
        active = deals[(deals["Create"] <= day) & (deals["Pay"].isna() | (deals["Pay"] >= day))].copy()
        if active.empty:
            continue
        F = make_features_for_day(day, active)

        if use_baseline:
            # baseline daily probability = (avg daily payments historically) / (active deals today)
            hist = X[(X["Pay"].notna()) & (X["Pay"] < cm_start)]
            if hist.empty:
                p = 0.0
            else:
                per_day = hist["Pay"].dt.date.value_counts().mean()  # avg daily payments across history
                p = float(per_day) / max(1, len(F))
                p = max(0.0, min(0.2, p))  # cap
            probs = np.full(len(F), p)
        else:
            feats = F[num_cols + cat_cols] if (num_cols or cat_cols) else pd.DataFrame(index=F.index)
            probs = pipe.predict_proba(feats)[:,1]

        F["p"] = probs
        F["Day"] = str(day.date())
        F["Day of Week"] = day.day_name()

        keep_cols = ["p","Day","Day of Week"] + g_cols
        rows.append(F[keep_cols])

    pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["p","Day","Day of Week"]+g_cols)
    if pred.empty:
        st.info("No active deals to score in the selected horizon."); st.stop()

    # ---------- New-deal inflow (optional) ----------
    if include_inflow:
        # last full month window
        lm_start = (cm_start - pd.offsets.MonthBegin(1))
        lm_end   = (cm_start - pd.Timedelta(days=1))
        lastm = X[(X["Create"] >= lm_start) & (X["Create"] <= lm_end)].copy()
        # cohort M0/M1 rates by grouped caps (using same capped columns)
        def month_code(ts): return ts.dt.year * 12 + ts.dt.month
        dfC = X.copy()
        dfC["Create_M"] = dfC["Create"].dt.to_period("M").dt.to_timestamp()
        dfC["Pay_M"]    = dfC["Pay"].dt.to_period("M").dt.to_timestamp()
        for label in ["Country","JetLearn Deal Source","Student/Academic Counsellor"]:
            if label in use_cat and cap_map.get(label):
                col = cap_map[label]
                dfC[col] = dfC[col].astype(str).fillna("NA")
        # pick grouping keys aligned with g_cols (to join nicely)
        keys = g_cols.copy()
        if not keys:
            keys = []  # global rates

        hist = dfC[dfC["Create_M"].notna() & (dfC["Create_M"] < cm_start)].copy()
        if not hist.empty:
            cm = month_code(hist["Create_M"])
            pm = month_code(hist["Pay_M"])
            hist["Lag"] = (pm - cm)

            by = keys if keys else None
            if keys:
                trials = hist.groupby(keys)["Create_M"].count().rename("trials")
                succ0  = hist[hist["Lag"]==0].groupby(keys)["Lag"].count().rename("succ0")
                succ1  = hist[hist["Lag"]==1].groupby(keys)["Lag"].count().rename("succ1")
            else:
                trials = pd.Series({"_": hist["Create_M"].count()}, name="trials")
                succ0  = pd.Series({"_": hist[hist["Lag"]==0]["Lag"].count()}, name="succ0")
                succ1  = pd.Series({"_": hist[hist["Lag"]==1]["Lag"].count()}, name="succ1")

            rates = pd.concat([trials, succ0, succ1], axis=1).fillna(0)
            g_trials = float(rates["trials"].sum())
            g_r0 = (rates["succ0"].sum()/g_trials) if g_trials>0 else 0.0
            g_r1 = (rates["succ1"].sum()/g_trials) if g_trials>0 else 0.0
            alpha = 20.0
            rates["r0"] = (rates["succ0"] + alpha*g_r0) / (rates["trials"] + alpha)
            rates["r1"] = (rates["succ1"] + alpha*g_r1) / (rates["trials"] + alpha)
            rates = rates.reset_index()
        else:
            rates = pd.DataFrame(columns=keys+["trials","succ0","succ1","r0","r1"])

        # creates per DOW from last full month
        if not lastm.empty:
            # align capped grouping cols
            for label in use_cat:
                col = cap_map.get(label)
                if col and col in lastm.columns:
                    lastm[col] = lastm[col].astype(str).fillna("NA")
            lastm["dow"] = lastm["Create"].dt.dayofweek
            days_last_month = pd.date_range(lm_start, lm_end, freq="D")
            dow_counts = pd.Series(days_last_month.dayofweek).value_counts().to_dict()
            for d in range(7): dow_counts.setdefault(d, 0)
            gcols = keys.copy()
            creates_dow = lastm.groupby(gcols+["dow"])["Create"].count().rename("creates").reset_index()
            creates_dow["rate_per_day"] = creates_dow.apply(lambda r: (r["creates"] / max(1, dow_counts[int(r["dow"])])), axis=1)

            # expected creates for each forecast day
            future_days = pd.date_range(start=today, end=end_next, freq="D")
            recs=[]
            for d in future_days:
                dw = int(d.dayofweek)
                match = creates_dow[creates_dow["dow"]==dw]
                if match.empty: continue
                for _, r in match.iterrows():
                    recs.append(tuple([*(r[k] for k in gcols), str(d.date()), float(r["rate_per_day"])]))
            if recs:
                E = pd.DataFrame(recs, columns=gcols+["Day","exp_creates"])
                exp_creates = E.groupby(gcols+["Day"])["exp_creates"].sum().reset_index()
            else:
                exp_creates = pd.DataFrame(columns=gcols+["Day","exp_creates"])
        else:
            exp_creates = pd.DataFrame(columns=keys+["Day","exp_creates"])

        # turn creates into expected payments using r0/r1
        if not exp_creates.empty and not rates.empty:
            rkey = rates.set_index(keys if keys else rates.columns[:0].tolist())[["r0","r1"]].to_dict(orient="index")
            # split days into this & next month
            end_this_s = str(end_this.date())
            exp_creates["month_flag"] = np.where(exp_creates["Day"] <= end_this_s, "this", "next")
            # totals by group over months
            tot = exp_creates.groupby(keys+["month_flag"])["exp_creates"].sum().reset_index()
            # distribute uniformly across days within each month
            days_this = [str(d.date()) for d in pd.date_range(start=today, end=end_this, freq="D")]
            days_next = [str(d.date()) for d in pd.date_range(start=end_this+pd.Timedelta(days=1), end=end_next, freq="D")]
            n_this = max(1, len(days_this)); n_next = max(1, len(days_next))

            inflow_rows=[]
            for _, r in tot.iterrows():
                gvals = tuple(r[k] for k in keys) if keys else tuple()
                rr = rkey.get(gvals, {"r0": rates["r0"].mean() if "r0" in rates else 0.0,
                                      "r1": rates["r1"].mean() if "r1" in rates else 0.0})
                if r["month_flag"]=="this":
                    m0_total = float(r["exp_creates"]) * float(rr["r0"])
                    per_day = m0_total / n_this
                    for d in days_this:
                        inflow_rows.append((*gvals, d, per_day))
                    # spill M1 of this month's creates into next month
                    if not fast_mode:
                        m1_total = float(r["exp_creates"]) * float(rr["r1"])
                        per_day = m1_total / n_next
                        for d in days_next:
                            inflow_rows.append((*gvals, d, per_day))
                else:
                    # M0 for next month's creates
                    m0n_total = float(r["exp_creates"]) * float(rr["r0"])
                    per_day = m0n_total / n_next
                    for d in days_next:
                        inflow_rows.append((*gvals, d, per_day))
            inflow = pd.DataFrame(inflow_rows, columns=keys+["Day","p"])
        else:
            inflow = pd.DataFrame(columns=keys+["Day","p"])
    else:
        inflow = pd.DataFrame(columns=["Day","p"]+g_cols)

    # ---------- Combine existing-deal predictions + inflow ----------
    pred_all = pred.copy()
    if include_inflow and not inflow.empty:
        # align column names with pred
        inflow2 = inflow.copy()
        # rename capped cols to display names if needed
        inflow2.rename(columns=rename_map, inplace=True)
        pred_all.rename(columns=rename_map, inplace=True)
        pred_all = pd.concat([pred_all, inflow2], ignore_index=True)
    else:
        pred_all.rename(columns=rename_map, inplace=True)

    # ---------- Summaries ----------
    def summarize(mask, label):
        sub = pred_all.loc[mask]
        gcols = [c for c in ["Counsellor","Deal Source","Country","Day","Day of Week"] if c in sub.columns]
        if not gcols:
            return pd.DataFrame({label:[int(round(sub['p'].sum()))]})
        g = sub.groupby(gcols, dropna=False)["p"].sum().reset_index()
        g[label] = g["p"].round(0).astype(int)
        return g.drop(columns=["p"])

    mask_today = pred_all["Day"] == str(today.date())
    mask_tom   = pred_all["Day"] == str((today + pd.Timedelta(days=1)).date())
    mask_m     = pred_all["Day"].between(str(today.date()), str(end_this.date()))
    mask_n     = pred_all["Day"].between(str((end_this+pd.Timedelta(days=1)).date()), str(end_next.date()))

    g_today = summarize(mask_today, "Today")
    g_tom   = summarize(mask_tom, "Tomorrow")
    g_this  = summarize(mask_m, "This Month")
    g_next  = summarize(mask_n, "Next Month")

    def smart_merge(a,b):
        if a.empty: return b
        if b.empty: return a
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
    disp_cols = [c for c in ["Counsellor","Deal Source","Country","Day","Day of Week","Today","Tomorrow","This Month","Next Month"] if c in out.columns]
    st.dataframe(out[disp_cols].sort_values(by=[c for c in ["Today","This Month","Next Month"] if c in out.columns], ascending=False),
                 use_container_width=True)
    st.download_button("Download — ML Predictability CSV",
                       out[disp_cols].to_csv(index=False).encode("utf-8"),
                       file_name="ml_predictability_payment_count_inflow.csv",
                       mime="text/csv")

    if "Day" in disp_cols:
        try:
            long = out.melt(id_vars=[c for c in disp_cols if c not in ["Today","Tomorrow","This Month","Next Month"]],
                            value_vars=[c for c in ["Today","Tomorrow","This Month","Next Month"] if c in out.columns],
                            var_name="Bucket", value_name="Count")
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

    st.caption("Counts = predicted payments from **existing open deals** (data-specific features) "
               "+ expected payments from **new deals** (M0/M1 from history, creates by DOW).")
