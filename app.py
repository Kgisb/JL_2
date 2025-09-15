# app_compare_smart.py — Pro, mobile-friendly, multi-measure A/B analyzer (smart UI)
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ------------------------ Page & Theme ------------------------
st.set_page_config(page_title="MTD vs Cohort — A/B Compare (Smart UI)", layout="wide", page_icon="✨")

st.markdown("""
<style>
/* Base & typography */
:root{
  --bg:#ffffff; --card:#ffffff; --muted:#6b7280; --text:#0f172a; --accent:#2563eb;
  --border: rgba(15,23,42,.12); --soft: rgba(2,6,23,.05);
}
html, body, [class*="css"]  { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
                              Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; }
.block-container { padding-top: 0.6rem; padding-bottom: 1.2rem; }

/* Sticky header bar */
.header {
  position: sticky; top: 0; z-index: 10; background: var(--bg);
  border-bottom: 1px solid var(--border); padding: 8px 0 6px;
}
.h-title { font-weight: 700; font-size: 1.05rem; }
.h-sub   { color: var(--muted); font-size: .82rem; margin-top: 2px; }

/* Cards & sections */
.card {
  border: 1px solid var(--border); background: var(--card); border-radius: 14px; padding: 12px 14px;
}
.section-title {
  display:flex; align-items:center; gap:.5rem; font-weight:700; margin:.25rem 0 .5rem 0;
}
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); color:#111827; background:#f9fafb; }
hr.soft { border:0; height:1px; background:var(--border); margin: .6rem 0 1rem; }

/* KPI card */
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:750; line-height:1.05; color:var(--text); }
.kpi .delta { font-size:.85rem; color:var(--accent); }

/* Mobile tweaks */
@media (max-width: 820px) {
  .block-container { padding-left: .5rem; padding-right: .5rem; }
  .h-title { font-size: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------ Constants ------------------------
REQUIRED_COLS = [
    "Pipeline","JetLearn Deal Source","Country",
    "Student/Academic Counsellor","Deal Stage","Create Date",
]

# ------------------------ Utilities ------------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    """Return list of date-like columns excluding 'Create Date'; parse to datetime."""
    date_like = []
    for col in df.columns:
        if col == "Create Date":
            continue
        if any(k in col.lower() for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed
                date_like.append(col)
    # Prefer 'Payment Received Date' first if present
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c != "Payment Received Date"]
    return date_like

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    if all_checked:
        return pd.Series(True, index=series.index)
    uniq = series.dropna().astype(str).nunique()
    if selected_values and len(selected_values) == uniq:
        return pd.Series(True, index=series.index)
    if not selected_values:
        return pd.Series(False, index=series.index)
    return series.astype(str).isin(selected_values)

def safe_minmax_date(s: pd.Series, fallback=(date(2020,1,1), date.today())):
    if s.isna().all():
        return fallback
    return (pd.to_datetime(s.min()).date(), pd.to_datetime(s.max()).date())

def this_month_bounds():
    today = pd.Timestamp.today().date()
    start = today.replace(day=1)
    next_start = (start.replace(year=start.year+1, month=1) if start.month==12 else start.replace(month=start.month+1))
    return start, (next_start - timedelta(days=1))

def last_month_bounds():
    first_this = pd.Timestamp.today().date().replace(day=1)
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return first_prev, last_prev

def last_7_days_bounds():
    today = pd.Timestamp.today().date()
    return today - timedelta(days=6), today

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def load_and_prepare(data_bytes, path_text):
    # Load
    if data_bytes:
        df = robust_read_csv(BytesIO(data_bytes))
    elif path_text:
        df = robust_read_csv(path_text)
    else:
        raise ValueError("Please upload a CSV or provide a file path.")
    df.columns = [c.strip() for c in df.columns]

    # Required cols
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

    # Exclude invalid
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

    # Parse main dates
    df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
    df["Create_Month"] = df["Create Date"].dt.to_period("M")

    # Detect other date-like
    date_like_cols = detect_measure_date_columns(df)

    return df, date_like_cols

# ------------------------ Altair theme (colors) ------------------------
# Simple palette used consistently
PALETTE = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#ec4899"]

def altair_themed_line(df, x, y, color=None, tooltip=None, height=260):
    ch = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(x, title=None),
        y=alt.Y(y, title=None),
        color=alt.Color(color, scale=alt.Scale(range=PALETTE)) if color else alt.value(PALETTE[0]),
        tooltip=tooltip or []
    ).properties(height=height)
    return ch

# ------------------------ Header ------------------------
with st.container():
    st.markdown('<div class="header">', unsafe_allow_html=True)
    h1, h2 = st.columns([7,5])
    with h1:
        st.markdown('<div class="h-title">✨ MTD vs Cohort — A/B Compare (Smart)</div>', unsafe_allow_html=True)
        st.markdown('<div class="h-sub">Minimal, mobile-friendly analysis • multi-measure • splits • smart compare</div>', unsafe_allow_html=True)
    with h2:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Clone A → B"):
                for k in list(st.session_state.keys()):
                    if str(k).startswith("A_"):
                        st.session_state[str(k).replace("A_","B_",1)] = st.session_state[k]
                st.rerun()
        with c2:
            if st.button("Clone B → A"):
                for k in list(st.session_state.keys()):
                    if str(k).startswith("B_"):
                        st.session_state[str(k).replace("B_","A_",1)] = st.session_state[k]
                st.rerun()
        with c3:
            if st.button("Reset"):
                st.session_state.clear()
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------ Data source ------------------------
with st.expander("📦 Data source", expanded=True):
    col_u, col_p = st.columns([3,2])
    with col_u:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with col_p:
        default_path = st.text_input("…or CSV path", value="Master_sheet_DB_10percent.csv")
    try:
        df, date_like_cols = load_and_prepare(uploaded.getvalue() if uploaded else None,
                                              default_path if not uploaded else None)
        st.success("Data loaded", icon="✅")
    except Exception as e:
        st.error(str(e))
        st.stop()

if not date_like_cols:
    st.error("No date-like columns found besides 'Create Date' (e.g., 'Payment Received Date').")
    st.stop()

# ------------------------ UI helpers ------------------------
def filter_block(df, label, colname, key_prefix):
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])
    with st.container():
        c1, c2 = st.columns([1,3])
        all_flag = c1.toggle("All", value=True, key=f"{key_prefix}_all")
        selected = c2.multiselect(label, options, default=options, disabled=all_flag, key=f"{key_prefix}_sel")
    return all_flag, selected

def date_range_slider(name, series, key_prefix):
    """Date slider based on series min/max (mobile-friendly)."""
    dmin, dmax = safe_minmax_date(series)
    return st.slider(f"[{name}] Range", min_value=dmin, max_value=dmax, value=(dmin, dmax), key=f"{key_prefix}_slider")

def ensure_month_cols(base: pd.DataFrame, measures):
    for m in measures:
        col = f"{m}_Month"
        if col not in base.columns:
            base[col] = base[m].dt.to_period("M")
    return base

def panel_controls(name: str, df: pd.DataFrame, date_like_cols):
    st.markdown(f"<div class='section-title'>Scenario {name} <span class='badge'>independent</span></div>", unsafe_allow_html=True)
    with st.container():
        with st.container():
            with st.expander(f"[{name}] Global filters", expanded=True):
                g1, g2 = st.columns(2)
                with g1:
                    pipe_all, pipe_sel = filter_block(df, "Pipeline", "Pipeline", f"{name}_pipe")
                    src_all,  src_sel  = filter_block(df, "Deal Source", "JetLearn Deal Source", f"{name}_src")
                with g2:
                    ctry_all, ctry_sel = filter_block(df, "Country", "Country", f"{name}_ctry")
                    cslr_all, cslr_sel = filter_block(df, "Counsellor", "Student/Academic Counsellor", f"{name}_cslr")

        mask_cat = (
            in_filter(df["Pipeline"], pipe_all, pipe_sel) &
            in_filter(df["JetLearn Deal Source"], src_all, src_sel) &
            in_filter(df["Country"], ctry_all, ctry_sel) &
            in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
        )
        base = df[mask_cat].copy()

        st.markdown("###### Measures & windows")
        mrow1 = st.columns([3,2,2])
        measures = mrow1[0].multiselect(f"[{name}] Measure date(s) — counts per field",
                                        options=date_like_cols,
                                        default=[date_like_cols[0]] if date_like_cols else [],
                                        key=f"{name}_measures")
        # Segment control for which windows are active
        window_mode = mrow1[1].radio(f"[{name}] Mode", ["MTD","Cohort","Both"], horizontal=True, key=f"{name}_mode")
        compact = mrow1[2].toggle("Compact Mode (mobile)", value=False, key=f"{name}_compact")

        mtd = window_mode in ("MTD","Both")
        cohort = window_mode in ("Cohort","Both")

        if not measures:
            st.warning("Pick at least one Measure date.")
            measures = []

        base = ensure_month_cols(base, measures)

        # Date pickers with sliders (mobile)
        c1, c2 = st.columns(2)
        mtd_from = mtd_to = coh_from = coh_to = None
        if mtd:
            with c1:
                st.caption("Create-Date window")
                mtd_from, mtd_to = date_range_slider(name + " · MTD", base["Create Date"], f"{name}_mtd")
        if cohort:
            with c2:
                st.caption("Measure-Date window")
                # for default slider bounds, use first measure if any
                first_series = base[measures[0]] if measures else base["Create Date"]
                coh_from, coh_to = date_range_slider(name + " · Cohort", first_series, f"{name}_coh")

        # Splits & leaderboards (reveal on demand)
        with st.expander(f"[{name}] Splits & leaderboards (optional)", expanded=False):
            srow = st.columns([3,2,2])
            split_dims = srow[0].multiselect(f"[{name}] Split by", ["JetLearn Deal Source", "Country"], default=[], key=f"{name}_split")
            show_top_countries = srow[1].toggle("Top 5 Countries", value=True, key=f"{name}_top_ctry")
            show_top_sources   = srow[2].toggle("Top 3 Deal Sources", value=True, key=f"{name}_top_src")
            show_combo_pairs   = st.toggle("Country × Deal Source (Top 10)", value=False, key=f"{name}_pair")

        return dict(
            name=name, base=base, measures=measures, mtd=mtd, cohort=cohort,
            mtd_from=mtd_from, mtd_to=mtd_to, coh_from=coh_from, coh_to=coh_to,
            split_dims=split_dims, show_top_countries=show_top_countries,
            show_top_sources=show_top_sources, show_combo_pairs=show_combo_pairs,
            pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
            ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel,
            compact=compact
        )

def compute_outputs(meta):
    base = meta["base"]; measures = meta["measures"]
    mtd = meta["mtd"]; cohort = meta["cohort"]
    mtd_from, mtd_to = meta["mtd_from"], meta["mtd_to"]
    coh_from, coh_to = meta["coh_from"], meta["coh_to"]
    split_dims = meta["split_dims"]
    show_top_countries = meta["show_top_countries"]
    show_top_sources   = meta["show_top_sources"]
    show_combo_pairs   = meta["show_combo_pairs"]

    metrics_rows = []
    tables = {}
    charts = {}

    # ---------- MTD ----------
    if mtd and mtd_from and mtd_to and len(measures)>0:
        in_create_window = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
        sub_mtd = base[in_create_window].copy()

        # Flags per measure
        mtd_flag_cols = []
        for m in measures:
            col_flag = f"__MTD__{m}"
            sub_mtd[col_flag] = ((sub_mtd[m].notna()) & (sub_mtd[f"{m}_Month"] == sub_mtd["Create_Month"])).astype(int)
            mtd_flag_cols.append(col_flag)
            metrics_rows.append({"Scope":"MTD", "Metric":f"Count on '{m}'", "Window":f"{mtd_from} → {mtd_to}",
                                 "Value": int(sub_mtd[col_flag].sum())})
        metrics_rows.append({"Scope":"MTD", "Metric":"Create Count in window", "Window":f"{mtd_from} → {mtd_to}",
                             "Value": int(len(sub_mtd))})

        # Splits
        if split_dims:
            agg_dict = {flag: "sum" for flag in mtd_flag_cols}
            agg_dict["_CreateCount"] = "sum"
            sub_mtd["_CreateCount"] = 1
            grp = sub_mtd.groupby(split_dims, dropna=False).agg(agg_dict).reset_index()
            rename_map = {f"__MTD__{m}": f"MTD: {m}" for m in measures}
            grp = grp.rename(columns={"_CreateCount":"Create Count in window", **rename_map})
            grp = grp.sort_values(by=f"MTD: {measures[0]}", ascending=False)
            tables[f"MTD split by {', '.join(split_dims)}"] = grp

        # Leaderboards
        if show_top_countries and "Country" in sub_mtd.columns:
            agg_dict = {f"__MTD__{m}":"sum" for m in measures}
            g2 = sub_mtd.groupby("Country", dropna=False).agg(agg_dict).reset_index()
            g2 = g2.rename(columns={f"__MTD__{m}": f"MTD: {m}" for m in measures})
            g2 = g2.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)
            tables["Top 5 Countries — MTD"] = g2

        if show_top_sources and "JetLearn Deal Source" in sub_mtd.columns:
            agg_dict = {f"__MTD__{m}":"sum" for m in measures}
            g3 = sub_mtd.groupby("JetLearn Deal Source", dropna=False).agg(agg_dict).reset_index()
            g3 = g3.rename(columns={f"__MTD__{m}": f"MTD: {m}" for m in measures})
            g3 = g3.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(3)
            tables["Top 3 Deal Sources — MTD"] = g3

        # Trend — one line per measure across Create_Month
        trend = sub_mtd.groupby("Create_Month")[[f"__MTD__{m}" for m in measures]].sum().reset_index()
        trend = trend.rename(columns={f"__MTD__{m}": f"{m}" for m in measures})
        trend["Create_Month"] = trend["Create_Month"].astype(str)
        long = trend.melt(id_vars="Create_Month", var_name="Measure", value_name="Count")
        charts["MTD Trend"] = altair_themed_line(long, "Create_Month:O", "Count:Q", color="Measure:N",
                                                 tooltip=["Create_Month","Measure","Count"])

    # ---------- Cohort ----------
    if cohort and coh_from and coh_to and len(measures)>0:
        tmp = base.copy()
        coh_flag_cols = []
        for m in measures:
            col_flag = f"__COH__{m}"
            tmp[col_flag] = tmp[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both").astype(int)
            coh_flag_cols.append(col_flag)
            metrics_rows.append({"Scope":"Cohort", "Metric":f"Count on '{m}'", "Window":f"{coh_from} → {coh_to}",
                                 "Value": int(tmp[col_flag].sum())})
        in_create_cohort = base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
        metrics_rows.append({"Scope":"Cohort", "Metric":"Create Count in Cohort window",
                             "Window":f"{coh_from} → {coh_to}", "Value": int(in_create_cohort.sum())})

        # Splits
        if split_dims:
            agg_dict = {flag: "sum" for flag in coh_flag_cols}
            agg_dict["_CreateInCohort"] = "sum"
            tmp["_CreateInCohort"] = in_create_cohort.astype(int)
            grp2 = tmp.groupby(split_dims, dropna=False).agg(agg_dict).reset_index()
            grp2 = grp2.rename(columns={"_CreateInCohort":"Create Count in Cohort window",
                                        **{f"__COH__{m}": f"Cohort: {m}" for m in measures}})
            grp2 = grp2.sort_values(by=f"Cohort: {measures[0]}", ascending=False)
            tables[f"Cohort split by {', '.join(split_dims)}"] = grp2

        # Leaderboards
        if show_top_countries and "Country" in base.columns:
            agg_dict = {f"__COH__{m}":"sum" for m in measures}
            g2 = tmp.groupby("Country", dropna=False).agg(agg_dict).reset_index()
            g2 = g2.rename(columns={f"__COH__{m}": f"Cohort: {m}" for m in measures})
            g2 = g2.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)
            tables["Top 5 Countries — Cohort"] = g2

        if show_top_sources and "JetLearn Deal Source" in base.columns:
            agg_dict = {f"__COH__{m}":"sum" for m in measures}
            g3 = tmp.groupby("JetLearn Deal Source", dropna=False).agg(agg_dict).reset_index()
            g3 = g3.rename(columns={f"__COH__{m}": f"Cohort: {m}" for m in measures})
            g3 = g3.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(3)
            tables["Top 3 Deal Sources — Cohort"] = g3

        # Trend — per measure by measure month
        trend_frames = []
        for m in measures:
            mask = base[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
            loc = base.loc[mask, [m]].copy()
            loc["Measure_Month"] = loc[m].dt.to_period("M").astype(str)
            t = loc.groupby("Measure_Month")["Measure_Month"].count().reset_index(name="Count")
            t["Measure"] = m
            trend_frames.append(t)
        if trend_frames:
            trend_coh = pd.concat(trend_frames, ignore_index=True)
            charts["Cohort Trend"] = altair_themed_line(trend_coh, "Measure_Month:O", "Count:Q", color="Measure:N",
                                                        tooltip=["Measure_Month","Measure","Count"])

    return metrics_rows, tables, charts

def kpi_grid(dfk, label_prefix=""):
    if dfk.empty:
        st.info("No KPIs yet.")
        return
    cols = st.columns(4)
    for i, row in dfk.iterrows():
        with cols[i % 4]:
            st.markdown(f"""
<div class="kpi">
  <div class="label">{label_prefix}{row['Scope']} — {row['Metric']}</div>
  <div class="value">{row['Value']:,}</div>
  <div class="delta">{row['Window']}</div>
</div>
""", unsafe_allow_html=True)

def build_compare_delta(dfA, dfB):
    if dfA.empty or dfB.empty:
        return pd.DataFrame()
    key = ["Scope","Metric"]
    a = dfA[key + ["Value"]].rename(columns={"Value":"A"})
    b = dfB[key + ["Value"]].rename(columns={"Value":"B"})
    out = pd.merge(a, b, on=key, how="inner")
    out["Δ"] = out["B"] - out["A"]
    out["Δ%"] = (out["Δ"] / out["A"].replace(0, pd.NA) * 100).round(1)
    return out

def mk_caption(meta):
    return (
        f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
        f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(meta['pipe_sel']) or 'None'} · "
        f"Deal Source: {'All' if meta['src_all'] else ', '.join(meta['src_sel']) or 'None'} · "
        f"Country: {'All' if meta['ctry_all'] else ', '.join(meta['ctry_sel']) or 'None'} · "
        f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(meta['cslr_sel']) or 'None'}"
    )

# ------------------------ Panels (A/B) ------------------------
# Compact mode stacks vertically on mobile
left, right = st.columns(2, gap="large")
with left:
    metaA = panel_controls("A", df, date_like_cols)
with (st if metaA["compact"] else right):
    metaB = panel_controls("B", df, date_like_cols)

# ------------------------ Actions / Reveal toggles ------------------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns([2,2,2,2])
show_kpis     = r1.toggle("Show KPIs", value=True)
show_splits   = r2.toggle("Show Splits & Leaderboards", value=False)
show_trends   = r3.toggle("Show Trends", value=False)
show_compare  = r4.toggle("Show Smart Compare", value=True)

# ------------------------ Compute ------------------------
with st.spinner("Crunching numbers…"):
    metricsA, tablesA, chartsA = compute_outputs(metaA)
    metricsB, tablesB, chartsB = compute_outputs(metaB)

# ------------------------ KPIs ------------------------
if show_kpis:
    st.markdown("### 📌 KPI Overview")
    kc1, kc2 = st.columns(2)
    with kc1:
        st.markdown("**Scenario A**")
        dfA = pd.DataFrame(metricsA)
        kpi_grid(dfA, "A · ")
    with kc2:
        st.markdown("**Scenario B**")
        dfB = pd.DataFrame(metricsB)
        kpi_grid(dfB, "B · ")

# ------------------------ Splits & Leaderboards ------------------------
if show_splits:
    st.markdown("### 🧩 Splits & Leaderboards")
    tabA, tabB = st.tabs(["Scenario A", "Scenario B"])
    with tabA:
        if not tablesA:
            st.info("No split/leaderboard tables — toggle Splits & Leaderboards in Scenario A and run again.")
        else:
            for name, frame in tablesA.items():
                st.subheader("A · " + name)
                st.dataframe(frame, use_container_width=True)
                st.download_button("Download CSV (A · " + name + ")", to_csv_bytes(frame),
                                   file_name=f"A_{name.replace(' ','_')}.csv", mime="text/csv")
    with tabB:
        if not tablesB:
            st.info("No split/leaderboard tables — toggle Splits & Leaderboards in Scenario B and run again.")
        else:
            for name, frame in tablesB.items():
                st.subheader("B · " + name)
                st.dataframe(frame, use_container_width=True)
                st.download_button("Download CSV (B · " + name + ")", to_csv_bytes(frame),
                                   file_name=f"B_{name.replace(' ','_')}.csv", mime="text/csv")

# ------------------------ Trends ------------------------
if show_trends:
    st.markdown("### 📈 Trends")
    t1, t2 = st.columns(2)
    with t1:
        if "MTD Trend" in chartsA or "Cohort Trend" in chartsA:
            st.markdown("**Scenario A**")
            if "MTD Trend" in chartsA: st.altair_chart(chartsA["MTD Trend"], use_container_width=True)
            if "Cohort Trend" in chartsA: st.altair_chart(chartsA["Cohort Trend"], use_container_width=True)
        else:
            st.info("Enable MTD/Cohort in A and set ranges.")
    with t2:
        if "MTD Trend" in chartsB or "Cohort Trend" in chartsB:
            st.markdown("**Scenario B**")
            if "MTD Trend" in chartsB: st.altair_chart(chartsB["MTD Trend"], use_container_width=True)
            if "Cohort Trend" in chartsB: st.altair_chart(chartsB["Cohort Trend"], use_container_width=True)
        else:
            st.info("Enable MTD/Cohort in B and set ranges.")

# ------------------------ Smart Compare ------------------------
if show_compare:
    st.markdown("### 🧠 Smart Compare (A vs B)")
    if 'dfA' in locals() and 'dfB' in locals() and not pd.DataFrame(metricsA).empty and not pd.DataFrame(metricsB).empty:
        cmp = build_compare_delta(pd.DataFrame(metricsA), pd.DataFrame(metricsB))
        if cmp.empty:
            st.info("Adjust scenarios to produce comparable KPIs.")
        else:
            st.dataframe(cmp, use_container_width=True)
            try:
                # Quick small-multiples when both scenarios use same measures
                sameA = [m for m in metaA["measures"]]
                sameB = [m for m in metaB["measures"]]
                if set(sameA) == set(sameB):
                    sub = cmp[cmp["Metric"].str.startswith("Count on '")].copy()
                    if not sub.empty:
                        sub["Measure"] = sub["Metric"].str.extract(r"Count on '(.+)'")
                        a_long = sub.rename(columns={"A":"Value"})[["Measure","Scope","Value"]]; a_long["Scenario"] = "A"
                        b_long = sub.rename(columns={"B":"Value"})[["Measure","Scope","Value"]]; b_long["Scenario"] = "B"
                        long = pd.concat([a_long, b_long], ignore_index=True)
                        ch = alt.Chart(long).mark_bar().encode(
                            x=alt.X("Scope:N", title=None),
                            y=alt.Y("Value:Q"),
                            color=alt.Color("Scenario:N", scale=alt.Scale(range=PALETTE[:2])),
                            column=alt.Column("Measure:N", header=alt.Header(title=None, labelAngle=0)),
                            tooltip=["Measure","Scenario","Scope","Value"]
                        ).properties(height=260)
                        st.altair_chart(ch, use_container_width=True)
            except Exception:
                pass
    else:
        st.info("Turn on KPIs for both scenarios to enable compare.")

# ------------------------ Foot captions ------------------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.caption("**Scenario A** — " + mk_caption(metaA))
st.caption("**Scenario B** — " + mk_caption(metaB))
st.caption("Excluded globally: 1.2 Invalid Deal")
