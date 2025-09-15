# app_compare_smart.py — Multi-Measure MTD/Cohort (columns grow as you select more measures)
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

st.set_page_config(page_title="MTD vs Cohort — A/B Compare (Smart UI)", layout="wide", page_icon="✨")
st.markdown("""
<style>
.block-container {padding-top: 0.6rem; padding-bottom: 0.6rem;}
:root { --muted: #6b7280; --card: #ffffff; --border: rgba(49,51,63,.15); }
hr.soft { border:0; height:1px; background:var(--border); margin: 0.6rem 0 1rem; }
.sticky-header {position: sticky; top: 0; background: white; z-index: 99; padding: 8px 0 6px; border-bottom: 1px solid var(--border);}
.kpi {padding:8px 10px; border:1px solid var(--border); border-radius:12px; background:var(--card);}
.kpi .label {color:var(--muted); font-size:.8rem; margin-bottom:4px;}
.kpi .value {font-size:1.4rem; font-weight:700; line-height:1.1;}
.kpi .delta {font-size:.9rem; color:#2563eb;}
.badge {display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); color:#111827; background:#f9fafb;}
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = [
    "Pipeline","JetLearn Deal Source","Country",
    "Student/Academic Counsellor","Deal Stage","Create Date",
]

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

@st.cache_data(show_spinner=False)
def load_and_prepare(data_bytes, path_text):
    # load
    if data_bytes:
        df = robust_read_csv(BytesIO(data_bytes))
    elif path_text:
        df = robust_read_csv(path_text)
    else:
        raise ValueError("Please upload a CSV or provide a file path.")
    df.columns = [c.strip() for c in df.columns]

    # required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

    # exclude invalid deals
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

    # parse core dates
    df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
    df["Create_Month"] = df["Create Date"].dt.to_period("M")

    # detect other date-like columns
    date_like_cols = detect_measure_date_columns(df)

    return df, date_like_cols

# ---------- Header with Clone buttons ----------
with st.container():
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    hc1, hc2 = st.columns([6,4])
    with hc1:
        st.markdown("### ✨ MTD vs Cohort — A/B Compare (Smart)")
        st.caption("Minimal, interactive analysis with independent A/B filters, presets, multi-measure splits & smart compare.")
    with hc2:
        st.write(""); st.write("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clone A → B"):
                for k in list(st.session_state.keys()):
                    if str(k).startswith("A_"):
                        st.session_state[str(k).replace("A_","B_",1)] = st.session_state[k]
                st.rerun()
        with col2:
            if st.button("Clone B → A"):
                for k in list(st.session_state.keys()):
                    if str(k).startswith("B_"):
                        st.session_state[str(k).replace("B_","A_",1)] = st.session_state[k]
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Data source ----------
ds = st.expander("Data source", expanded=True)
with ds:
    c1, c2 = st.columns([3,2])
    with c1:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with c2:
        default_path = st.text_input("…or CSV path", value="Master_sheet_DB_10percent.csv")
    try:
        df, date_like_cols = load_and_prepare(uploaded.getvalue() if uploaded else None,
                                              default_path if not uploaded else None)
        st.success("Data loaded", icon="✅")
    except Exception as e:
        st.error(str(e))
        raise

if not date_like_cols:
    st.error("No date-like columns found besides 'Create Date' (e.g., 'Payment Received Date').")
    raise SystemExit

# ---------- UI helpers ----------
def filter_block(df, label, colname, key_prefix):
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])
    c1, c2 = st.columns([1,3])
    all_flag = c1.checkbox("All", value=True, key=f"{key_prefix}_all")
    selected = c2.multiselect(label, options, default=options, disabled=all_flag, key=f"{key_prefix}_sel")
    return all_flag, selected

def date_preset_row(name, base_series, measure_series, key_prefix, is_mtd=True):
    presets = ["This month","Last month","Last 7 days","All time","Custom"]
    preset = st.radio(f"[{name}] Range preset", presets, horizontal=True, key=f"{key_prefix}_preset")
    dmin, dmax = safe_minmax_date(base_series if is_mtd else measure_series)
    if preset == "This month":
        dt_from, dt_to = this_month_bounds()
    elif preset == "Last month":
        dt_from, dt_to = last_month_bounds()
    elif preset == "Last 7 days":
        dt_from, dt_to = last_7_days_bounds()
    elif preset == "All time":
        dt_from, dt_to = dmin, dmax
    else:
        dt_from, dt_to = dmin, dmax
    c1, c2 = st.columns(2)
    dt_from = c1.date_input(f"[{name}] From", dt_from, key=f"{key_prefix}_from")
    dt_to   = c2.date_input(f"[{name}] To",   dt_to,   key=f"{key_prefix}_to")
    if dt_from > dt_to:
        st.error(f"{name}: 'From' is after 'To'.")
        return None, None
    return dt_from, dt_to

def ensure_month_cols(base: pd.DataFrame, measures):
    for m in measures:
        col = f"{m}_Month"
        if col not in base.columns:
            base[col] = base[m].dt.to_period("M")
    return base

def panel_controls(name: str, df: pd.DataFrame, date_like_cols):
    st.markdown(f"#### Scenario {name} <span class='badge'>independent</span>", unsafe_allow_html=True)
    with st.container():
        # Global filters
        with st.expander(f"[{name}] Global filters", expanded=True):
            f1c1, f1c2 = st.columns(2)
            with f1c1:
                pipe_all, pipe_sel = filter_block(df, "Pipeline", "Pipeline", f"{name}_pipe")
                src_all,  src_sel  = filter_block(df, "Deal Source", "JetLearn Deal Source", f"{name}_src")
            with f1c2:
                ctry_all, ctry_sel = filter_block(df, "Country", "Country", f"{name}_ctry")
                cslr_all, cslr_sel = filter_block(df, "Counsellor", "Student/Academic Counsellor", f"{name}_cslr")

        mask_cat = (
            in_filter(df["Pipeline"], pipe_all, pipe_sel) &
            in_filter(df["JetLearn Deal Source"], src_all, src_sel) &
            in_filter(df["Country"], ctry_all, ctry_sel) &
            in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
        )
        base = df[mask_cat].copy()

        # Measure(s) + toggles
        st.markdown("##### Measures & windows")
        mc1, mc2, mc3 = st.columns([3,2,2])
        measures = mc1.multiselect(f"[{name}] Measure date(s) — counts per selected field",
                                   options=date_like_cols,
                                   default=[date_like_cols[0]] if date_like_cols else [],
                                   key=f"{name}_measures")
        mtd    = mc2.toggle(f"[{name}] MTD", value=True, key=f"{name}_mtd")
        cohort = mc3.toggle(f"[{name}] Cohort", value=True, key=f"{name}_coh")

        if not measures:
            st.warning("Pick at least one Measure date.")
            measures = []

        base = ensure_month_cols(base, measures)

        # Ranges
        mtd_from = mtd_to = coh_from = coh_to = None
        # for presets, use first selected measure for defaults; if none, fallback to Create Date
        first_measure_series = base[measures[0]] if measures else base["Create Date"]
        if mtd:
            st.caption("Create-Date window")
            mtd_from, mtd_to = date_preset_row(name, base["Create Date"], first_measure_series, f"{name}_mtd", True)
        if cohort:
            st.caption("Measure-Date window")
            coh_from, coh_to = date_preset_row(name, base["Create Date"], first_measure_series, f"{name}_coh", False)

        # Split options (categorical only)
        with st.expander(f"[{name}] Splits & leaderboards", expanded=False):
            bc1, bc2, bc3 = st.columns([3,2,2])
            split_dims = bc1.multiselect(
                f"[{name}] Split by (pick one or more)",
                ["JetLearn Deal Source", "Country"],
                default=[],
                key=f"{name}_split"
            )
            show_top_countries = bc2.checkbox(f"[{name}] Top 5 Countries", value=True, key=f"{name}_top_ctry")
            show_top_sources   = bc3.checkbox(f"[{name}] Top 3 Deal Sources", value=True, key=f"{name}_top_src")
            show_combo_pairs   = st.checkbox(f"[{name}] Country × Deal Source (Top 10)", value=False, key=f"{name}_pair")

        return dict(
            name=name, base=base, measures=measures, mtd=mtd, cohort=cohort,
            mtd_from=mtd_from, mtd_to=mtd_to, coh_from=coh_from, coh_to=coh_to,
            split_dims=split_dims, show_top_countries=show_top_countries,
            show_top_sources=show_top_sources, show_combo_pairs=show_combo_pairs,
            pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
            ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel
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

        # Build flags per measure
        mtd_flag_cols = []
        for m in measures:
            col_flag = f"__MTD__{m}"
            sub_mtd[col_flag] = ((sub_mtd[m].notna()) & (sub_mtd[f"{m}_Month"] == sub_mtd["Create_Month"])).astype(int)
            mtd_flag_cols.append(col_flag)
            # KPI per measure
            metrics_rows.append({"Scope":"MTD", "Metric":f"Count on '{m}'", "Window":f"{mtd_from} → {mtd_to}",
                                 "Value": int(sub_mtd[col_flag].sum())})

        # Create count in window (once)
        metrics_rows.append({"Scope":"MTD", "Metric":"Create Count in window", "Window":f"{mtd_from} → {mtd_to}",
                             "Value": int(len(sub_mtd))})

        # Split table (one table with columns per measure)
        if split_dims:
            agg_dict = {flag: "sum" for flag in mtd_flag_cols}
            agg_dict["_CreateCount"] = "sum"
            sub_mtd["_CreateCount"] = 1
            grp = sub_mtd.groupby(split_dims, dropna=False).agg(agg_dict).reset_index()
            # Rename measure columns to friendly names
            rename_map = {f"__MTD__{m}": f"MTD: {m}" for m in measures}
            grp = grp.rename(columns={"_CreateCount":"Create Count in window", **rename_map})
            # Sort by the first measure
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

        if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(sub_mtd.columns):
            agg_dict = {f"__MTD__{m}":"sum" for m in measures}
            g4 = sub_mtd.groupby(["Country","JetLearn Deal Source"], dropna=False).agg(agg_dict).reset_index()
            g4 = g4.rename(columns={f"__MTD__{m}": f"MTD: {m}" for m in measures})
            g4 = g4.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(10)
            tables["Top Country × Deal Source — MTD"] = g4

        # Trend (MTD) across Create_Month — one line per measure
        trend = sub_mtd.groupby("Create_Month")[mtd_flag_cols].sum().reset_index()
        trend = trend.rename(columns={f"__MTD__{m}": f"{m}" for m in measures})
        trend["Create_Month"] = trend["Create_Month"].astype(str)
        long = trend.melt(id_vars="Create_Month", var_name="Measure", value_name="Count")
        charts["MTD Trend"] = alt.Chart(long).mark_line(point=True).encode(
            x="Create_Month:O", y="Count:Q", color="Measure:N",
            tooltip=["Create_Month","Measure","Count"]
        ).properties(height=260)

    # ---------- Cohort ----------
    if cohort and coh_from and coh_to and len(measures)>0:
        # Build per-measure inclusion flags
        coh_flag_cols = []
        tmp = base.copy()
        for m in measures:
            col_flag = f"__COH__{m}"
            tmp[col_flag] = tmp[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both").astype(int)
            coh_flag_cols.append(col_flag)
            metrics_rows.append({"Scope":"Cohort", "Metric":f"Count on '{m}'", "Window":f"{coh_from} → {coh_to}",
                                 "Value": int(tmp[col_flag].sum())})

        # Create count in cohort window (Create Date within range)
        in_create_cohort = base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
        metrics_rows.append({"Scope":"Cohort", "Metric":"Create Count in Cohort window",
                             "Window":f"{coh_from} → {coh_to}", "Value": int(in_create_cohort.sum())})

        # Split table (columns per measure)
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

        # Trend (Cohort) — by each measure's month; one line per measure
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
            charts["Cohort Trend"] = alt.Chart(trend_coh).mark_line(point=True).encode(
                x="Measure_Month:O", y="Count:Q", color="Measure:N",
                tooltip=["Measure_Month","Measure","Count"]
            ).properties(height=260)

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

# ---------- Build panels ----------
cA, cB = st.columns(2, gap="large")
with cA:
    metaA = panel_controls("A", df, date_like_cols)
with cB:
    metaB = panel_controls("B", df, date_like_cols)

# ---------- Compute ----------
with st.spinner("Calculating results..."):
    metricsA, tablesA, chartsA = compute_outputs(metaA)
    metricsB, tablesB, chartsB = compute_outputs(metaB)

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------- KPIs ----------
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

# ---------- Smart compare ----------
if 'dfA' in locals() and 'dfB' in locals() and not dfA.empty and not dfB.empty:
    st.markdown("### 🧠 Smart Compare (A vs B)")
    cmp = build_compare_delta(dfA, dfB)
    if cmp.empty:
        st.info("Adjust scenarios to produce comparable KPIs.")
    else:
        st.dataframe(cmp, use_container_width=True)
        # Optional quick viz when both scenarios share the same set of measures:
        # Show bars for each MTD/Cohort metric per measure
        try:
            same_measures = set(metaA["measures"]) == set(metaB["measures"])
            if same_measures:
                # pick just "Count on '...'" rows
                sub = cmp[cmp["Metric"].str.startswith("Count on '")].copy()
                if not sub.empty:
                    # Extract measure name inside quotes to a column
                    sub["Measure"] = sub["Metric"].str.extract(r"Count on '(.+)'")
                    a_long = sub.rename(columns={"A":"Value"})[["Measure","Scope","Value"]]; a_long["Scenario"] = "A"
                    b_long = sub.rename(columns={"B":"Value"})[["Measure","Scope","Value"]]; b_long["Scenario"] = "B"
                    long = pd.concat([a_long, b_long], ignore_index=True)
                    chart = alt.Chart(long).mark_bar().encode(
                        x=alt.X("Scope:N", title=None),
                        y=alt.Y("Value:Q"),
                        color=alt.Color("Scenario:N"),
                        column=alt.Column("Measure:N", header=alt.Header(title=None, labelAngle=0)),
                        tooltip=["Measure","Scenario","Scope","Value"]
                    ).properties(height=260)
                    st.altair_chart(chart, use_container_width=True)
        except Exception:
            pass
else:
    st.info("Tip: turn on MTD and/or Cohort in both scenarios to enable Smart Compare.")

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------- Details ----------
tabA, tabB = st.tabs(["📋 Scenario A Details", "📋 Scenario B Details"])
with tabA:
    if not tablesA and not chartsA:
        st.info("No details for Scenario A — adjust filters.")
    else:
        for name, frame in tablesA.items():
            st.subheader(name + " (A)")
            st.dataframe(frame, use_container_width=True)
        if "MTD Trend" in chartsA:
            st.subheader("MTD Trend (A)")
            st.altair_chart(chartsA["MTD Trend"], use_container_width=True)
        if "Cohort Trend" in chartsA:
            st.subheader("Cohort Trend (A)")
            st.altair_chart(chartsA["Cohort Trend"], use_container_width=True)

with tabB:
    if not tablesB and not chartsB:
        st.info("No details for Scenario B — adjust filters.")
    else:
        for name, frame in tablesB.items():
            st.subheader(name + " (B)")
            st.dataframe(frame, use_container_width=True)
        if "MTD Trend" in chartsB:
            st.subheader("MTD Trend (B)")
            st.altair_chart(chartsB["MTD Trend"], use_container_width=True)
        if "Cohort Trend" in chartsB:
            st.subheader("Cohort Trend (B)")
            st.altair_chart(chartsB["Cohort Trend"], use_container_width=True)

def mk_caption(meta):
    return (
        f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
        f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(meta['pipe_sel']) or 'None'} · "
        f"Deal Source: {'All' if meta['src_all'] else ', '.join(meta['src_sel']) or 'None'} · "
        f"Country: {'All' if meta['ctry_all'] else ', '.join(meta['ctry_sel']) or 'None'} · "
        f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(meta['cslr_sel']) or 'None'}"
    )

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.caption("**Scenario A** — " + mk_caption(metaA))
st.caption("**Scenario B** — " + mk_caption(metaB))
st.caption("Excluded globally: 1.2 Invalid Deal")
