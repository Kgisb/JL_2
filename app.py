# app_compare.py — MTD vs Cohort Analyzer with A/B filter sets
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ---------------- Page config & styling ----------------
st.set_page_config(page_title="MTD vs Cohort Analyzer — A/B Compare", layout="wide", page_icon="🧪")
st.markdown("""
<style>
.block-container {padding-top: 0.7rem; padding-bottom: 1rem;}
.kpi-card {
  padding: 14px 16px; border:1px solid rgba(49,51,63,.2);
  border-radius: 12px; background:linear-gradient(180deg,#fff 0%,#fafafa 100%);
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.kpi-title { font-size:.9rem; color:#6b7280; }
.kpi-value { font-size:1.6rem; font-weight:700; margin:2px 0; }
.kpi-sub   { color:#6b7280; font-size:.85rem; margin-top:-6px;}
hr.soft { border:0; height:1px; background:rgba(49,51,63,.1); margin:0.6rem 0 1rem;}
.badge {display:inline-block; padding:2px 8px; font-size:.75rem; border-radius:999px; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe;}
.panel {border:1px solid rgba(49,51,63,.15); border-radius:12px; padding:10px 12px; background:#fff;}
</style>
""", unsafe_allow_html=True)

# ---------------- Utilities ----------------
REQUIRED_COLS = [
    "Pipeline","JetLearn Deal Source","Country",
    "Student/Academic Counsellor","Deal Stage","Create Date",
]

def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try: return pd.read_csv(file_or_path, encoding=enc)
        except Exception: pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    date_like = []
    for col in df.columns:
        if col == "Create Date": continue
        if any(k in col.lower() for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed; date_like.append(col)
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c!="Payment Received Date"]
    return date_like

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    if all_checked: return pd.Series(True, index=series.index)
    uniq = series.dropna().astype(str).nunique()
    if selected_values and len(selected_values)==uniq: return pd.Series(True, index=series.index)
    if not selected_values: return pd.Series(False, index=series.index)
    return series.astype(str).isin(selected_values)

def safe_minmax_date(s: pd.Series, fallback=(date(2020,1,1), date.today())):
    if s.isna().all(): return fallback
    return (pd.to_datetime(s.min()).date(), pd.to_datetime(s.max()).date())

def this_month_bounds():
    today = pd.Timestamp.today().date()
    start = today.replace(day=1)
    next_start = (start.replace(year=start.year+1, month=1) if start.month==12
                  else start.replace(month=start.month+1))
    return start, (next_start - timedelta(days=1))

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------- Data loader ----------------
@st.cache_data(show_spinner=False)
def load_and_prepare(data_bytes, path_text):
    if data_bytes:
        df = robust_read_csv(BytesIO(data_bytes))
    elif path_text:
        df = robust_read_csv(path_text)
    else:
        raise ValueError("Please upload a CSV or provide a file path.")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

    # global exclusion
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

    # parse dates
    df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
    df["Create_Month"] = df["Create Date"].dt.to_period("M")
    date_like_cols = detect_measure_date_columns(df)
    return df, date_like_cols

# ---------------- Controls: Data ----------------
st.sidebar.title("⚙️ Controls")

with st.sidebar.expander("Data source", expanded=True):
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    default_path = st.text_input("...or enter CSV path", value="Master_sheet_DB_10percent.csv")
    try:
        df, date_like_cols = load_and_prepare(uploaded.getvalue() if uploaded else None,
                                              default_path if not uploaded else None)
        st.success("Data loaded.", icon="✅")
    except Exception as e:
        st.error(str(e)); st.stop()

if not date_like_cols:
    st.error("No other date-like columns found (e.g., 'Payment Received Date').")
    st.stop()

# ---------------- Panel builder (Scenario A / B) ----------------
def filter_block(df, label, colname, key_prefix):
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])
    c1, c2 = st.columns([1, 3])
    all_flag = c1.checkbox("All", value=True, key=f"{key_prefix}_all")
    selected = c2.multiselect(label, options, default=options, disabled=all_flag,
                              key=f"{key_prefix}_sel", help=f"Filter by {label.lower()}")
    return all_flag, selected

def panel_controls(name: str, df: pd.DataFrame, date_like_cols):
    st.markdown(f"### Scenario {name} <span class='badge'>independent filters</span>", unsafe_allow_html=True)
    with st.container():
        # Global filters
        st.markdown("#### Global Filters")
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

        # Measure & toggles
        st.markdown("#### Measure & Windows")
        mc1, mc2, mc3 = st.columns([3,2,2])
        measure_col = mc1.selectbox(f"[{name}] Measure date", date_like_cols, index=0, key=f"{name}_measure")
        if f"{measure_col}_Month" not in base.columns:
            base[f"{measure_col}_Month"] = base[measure_col].dt.to_period("M")
        mtd    = mc2.toggle(f"[{name}] Enable MTD", value=True, key=f"{name}_mtd")
        cohort = mc3.toggle(f"[{name}] Enable Cohort", value=True, key=f"{name}_coh")

        # MTD window (Create)
        if mtd:
            st.markdown("##### 🗓️ MTD window (Create Date)")
            p1, p2, p3 = st.columns([2,3,3])
            preset = p1.selectbox(f"[{name}] Preset", ["This month","All time","Custom"], index=0, key=f"{name}_mtd_preset")
            cmin, cmax = safe_minmax_date(base["Create Date"])
            if preset == "This month":  mtd_from, mtd_to = this_month_bounds()
            elif preset == "All time":  mtd_from, mtd_to = cmin, cmax
            else:                       mtd_from, mtd_to = cmin, cmax
            mtd_from = p2.date_input(f"[{name}] From (Create)", mtd_from, key=f"{name}_mtd_from")
            mtd_to   = p3.date_input(f"[{name}] To (Create)",   mtd_to,   key=f"{name}_mtd_to")
            if mtd_from > mtd_to:
                st.error(f"{name} / MTD: 'From' is after 'To'."); mtd = False
        else:
            mtd_from = mtd_to = None

        # Cohort window (Measure)
        if cohort:
            st.markdown("##### 🗓️ Cohort window (Measure Date)")
            q1, q2, q3 = st.columns([2,3,3])
            preset2 = q1.selectbox(f"[{name}] Preset", ["This month","All time","Custom"], index=1, key=f"{name}_coh_preset")
            mmin, mmax = safe_minmax_date(base[measure_col])
            if preset2 == "This month": coh_from, coh_to = this_month_bounds()
            elif preset2 == "All time": coh_from, coh_to = mmin, mmax
            else:                       coh_from, coh_to = mmin, mmax
            coh_from = q2.date_input(f"[{name}] From (Measure)", coh_from, key=f"{name}_coh_from")
            coh_to   = q3.date_input(f"[{name}] To (Measure)",   coh_to,   key=f"{name}_coh_to")
            if coh_from > coh_to:
                st.error(f"{name} / Cohort: 'From' is after 'To'."); cohort = False
        else:
            coh_from = coh_to = None

        # Splits & leaderboards
        st.markdown("#### Breakdowns & Leaderboards")
        bc1, bc2, bc3, bc4 = st.columns([3,2,2,2])
        split_dims = bc1.multiselect(f"[{name}] Split by", ["JetLearn Deal Source","Country"], default=[], key=f"{name}_split")
        show_top_countries = bc2.checkbox(f"[{name}] Top 5 Countries", value=True, key=f"{name}_top_ctry")
        show_top_sources   = bc3.checkbox(f"[{name}] Top 3 Deal Sources", value=True, key=f"{name}_top_src")
        show_combo_pairs   = bc4.checkbox(f"[{name}] Country × Deal Source (Top 10)", value=False, key=f"{name}_pair")

        meta = dict(
            name=name, base=base, measure_col=measure_col, mtd=mtd, cohort=cohort,
            mtd_from=mtd_from, mtd_to=mtd_to, coh_from=coh_from, coh_to=coh_to,
            split_dims=split_dims, show_top_countries=show_top_countries,
            show_top_sources=show_top_sources, show_combo_pairs=show_combo_pairs,
            pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
            ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel
        )
        return meta

def compute_outputs(meta):
    base = meta["base"]; measure_col = meta["measure_col"]
    mtd = meta["mtd"]; cohort = meta["cohort"]
    mtd_from, mtd_to = meta["mtd_from"], meta["mtd_to"]
    coh_from, coh_to = meta["coh_from"], meta["coh_to"]
    split_dims = meta["split_dims"]
    show_top_countries = meta["show_top_countries"]
    show_top_sources   = meta["show_top_sources"]
    show_combo_pairs   = meta["show_combo_pairs"]

    if f"{measure_col}_Month" not in base.columns:
        base[f"{measure_col}_Month"] = base[measure_col].dt.to_period("M")

    metrics_rows = []
    tables = {}
    charts = {}

    # MTD
    if mtd and mtd_from and mtd_to:
        in_create_window = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
        sub_mtd = base[in_create_window].copy()
        mtd_flag = (sub_mtd[measure_col].notna()) & (sub_mtd[f"{measure_col}_Month"] == sub_mtd["Create_Month"])

        metrics_rows += [
            {"Scope":"MTD","Metric":f"Count on '{measure_col}'","Window":f"{mtd_from} → {mtd_to}","Value":int(mtd_flag.sum())},
            {"Scope":"MTD","Metric":"Create Count in window","Window":f"{mtd_from} → {mtd_to}","Value":int(len(sub_mtd))},
        ]

        if split_dims:
            g = sub_mtd.copy()
            g["_MTD Count"] = mtd_flag.astype(int)
            g["_Create Count in window"] = 1
            grp = g.groupby(split_dims, dropna=False).agg({
                "_Create Count in window":"sum",
                "_MTD Count":"sum"
            }).reset_index().rename(columns={
                "_Create Count in window":"Create Count in window",
                "_MTD Count":f"MTD Count on '{measure_col}'"
            }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False)
            tables[f"MTD split by {', '.join(split_dims)}"] = grp

            if len(split_dims)==1:
                dim = split_dims[0]
                charts["MTD Split"] = alt.Chart(grp).mark_bar().encode(
                    x=alt.X(dim, sort='-y'),
                    y=alt.Y(f"MTD Count on '{measure_col}'", title="MTD Count"),
                    tooltip=[dim, f"MTD Count on '{measure_col}'","Create Count in window"]
                ).properties(height=320)
            elif len(split_dims)==2:
                charts["MTD Split Heatmap"] = alt.Chart(grp).mark_rect().encode(
                    x=alt.X(split_dims[0], title=split_dims[0]),
                    y=alt.Y(split_dims[1], title=split_dims[1]),
                    color=alt.Color(f"MTD Count on '{measure_col}'", title="MTD Count"),
                    tooltip=split_dims + [f"MTD Count on '{measure_col}'","Create Count in window"]
                ).properties(height=360)

        if show_top_countries and "Country" in sub_mtd.columns:
            g2 = sub_mtd.copy(); g2["_MTD Count"] = mtd_flag.astype(int); g2["_Create Count in window"] = 1
            top_ctry = g2.groupby("Country", dropna=False).agg({
                "_Create Count in window":"sum","_MTD Count":"sum"}).reset_index().rename(columns={
                "_Create Count in window":"Create Count in window","_MTD Count":f"MTD Count on '{measure_col}'"
            }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False).head(5)
            tables["Top 5 Countries — MTD"] = top_ctry

        if show_top_sources and "JetLearn Deal Source" in sub_mtd.columns:
            g3 = sub_mtd.copy(); g3["_MTD Count"] = mtd_flag.astype(int); g3["_Create Count in window"] = 1
            top_src = g3.groupby("JetLearn Deal Source", dropna=False).agg({
                "_Create Count in window":"sum","_MTD Count":"sum"}).reset_index().rename(columns={
                "_Create Count in window":"Create Count in window","_MTD Count":f"MTD Count on '{measure_col}'"
            }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False).head(3)
            tables["Top 3 Deal Sources — MTD"] = top_src

        if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(sub_mtd.columns):
            g4 = sub_mtd.copy(); g4["_MTD Count"] = mtd_flag.astype(int); g4["_Create Count in window"] = 1
            both = g4.groupby(["Country","JetLearn Deal Source"], dropna=False).agg({
                "_Create Count in window":"sum","_MTD Count":"sum"}).reset_index().rename(columns={
                "_Create Count in window":"Create Count in window","_MTD Count":f"MTD Count on '{measure_col}'"
            }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False).head(10)
            tables["Top Country × Deal Source — MTD"] = both

        trend_mtd = sub_mtd.assign(flag=mtd_flag.astype(int)).groupby("Create_Month", dropna=False)["flag"].sum().reset_index()
        trend_mtd = trend_mtd.rename(columns={"Create_Month":"Month","flag":"MTD Count"})
        trend_mtd["Month"] = trend_mtd["Month"].astype(str)
        charts["MTD Trend"] = alt.Chart(trend_mtd).mark_line(point=True).encode(
            x="Month:O", y="MTD Count:Q", tooltip=["Month","MTD Count"]
        ).properties(height=280)
        tables["Trend — MTD (by Create Month)"] = trend_mtd

    # Cohort
    if cohort and coh_from and coh_to:
        in_measure_window = base[measure_col].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
        in_create_cohort  = base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")

        metrics_rows += [
            {"Scope":"Cohort","Metric":f"Count on '{measure_col}'","Window":f"{coh_from} → {coh_to}","Value":int(in_measure_window.sum())},
            {"Scope":"Cohort","Metric":"Create Count in Cohort window","Window":f"{coh_from} → {coh_to}","Value":int(in_create_cohort.sum())},
        ]

        if split_dims:
            g = base.copy()
            g["_Cohort Count"] = in_measure_window.astype(int)
            g["_Create Count in Cohort window"] = in_create_cohort.astype(int)
            grp2 = g.groupby(split_dims, dropna=False).agg({
                "_Cohort Count":"sum","_Create Count in Cohort window":"sum"
            }).reset_index().rename(columns={
                "_Cohort Count":f"Cohort Count on '{measure_col}'",
                "_Create Count in Cohort window":"Create Count in Cohort window"
            }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False)
            tables[f"Cohort split by {', '.join(split_dims)}"] = grp2

            if len(split_dims)==1:
                dim = split_dims[0]
                charts["Cohort Split"] = alt.Chart(grp2).mark_bar().encode(
                    x=alt.X(dim, sort='-y'),
                    y=alt.Y(f"Cohort Count on '{measure_col}'", title="Cohort Count"),
                    tooltip=[dim, f"Cohort Count on '{measure_col}'","Create Count in Cohort window"]
                ).properties(height=320)
            elif len(split_dims)==2:
                charts["Cohort Split Heatmap"] = alt.Chart(grp2).mark_rect().encode(
                    x=alt.X(split_dims[0], title=split_dims[0]),
                    y=alt.Y(split_dims[1], title=split_dims[1]),
                    color=alt.Color(f"Cohort Count on '{measure_col}'", title="Cohort Count"),
                    tooltip=split_dims + [f"Cohort Count on '{measure_col}'","Create Count in Cohort window"]
                ).properties(height=360)

        if show_top_countries and "Country" in base.columns:
            g2 = base.copy()
            g2["_Cohort Count"] = in_measure_window.astype(int)
            g2["_Create Count in Cohort window"] = in_create_cohort.astype(int)
            top_ctry2 = g2.groupby("Country", dropna=False).agg({
                "_Cohort Count":"sum","_Create Count in Cohort window":"sum"
            }).reset_index().rename(columns={
                "_Cohort Count":f"Cohort Count on '{measure_col}'",
                "_Create Count in Cohort window":"Create Count in Cohort window"
            }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False).head(5)
            tables["Top 5 Countries — Cohort"] = top_ctry2

        if show_top_sources and "JetLearn Deal Source" in base.columns:
            g3 = base.copy()
            g3["_Cohort Count"] = in_measure_window.astype(int)
            g3["_Create Count in Cohort window"] = in_create_cohort.astype(int)
            top_src2 = g3.groupby("JetLearn Deal Source", dropna=False).agg({
                "_Cohort Count":"sum","_Create Count in Cohort window":"sum"
            }).reset_index().rename(columns={
                "_Cohort Count":f"Cohort Count on '{measure_col}'",
                "_Create Count in Cohort window":"Create Count in Cohort window"
            }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False).head(3)
            tables["Top 3 Deal Sources — Cohort"] = top_src2

        if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(base.columns):
            g4 = base.copy()
            g4["_Cohort Count"] = in_measure_window.astype(int)
            g4["_Create Count in Cohort window"] = in_create_cohort.astype(int)
            both2 = g4.groupby(["Country","JetLearn Deal Source"], dropna=False).agg({
                "_Cohort Count":"sum","_Create Count in Cohort window":"sum"
            }).reset_index().rename(columns={
                "_Cohort Count":f"Cohort Count on '{measure_col}'",
                "_Create Count in Cohort window":"Create Count in Cohort window"
            }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False).head(10)
            tables["Top Country × Deal Source — Cohort"] = both2

        trend_coh = base.loc[in_measure_window].copy()
        trend_coh["Measure_Month"] = trend_coh[measure_col].dt.to_period("M")
        trend_coh = trend_coh.groupby("Measure_Month")["Measure_Month"].count().reset_index(name="Cohort Count")
        trend_coh["Measure_Month"] = trend_coh["Measure_Month"].astype(str)
        charts["Cohort Trend"] = alt.Chart(trend_coh).mark_line(point=True).encode(
            x="Measure_Month:O", y="Cohort Count:Q", tooltip=["Measure_Month","Cohort Count"]
        ).properties(height=280)
        tables["Trend — Cohort (by Measure Month)"] = trend_coh

    return metrics_rows, tables, charts

def kpi_cards(dfk):
    cols = st.columns(4)
    for i, row in dfk.iterrows():
        with cols[i % 4]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{row['Scope']} — {row['Metric']}</div>
                <div class="kpi-value">{row['Value']:,}</div>
                <div class="kpi-sub">{row['Window']}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- Header ----------------
lc, rc = st.columns([7,3])
with lc:
    st.markdown("## 🧪 A/B Compare — MTD vs Cohort Analyzer")
    st.caption("Two independent filter panels to compare outputs across **different time frames** (and segments). Excludes `1.2 Invalid Deal`.")
with rc:
    if st.button("Clone A → B", help="Copy Scenario A selections into Scenario B"):
        for k, v in st.session_state.items():
            if str(k).startswith("A_"):
                st.session_state[str(k).replace("A_","B_",1)] = v
        st.experimental_rerun()

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------------- Panels ----------------
colA, colB = st.columns(2, gap="large")
with colA:
    metaA = panel_controls("A", df, date_like_cols)
with colB:
    metaB = panel_controls("B", df, date_like_cols)

# ---------------- Compute ----------------
with st.spinner("Calculating A/B results..."):
    metricsA, tablesA, chartsA = compute_outputs(metaA)
    metricsB, tablesB, chartsB = compute_outputs(metaB)

# ---------------- KPIs side-by-side ----------------
st.markdown("### 📌 KPIs")
kc1, kc2 = st.columns(2)
with kc1:
    st.markdown("**Scenario A**")
    if metricsA:
        kpi_cards(pd.DataFrame(metricsA))
    else:
        st.info("No KPIs (turn on MTD and/or Cohort).")
with kc2:
    st.markdown("**Scenario B**")
    if metricsB:
        kpi_cards(pd.DataFrame(metricsB))
    else:
        st.info("No KPIs (turn on MTD and/or Cohort).")

# ---------------- Quick Comparison (if same measure) ----------------
if metricsA and metricsB and (metaA["measure_col"] == metaB["measure_col"]):
    st.markdown("### 🆚 Quick Compare")
    # Build a long table with Scope counts per scenario
    def to_long(metrics, tag):
        dfm = pd.DataFrame(metrics)
        dfm["Scenario"] = tag
        return dfm[["Scenario","Scope","Metric","Window","Value"]]
    long_cmp = pd.concat([to_long(metricsA, "A"), to_long(metricsB, "B")], ignore_index=True)
    # Only compare "Count on 'measure'" KPIs
    mask_main = long_cmp["Metric"].str.startswith("Count on '")
    main_cmp = long_cmp[mask_main].copy()
    if not main_cmp.empty:
        chart = alt.Chart(main_cmp).mark_bar().encode(
            x=alt.X("Scope:N", title=None),
            y=alt.Y("Value:Q"),
            color=alt.Color("Scenario:N"),
            column=alt.Column("Scenario:N", header=alt.Header(title=None, labelAngle=0)),
            tooltip=["Scenario","Scope","Value","Window"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No comparable 'Count on ...' KPIs to chart.")
else:
    st.info("Tip: choose the **same Measure date** in both scenarios to see a side-by-side chart.")

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------------- Tabs for detail ----------------
tabA, tabB, tabExp = st.tabs(["📋 Scenario A Tables/Trends", "📋 Scenario B Tables/Trends", "⬇️ Export"])

with tabA:
    if not tablesA and not chartsA:
        st.info("No tables yet — adjust Scenario A filters.")
    else:
        for name, frame in tablesA.items():
            if "Trend" in name: continue
            st.subheader(name + " (A)")
            st.dataframe(frame, use_container_width=True)
        # Trends
        if "MTD Trend" in chartsA:
            st.subheader("MTD Trend (A)")
            st.altair_chart(chartsA["MTD Trend"], use_container_width=True)
        if "Cohort Trend" in chartsA:
            st.subheader("Cohort Trend (A)")
            st.altair_chart(chartsA["Cohort Trend"], use_container_width=True)

with tabB:
    if not tablesB and not chartsB:
        st.info("No tables yet — adjust Scenario B filters.")
    else:
        for name, frame in tablesB.items():
            if "Trend" in name: continue
            st.subheader(name + " (B)")
            st.dataframe(frame, use_container_width=True)
        # Trends
        if "MTD Trend" in chartsB:
            st.subheader("MTD Trend (B)")
            st.altair_chart(chartsB["MTD Trend"], use_container_width=True)
        if "Cohort Trend" in chartsB:
            st.subheader("Cohort Trend (B)")
            st.altair_chart(chartsB["Cohort Trend"], use_container_width=True)

with tabExp:
    any_tbl = False
    if tablesA:
        st.markdown("#### Scenario A")
        for name, frame in tablesA.items():
            st.download_button(
                f"Download A — {name}.csv",
                data=to_csv_bytes(frame),
                file_name=f"A_{name.replace(' ','_').replace('×','x')}.csv",
                mime="text/csv",
                key=f"dl_A_{name}"
            )
            any_tbl = True
    if tablesB:
        st.markdown("#### Scenario B")
        for name, frame in tablesB.items():
            st.download_button(
                f"Download B — {name}.csv",
                data=to_csv_bytes(frame),
                file_name=f"B_{name.replace(' ','_').replace('×','x')}.csv",
                mime="text/csv",
                key=f"dl_B_{name}"
            )
            any_tbl = True
    if not any_tbl:
        st.info("No tables available to export.")

# ---------------- Footers ----------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
def mk_caption(meta):
    return (
        f"Measure: {meta['measure_col']} · "
        f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(meta['pipe_sel']) or 'None'} · "
        f"Deal Source: {'All' if meta['src_all'] else ', '.join(meta['src_sel']) or 'None'} · "
        f"Country: {'All' if meta['ctry_all'] else ', '.join(meta['ctry_sel']) or 'None'} · "
        f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(meta['cslr_sel']) or 'None'}"
    )
st.caption("**Scenario A** — " + mk_caption(metaA))
st.caption("**Scenario B** — " + mk_caption(metaB))
st.caption("Excluded globally: 1.2 Invalid Deal")
