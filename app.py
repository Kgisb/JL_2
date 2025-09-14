# app.py — Smart MTD vs Cohort Analyzer (Pro UI, nested f-strings fixed)
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ---------------- Page config & styling ----------------
st.set_page_config(page_title="MTD vs Cohort Analyzer", layout="wide", page_icon="📊")
st.markdown("""
<style>
.block-container {padding-top: 0.8rem; padding-bottom: 1rem;}
.kpi-card {
  padding: 14px 16px; border:1px solid rgba(49,51,63,.2);
  border-radius: 12px; background:linear-gradient(180deg,#fff 0%,#fafafa 100%);
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.kpi-title { font-size:.9rem; color:#6b7280; }
.kpi-value { font-size:1.6rem; font-weight:700; margin:2px 0; }
.kpi-sub   { color:#6b7280; font-size:.85rem; margin-top:-6px;}
hr.soft { border:0; height:1px; background:rgba(49,51,63,.1); margin:0.6rem 0 1rem;}
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

# ---------------- Sidebar: data + global filters ----------------
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

def filter_block(label, colname, key_prefix):
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])
    c1, c2 = st.sidebar.columns([1, 3])
    all_flag = c1.checkbox("All", value=True, key=f"{key_prefix}_all")
    selected = c2.multiselect(label, options, default=options, disabled=all_flag,
                              key=f"{key_prefix}_sel", help=f"Filter by {label.lower()}")
    return all_flag, selected

st.sidebar.markdown("### Global Filters")
pipe_all,  pipe_sel  = filter_block("Pipeline", "Pipeline", "pipe")
src_all,   src_sel   = filter_block("Deal Source", "JetLearn Deal Source", "src")
ctry_all,  ctry_sel  = filter_block("Country", "Country", "ctry")
cslr_all,  cslr_sel  = filter_block("Counsellor", "Student/Academic Counsellor", "cslr")

mask_cat = (
    in_filter(df["Pipeline"], pipe_all,  pipe_sel) &
    in_filter(df["JetLearn Deal Source"], src_all,   src_sel) &
    in_filter(df["Country"], ctry_all,  ctry_sel) &
    in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
)
base = df[mask_cat].copy()

# ---------------- Header ----------------
h1c1, h1c2 = st.columns([6,1])
with h1c1:
    st.markdown("## 📊 MTD vs Cohort Analyzer")
    st.caption("Smart analysis comparing **MTD** (Create-date window; measure-month == create-month) vs **Cohort** (Measure-date window).")
with h1c2:
    if st.button("Reset filters", help="Reselect 'All' for every global filter"):
        for k in list(st.session_state.keys()):
            if k.endswith("_all"): st.session_state[k] = True
        st.experimental_rerun()
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------------- Measure + toggles + date presets ----------------
if not date_like_cols:
    st.error("No other date-like columns found (e.g., 'Payment Received Date').")
    st.stop()

topc1, topc2, topc3 = st.columns([3,2,2])
measure_col = topc1.selectbox("Measure date (what to count on)", date_like_cols, index=0)
if f"{measure_col}_Month" not in base.columns:
    base[f"{measure_col}_Month"] = base[measure_col].dt.to_period("M")

mtd    = topc2.toggle("Enable MTD", value=True, help="Create-Date window; counts only if Measure-month == Create-month")
cohort = topc3.toggle("Enable Cohort", value=True, help="Measure-Date window; Create date ignored for the measure count")

# MTD range (Create Date)
if mtd:
    st.markdown("### 🗓️ MTD window (Create Date)")
    p1, p2, p3 = st.columns([2,3,3])
    preset = p1.selectbox("Preset", ["This month", "All time", "Custom"], index=0, key="mtd_preset")
    cmin, cmax = safe_minmax_date(base["Create Date"])
    if preset == "This month":  mtd_from, mtd_to = this_month_bounds()
    elif preset == "All time":  mtd_from, mtd_to = cmin, cmax
    else:                       mtd_from, mtd_to = cmin, cmax
    mtd_from = p2.date_input("From (Create)", mtd_from, key="mtd_from")
    mtd_to   = p3.date_input("To (Create)",   mtd_to,   key="mtd_to")
    if mtd_from > mtd_to:
        st.error("MTD: 'From' is after 'To'. Adjust the Create-Date range."); mtd = False

# Cohort range (Measure Date)
if cohort:
    st.markdown("### 🗓️ Cohort window (Measure Date)")
    q1, q2, q3 = st.columns([2,3,3])
    preset2 = q1.selectbox("Preset", ["This month", "All time", "Custom"], index=1, key="coh_preset")
    mmin, mmax = safe_minmax_date(base[measure_col])
    if preset2 == "This month": coh_from, coh_to = this_month_bounds()
    elif preset2 == "All time": coh_from, coh_to = mmin, mmax
    else:                       coh_from, coh_to = mmin, mmax
    coh_from = q2.date_input("From (Measure)", coh_from, key="coh_from")
    coh_to   = q3.date_input("To (Measure)",   coh_to,   key="coh_to")
    if coh_from > coh_to:
        st.error("Cohort: 'From' is after 'To'. Adjust the Measure-Date range."); cohort = False

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------------- Breakdowns & leaderboards ----------------
bc1, bc2, bc3, bc4 = st.columns([3,2,2,2])
split_dims = bc1.multiselect("Split by (optional)", ["JetLearn Deal Source","Country"], default=[],
                             help="Add per-group tables & charts.")
show_top_countries = bc2.checkbox("Top 5 Countries", value=True)
show_top_sources   = bc3.checkbox("Top 3 Deal Sources", value=True)
show_combo_pairs   = bc4.checkbox("Country × Deal Source (Top 10)", value=False)

st.caption("Tip: use the tabs below to view KPIs, tables, trends, and exports.")

# ---------------- Compute ----------------
metrics_rows = []
tables = {}   # name -> DataFrame
charts = {}   # name -> Altair chart

with st.spinner("Crunching numbers..."):
    # ----- MTD -----
    if mtd:
        in_create_window = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
        sub_mtd = base[in_create_window].copy()
        mtd_flag = (sub_mtd[measure_col].notna()) & (sub_mtd[f"{measure_col}_Month"] == sub_mtd["Create_Month"])

        metrics_rows += [
            {"Scope":"MTD", "Metric": f"Count on '{measure_col}'", "Window": f"{mtd_from} → {mtd_to}", "Value": int(mtd_flag.sum())},
            {"Scope":"MTD", "Metric": "Create Count in window",    "Window": f"{mtd_from} → {mtd_to}", "Value": int(len(sub_mtd))}
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

        # Trend (MTD)
        trend_mtd = sub_mtd.assign(flag=mtd_flag.astype(int)).groupby("Create_Month", dropna=False)["flag"].sum().reset_index()
        trend_mtd = trend_mtd.rename(columns={"Create_Month":"Month","flag":"MTD Count"})
        trend_mtd["Month"] = trend_mtd["Month"].astype(str)
        charts["MTD Trend"] = alt.Chart(trend_mtd).mark_line(point=True).encode(
            x="Month:O", y="MTD Count:Q", tooltip=["Month","MTD Count"]
        ).properties(height=280)
        tables["Trend — MTD (by Create Month)"] = trend_mtd

    # ----- Cohort -----
    if cohort:
        in_measure_window = base[measure_col].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
        in_create_cohort  = base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")

        metrics_rows += [
            {"Scope":"Cohort", "Metric": f"Count on '{measure_col}'", "Window": f"{coh_from} → {coh_to}", "Value": int(in_measure_window.sum())},
            {"Scope":"Cohort", "Metric": "Create Count in Cohort window", "Window": f"{coh_from} → {coh_to}", "Value": int(in_create_cohort.sum())}
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

        # Trend (Cohort): monthly counts by measure month
        trend_coh = base.loc[in_measure_window].copy()
        trend_coh["Measure_Month"] = trend_coh[measure_col].dt.to_period("M")
        trend_coh = trend_coh.groupby("Measure_Month")["Measure_Month"].count().reset_index(name="Cohort Count")
        trend_coh["Measure_Month"] = trend_coh["Measure_Month"].astype(str)
        charts["Cohort Trend"] = alt.Chart(trend_coh).mark_line(point=True).encode(
            x="Measure_Month:O", y="Cohort Count:Q", tooltip=["Measure_Month","Cohort Count"]
        ).properties(height=280)
        tables["Trend — Cohort (by Measure Month)"] = trend_coh

# ---------------- Narrative insights (fixed: no nested f-strings) ----------------
st.markdown("### 🔎 Insights")
if not metrics_rows:
    st.info("Turn on MTD and/or Cohort to see insights.")
else:
    bullets = []
    try:
        # MTD country leader
        if "Top 5 Countries — MTD" in tables and not tables["Top 5 Countries — MTD"].empty:
            r = tables["Top 5 Countries — MTD"].iloc[0]
            mtd_col = f"MTD Count on '{measure_col}'"
            mtd_cnt = int(r[mtd_col]) if mtd_col in r and pd.notna(r[mtd_col]) else 0
            bullets.append(f"**MTD:** Country leader is **{r['Country']}** with **{mtd_cnt:,}**.")

        # Cohort country leader
        if "Top 5 Countries — Cohort" in tables and not tables["Top 5 Countries — Cohort"].empty:
            r = tables["Top 5 Countries — Cohort"].iloc[0]
            coh_col = f"Cohort Count on '{measure_col}'"
            coh_cnt = int(r[coh_col]) if coh_col in r and pd.notna(r[coh_col]) else 0
            bullets.append(f"**Cohort:** Country leader is **{r['Country']}** with **{coh_cnt:,}**.")

        # MTD source leader (name only)
        if "Top 3 Deal Sources — MTD" in tables and not tables["Top 3 Deal Sources — MTD"].empty:
            r = tables["Top 3 Deal Sources — MTD"].iloc[0]
            bullets.append(f"**MTD:** Top deal source is **{r['JetLearn Deal Source']}**.")

        # Cohort source leader (name only)
        if "Top 3 Deal Sources — Cohort" in tables and not tables["Top 3 Deal Sources — Cohort"].empty:
            r = tables["Top 3 Deal Sources — Cohort"].iloc[0]
            bullets.append(f"**Cohort:** Top deal source is **{r['JetLearn Deal Source']}**.")
    except Exception:
        pass

    if bullets:
        st.markdown("- " + "\n- ".join(bullets))
    else:
        st.write("No highlights yet — try enabling splits or leaderboards.")

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------------- Tabs: KPIs | Tables | Trends | Export ----------------
tab_kpi, tab_tbl, tab_tr, tab_dl = st.tabs(["📌 KPIs","📋 Tables","📈 Trends","⬇️ Export"])

with tab_kpi:
    if 'metrics_rows' not in locals() or not metrics_rows:
        st.info("No KPIs yet.")
    else:
        dfk = pd.DataFrame(metrics_rows)
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

with tab_tbl:
    if not tables:
        st.info("No tables yet — adjust filters and toggles.")
    else:
        for name, frame in tables.items():
            st.subheader(name)
            st.dataframe(frame, use_container_width=True)

with tab_tr:
    shown = False
    if "MTD Trend" in charts:
        st.subheader("MTD Trend (by Create Month)")
        st.altair_chart(charts["MTD Trend"], use_container_width=True)
        shown=True
    if "Cohort Trend" in charts:
        st.subheader("Cohort Trend (by Measure Month)")
        st.altair_chart(charts["Cohort Trend"], use_container_width=True)
        shown=True
    if not shown:
        st.info("Turn on MTD and/or Cohort to see trends.")

with tab_dl:
    if not tables:
        st.info("Nothing to export yet.")
    else:
        st.markdown("#### Download result tables")
        for name, frame in tables.items():
            c1, c2 = st.columns([4,1])
            with c1: st.caption(name)
            with c2:
                st.download_button("CSV", data=to_csv_bytes(frame),
                                   file_name=f"{name.replace(' ','_').replace('×','x')}.csv",
                                   mime="text/csv", key=f"dl_{name}")

# ---------------- Footer ----------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.caption(
    f"**Measure:** {measure_col} · "
    f"Pipeline: {'All' if pipe_all else ', '.join(pipe_sel) or 'None'} · "
    f"Deal Source: {'All' if src_all else ', '.join(src_sel) or 'None'} · "
    f"Country: {'All' if ctry_all else ', '.join(ctry_sel) or 'None'} · "
    f"Counsellor: {'All' if cslr_all else ', '.join(cslr_sel) or 'None'} · "
    f"Excluded: 1.2 Invalid Deal"
)
