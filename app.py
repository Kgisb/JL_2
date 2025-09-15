# app_compare_pro_clean.py — Pro A/B analyzer with compact popover filters (no Compact Mode toggle)
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ------------------------ Page & Theme ------------------------
st.set_page_config(page_title="MTD vs Cohort — A/B Compare (Pro, Clean)",
                   layout="wide", page_icon="📊")

st.markdown("""
<style>
:root{
  --bg:#ffffff; --text:#0f172a; --muted:#6b7280;
  --blue:#1e40af; --blue-600:#2563eb; --blue-700:#1d4ed8;
  --border: rgba(15,23,42,.10); --card:#ffffff;
}
html, body, [class*="css"] {
  font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji";
}
.block-container { padding-top: .4rem; padding-bottom: .8rem; }

/* Top bar */
.nav {
  position: sticky; top: 0; z-index: 20; padding: 8px 12px;
  background: linear-gradient(90deg, var(--blue-700), var(--blue-600));
  color: #fff; border-radius: 12px; margin-bottom: 10px;
}
.nav .title { font-weight: 800; letter-spacing:.2px; }
.nav .sub   { font-size:.85rem; opacity:.9; margin-top:2px; }
.nav .btn {
  border:1px solid rgba(255,255,255,.35); color:#fff; background: transparent;
  padding: 6px 10px; border-radius: 10px; font-size:.85rem; cursor:pointer;
}
.nav .btn:hover { background: rgba(255,255,255,.08); }

/* Compact filters bar */
.filters-bar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 6px 8px; border: 1px solid var(--border); border-radius: 12px;
  background: #f8fafc;
}

/* Sections & cards */
.section-title { display:flex; align-items:center; gap:.5rem; font-weight:800; margin:.25rem 0 .6rem; }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
.kpi .delta { font-size:.84rem; color: var(--blue-600); }

@media (max-width: 820px) {
  .block-container { padding-left:.5rem; padding-right:.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------ Constants ------------------------
REQUIRED_COLS = [
    "Pipeline","JetLearn Deal Source","Country",
    "Student/Academic Counsellor","Deal Stage","Create Date",
]
PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# ------------------------ Utilities ------------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    date_like = []
    for col in df.columns:
        if col == "Create Date":
            continue
        if any(k in col.lower() for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed
                date_like.append(col)
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

# Date presets
def today_bounds():
    t = pd.Timestamp.today().date(); return t, t
def this_month_so_far_bounds():
    t = pd.Timestamp.today().date(); return t.replace(day=1), t
def last_month_bounds():
    first_this = pd.Timestamp.today().date().replace(day=1)
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return first_prev, last_prev
def quarter_start(y, q): return date(y, 3*(q-1)+1, 1)
def last_quarter_bounds():
    t = pd.Timestamp.today().date(); q = (t.month - 1)//3 + 1
    if q == 1: y, lq = t.year - 1, 4
    else:      y, lq = t.year, q - 1
    start = quarter_start(y, lq)
    next_start = quarter_start(y+1, 1) if lq == 4 else quarter_start(y, lq+1)
    return start, (next_start - timedelta(days=1))
def this_year_so_far_bounds():
    t = pd.Timestamp.today().date(); return date(t.year,1,1), t

def date_range_from_preset(label, series: pd.Series, key_prefix: str):
    presets = ["Today","This month so far","Last month","Last quarter","This year","Custom"]
    choice = st.radio(label, presets, horizontal=True, key=f"{key_prefix}_preset")
    if choice == "Today": return today_bounds()
    if choice == "This month so far": return this_month_so_far_bounds()
    if choice == "Last month": return last_month_bounds()
    if choice == "Last quarter": return last_quarter_bounds()
    if choice == "This year": return this_year_so_far_bounds()
    dmin, dmax = safe_minmax_date(series)
    rng = st.date_input("Custom range", (dmin, dmax), key=f"{key_prefix}_custom")
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        return rng[0], rng[1]
    return dmin, dmax

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def alt_line(df, x, y, color=None, tooltip=None, height=260):
    enc = dict(x=alt.X(x, title=None), y=alt.Y(y, title=None), tooltip=tooltip or [])
    if color: enc["color"] = alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

# ------------------------ Top Nav ------------------------
with st.container():
    st.markdown('<div class="nav">', unsafe_allow_html=True)
    c1, c2 = st.columns([6,6])
    with c1:
        left1, left2 = st.columns([1,11])
        with left1: st.markdown('<span class="menu-pill">☰</span>', unsafe_allow_html=True)
        with left2:
            st.markdown('<div class="title">MTD vs Cohort — A/B Compare (Pro, Clean)</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Compact popover filters • multi-measure • presets • smart compare</div>', unsafe_allow_html=True)
    with c2:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Clone A → B", key="clone_ab"):
                for k in list(st.session_state.keys()):
                    if str(k).startswith("A_"):
                        st.session_state[str(k).replace("A_","B_",1)] = st.session_state[k]
                st.rerun()
        with b2:
            if st.button("Clone B → A", key="clone_ba"):
                for k in list(st.session_state.keys()):
                    if str(k).startswith("B_"):
                        st.session_state[str(k).replace("B_","A_",1)] = st.session_state[k]
                st.rerun()
        with b3:
            if st.button("Reset", key="reset_all"):
                st.session_state.clear(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------ Data source ------------------------
with st.expander("📦 Data source", expanded=True):
    col_u, col_p = st.columns([3,2])
    with col_u: uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with col_p: default_path = st.text_input("…or CSV path", value="Master_sheet_DB_10percent.csv")
    if uploaded: df = robust_read_csv(BytesIO(uploaded.getvalue()))
    else:        df = robust_read_csv(default_path)

df.columns = [c.strip() for c in df.columns]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}"); st.stop()

# Exclude invalid deals, parse
df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"] = df["Create Date"].dt.to_period("M")
date_like_cols = detect_measure_date_columns(df)
if not date_like_cols:
    st.error("No date-like columns found besides 'Create Date' (e.g., 'Payment Received Date')."); st.stop()

# ------------------------ Compact Global Filters ------------------------
def summarize_values(values, all_flag, max_items=3):
    if all_flag: return "All"
    if not values: return "None"
    vals = [str(v) for v in values]
    if len(vals) <= max_items: return ", ".join(vals)
    return ", ".join(vals[:max_items]) + f" +{len(vals) - max_items} more"

def filter_pop(label, df, colname, key_prefix):
    """Compact control: button label shows summary; click reveals multiselect in popover or expander."""
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])
    all_key, sel_key = f"{key_prefix}_all", f"{key_prefix}_sel"
    if all_key not in st.session_state: st.session_state[all_key] = True
    if sel_key not in st.session_state: st.session_state[sel_key] = options

    all_flag = st.session_state[all_key]
    cur_selected = st.session_state[sel_key]
    summary = summarize_values(cur_selected, all_flag)

    if hasattr(st, "popover"):
        with st.popover(f"{label}: {summary}"):
            st.checkbox("All", value=all_flag, key=all_key)
            st.multiselect(f"Select {label}", options, default=options,
                           disabled=st.session_state[all_key], key=sel_key)
    else:
        with st.expander(f"{label}: {summary}", expanded=False):
            st.checkbox("All", value=all_flag, key=all_key)
            st.multiselect(f"Select {label}", options, default=options,
                           disabled=st.session_state[all_key], key=sel_key)

    return st.session_state[all_key], st.session_state[sel_key], f"{label}: {summary}"

def filters_toolbar(name, df):
    st.markdown("<div class='filters-bar'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([2,2,2,2,1])
    with c1: pipe_all, pipe_sel, s1 = filter_pop("Pipeline", df, "Pipeline", f"{name}_pipe")
    with c2: src_all,  src_sel,  s2 = filter_pop("Deal Source", df, "JetLearn Deal Source", f"{name}_src")
    with c3: ctry_all, ctry_sel, s3 = filter_pop("Country", df, "Country", f"{name}_ctry")
    with c4: cslr_all, cslr_sel, s4 = filter_pop("Counsellor", df, "Student/Academic Counsellor", f"{name}_cslr")
    with c5:
        if st.button("Clear", key=f"{name}_clear"):
            for prefix, col in [(f"{name}_pipe","Pipeline"), (f"{name}_src","JetLearn Deal Source"),
                                (f"{name}_ctry","Country"), (f"{name}_cslr","Student/Academic Counsellor")]:
                st.session_state[f"{prefix}_all"] = True
                st.session_state[f"{prefix}_sel"] = sorted([v for v in df[col].dropna().astype(str).unique()])
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Filters — " + " · ".join([s1, s2, s3, s4]))

    mask_cat = (
        in_filter(df["Pipeline"], pipe_all, pipe_sel) &
        in_filter(df["JetLearn Deal Source"], src_all, src_sel) &
        in_filter(df["Country"], ctry_all, ctry_sel) &
        in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
    )
    base = df[mask_cat].copy()
    return base, dict(pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
                      ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel)

def ensure_month_cols(base: pd.DataFrame, measures):
    for m in measures:
        col = f"{m}_Month"
        if col not in base.columns:
            base[col] = base[m].dt.to_period("M")
    return base

# ------------------------ Panel Controls ------------------------
def panel_controls(name: str, df: pd.DataFrame, date_like_cols):
    st.markdown(f"<div class='section-title'>Scenario {name} <span class='badge'>independent</span></div>", unsafe_allow_html=True)

    # Compact Global Filters (always hidden until clicked; summary shown)
    base, gstate = filters_toolbar(name, df)

    # Measures & windows row
    mrow1 = st.columns([4,2,2])
    measures = mrow1[0].multiselect(f"[{name}] Measure date(s)", options=date_like_cols,
                                    default=[date_like_cols[0]] if date_like_cols else [],
                                    key=f"{name}_measures")
    window_mode = mrow1[1].radio(f"[{name}] Mode", ["MTD","Cohort","Both"], horizontal=True, key=f"{name}_mode")
    _ = mrow1[2].markdown("&nbsp;")  # spacer

    mtd = window_mode in ("MTD","Both")
    cohort = window_mode in ("Cohort","Both")
    if not measures:
        st.warning("Pick at least one Measure date.")
    base = ensure_month_cols(base, measures)

    # Date presets (Create-Date for MTD; Measure-Date for Cohort)
    mtd_from = mtd_to = coh_from = coh_to = None
    c1, c2 = st.columns(2)
    if mtd:
        with c1:
            st.caption("Create-Date window (MTD)")
            mtd_from, mtd_to = date_range_from_preset(f"[{name}] MTD Range", base["Create Date"], f"{name}_mtd")
    if cohort:
        with c2:
            st.caption("Measure-Date window (Cohort)")
            first_series = base[measures[0]] if measures else base["Create Date"]
            coh_from, coh_to = date_range_from_preset(f"[{name}] Cohort Range", first_series, f"{name}_coh")

    # Splits & leaderboards (optional)
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
        **gstate
    )

# ------------------------ Engine ------------------------
def compute_outputs(meta):
    base = meta["base"]; measures = meta["measures"]
    mtd = meta["mtd"]; cohort = meta["cohort"]
    mtd_from, mtd_to = meta["mtd_from"], meta["mtd_to"]
    coh_from, coh_to = meta["coh_from"], meta["coh_to"]
    split_dims = meta["split_dims"]
    show_top_countries = meta["show_top_countries"]
    show_top_sources   = meta["show_top_sources"]
    show_combo_pairs   = meta["show_combo_pairs"]

    metrics_rows, tables, charts = [], {}, {}

    # ----- MTD -----
    if mtd and mtd_from and mtd_to and len(measures)>0:
        in_create_window = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
        sub_mtd = base[in_create_window].copy()
        mtd_flag_cols = []
        for m in measures:
            col_flag = f"__MTD__{m}"
            sub_mtd[col_flag] = ((sub_mtd[m].notna()) & (sub_mtd[f"{m}_Month"] == sub_mtd["Create_Month"])).astype(int)
            mtd_flag_cols.append(col_flag)
            metrics_rows.append({"Scope":"MTD","Metric":f"Count on '{m}'","Window":f"{mtd_from} → {mtd_to}","Value":int(sub_mtd[col_flag].sum())})
        metrics_rows.append({"Scope":"MTD","Metric":"Create Count in window","Window":f"{mtd_from} → {mtd_to}","Value":int(len(sub_mtd))})

        if split_dims:
            agg_dict = {flag:"sum" for flag in mtd_flag_cols}
            sub_mtd["_CreateCount"] = 1
            agg_dict["_CreateCount"] = "sum"
            grp = sub_mtd.groupby(split_dims, dropna=False).agg(agg_dict).reset_index()
            grp = grp.rename(columns={"_CreateCount":"Create Count in window", **{f"__MTD__{m}":f"MTD: {m}" for m in
