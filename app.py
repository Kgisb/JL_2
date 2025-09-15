# app_compare_pro_minimal_search_v3_fixed3.py
# Minimal A/B analyzer with streamlined "All → type to filter" multiselect UX
# FIX: never set a widget's session_state key during normal render; only via buttons + rerun

import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ---------- Page ----------
st.set_page_config(page_title="MTD vs Cohort — A/B Compare (Minimal+Smart)", layout="wide", page_icon="📊")
st.markdown("""
<style>
:root{ --text:#0f172a; --muted:#6b7280; --blue-600:#2563eb; --blue-700:#1d4ed8; --border: rgba(15,23,42,.10); --card:#fff; }
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.4rem; padding-bottom:.8rem; }
.nav { position:sticky; top:0; z-index:20; padding:8px 12px; background:linear-gradient(90deg,var(--blue-700),var(--blue-600)); color:#fff; border-radius:12px; margin-bottom:10px; }
.nav .title { font-weight:800; } .nav .sub { font-size:.85rem; opacity:.9; }
.filters-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:6px 8px; border:1px solid var(--border); border-radius:12px; background:#f8fafc; }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
.section-title { font-weight:800; margin:.25rem 0 .6rem; }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
.kpi .delta { font-size:.84rem; color: var(--blue-600); }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country","Student/Academic Counsellor","Deal Stage","Create Date"]
PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# ---------- Utils ----------
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
                df[col] = parsed
                date_like.append(col)
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c!="Payment Received Date"]
    return date_like

def coerce_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    if all_checked: return pd.Series(True, index=series.index)
    sel = [str(v) for v in coerce_list(selected_values)]
    if len(sel) == 0: return pd.Series(False, index=series.index)
    return series.astype(str).isin(sel)

def safe_minmax_date(s: pd.Series, fallback=(date(2020,1,1), date.today())):
    if s.isna().all(): return fallback
    return (pd.to_datetime(s.min()).date(), pd.to_datetime(s.max()).date())

def today_bounds(): t = pd.Timestamp.today().date(); return t, t
def this_month_so_far_bounds(): t = pd.Timestamp.today().date(); return t.replace(day=1), t
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
def this_year_so_far_bounds(): t = pd.Timestamp.today().date(); return date(t.year,1,1), t

def date_range_from_preset(label, series: pd.Series, key_prefix: str):
    presets = ["Today","This month so far","Last month","Last quarter","This year","Custom"]
    choice = st.radio(label, presets, horizontal=True, key=f"{key_prefix}_preset")
    if choice == "Today": f,t = today_bounds(); return f,t,choice,"Day"
    if choice == "This month so far": f,t = this_month_so_far_bounds(); return f,t,choice,"Day"
    if choice == "Last month": f,t = last_month_bounds(); return f,t,choice,"Month"
    if choice == "Last quarter": f,t = last_quarter_bounds(); return f,t,choice,"Month"
    if choice == "This year": f,t = this_year_so_far_bounds(); return f,t,choice,"Month"
    dmin,dmax = safe_minmax_date(series)
    rng = st.date_input("Custom range", (dmin,dmax), key=f"{key_prefix}_custom")
    grain = st.radio("Granularity", ["Day-wise","Week-wise","Month-wise"], horizontal=True, key=f"{key_prefix}_grain")
    grain = {"Day-wise":"Day","Week-wise":"Week","Month-wise":"Month"}[grain]
    if isinstance(rng,(tuple,list)) and len(rng)==2: return rng[0], rng[1], choice, grain
    return dmin, dmax, choice, grain

def to_csv_bytes(df: pd.DataFrame) -> bytes: return df.to_csv(index=False).encode("utf-8")

def alt_line(df, x, y, color=None, tooltip=None, height=260):
    enc = dict(x=alt.X(x, title=None), y=alt.Y(y, title=None), tooltip=tooltip or [])
    if color: enc["color"] = alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

# ---------- Top Nav ----------
with st.container():
    st.markdown('<div class="nav">', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([6,3,3])
    with c1:
        st.markdown('<div class="title">MTD vs Cohort — A/B Compare (Minimal+Smart)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">All → untick → type to search • multi-measure • presets • splits • trends</div>', unsafe_allow_html=True)
    with c2:
        if st.button("Clone A → B", key="clone_ab"):
            for k in list(st.session_state.keys()):
                if str(k).startswith("A_"): st.session_state[str(k).replace("A_","B_",1)] = st.session_state[k]
            st.rerun()
    with c3:
        col_ba,col_reset = st.columns(2)
        with col_ba:
            if st.button("Clone B → A", key="clone_ba"):
                for k in list(st.session_state.keys()):
                    if str(k).startswith("B_"): st.session_state[str(k).replace("B_","A_",1)] = st.session_state[k]
                st.rerun()
        with col_reset:
            if st.button("Reset", key="reset_all"): st.session_state.clear(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Data ----------
with st.expander("📦 Data source", expanded=True):
    col_u, col_p = st.columns([3,2])
    with col_u: uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with col_p: default_path = st.text_input("…or CSV path", value="Master_sheet_DB_10percent.csv")
    if uploaded: df = robust_read_csv(BytesIO(uploaded.getvalue()))
    else:        df = robust_read_csv(default_path)

df.columns = [c.strip() for c in df.columns]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")
    st.stop()

df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"] = df["Create Date"].dt.to_period("M")
date_like_cols = detect_measure_date_columns(df)
if not date_like_cols:
    st.error("No date-like columns besides 'Create Date' (e.g., 'Payment Received Date').")
    st.stop()

# ---------- Minimal, “All → type to filter” filter UI ----------
def summarize(values, all_flag, max_items=2):
    values = coerce_list(values)
    if all_flag: return "All"
    if not values: return "None"
    vals = [str(v) for v in values]
    return ", ".join(vals[:max_items]) + (f" +{len(vals)-max_items} more" if len(vals)>max_items else "")

def smart_multiselect(label: str, df: pd.DataFrame, colname: str, key_prefix: str):
    """
    When 'All' is unticked, show ONE native multiselect with type-ahead.
    IMPORTANT: don't write to widget keys inside render; only on buttons + rerun.
    """
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])

    all_key  = f"{key_prefix}_all"     # checkbox key
    sel_key  = f"{key_prefix}_sel"     # our stored selection (not a widget)
    ms_key   = f"{key_prefix}_ms"      # multiselect widget key

    # Initial state (only non-widget keys may be set safely)
    if all_key not in st.session_state: st.session_state[all_key] = True
    if sel_key not in st.session_state: st.session_state[sel_key] = options.copy()
    if ms_key  not in st.session_state: st.session_state[ms_key]  = st.session_state[sel_key]  # prefill once

    # Snapshot current state (do not mutate ms_key in normal flow)
    all_flag = bool(st.session_state[all_key])
    stored_sel = coerce_list(st.session_state.get(sel_key, []))

    # Effective selection used for filtering & defaults:
    effective_selected = options if all_flag else coerce_list(st.session_state.get(ms_key, stored_sel))

    summary  = summarize(effective_selected, all_flag)
    ctx = st.popover(f"{label}: {summary}") if hasattr(st, "popover") else st.expander(f"{label}: {summary}", expanded=False)
    with ctx:
        colA, colB = st.columns([1,3])
        with colA:
            st.checkbox("All", value=all_flag, key=all_key)
        with colB:
            if not st.session_state[all_key]:
                st.multiselect(
                    " ", options=options, default=effective_selected,
                    key=ms_key, placeholder=f"Type to search {label.lower()}…",
                    label_visibility="collapsed"
                )
                act1, act2 = st.columns([1,1])
                with act1:
                    if st.button("Select all", key=f"{key_prefix}_select_all", use_container_width=True):
                        # It's safe to write & rerun on button press
                        st.session_state[sel_key] = options.copy()
                        st.session_state[ms_key]  = options.copy()
                        st.experimental_rerun()
                with act2:
                    if st.button("Clear", key=f"{key_prefix}_clear", use_container_width=True):
                        st.session_state[sel_key] = []
                        st.session_state[ms_key]  = []
                        st.experimental_rerun()
            else:
                st.caption("All values selected")

    # Sync back (no writes to ms_key unless via buttons)
    if not st.session_state[all_key]:
        st.session_state[sel_key] = coerce_list(st.session_state.get(ms_key, effective_selected))
    else:
        st.session_state[sel_key] = options.copy()

    return st.session_state[all_key], st.session_state[sel_key], f"{label}: {summarize(st.session_state[sel_key], st.session_state[all_key])}"

def filters_toolbar(name: str, df: pd.DataFrame):
    st.markdown('<div class="filters-bar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([2,2,2,2,1])
    with c1: pipe_all, pipe_sel, s1 = smart_multiselect("Pipeline", df, "Pipeline", f"{name}_pipe")
    with c2: src_all,  src_sel,  s2 = smart_multiselect("Deal Source", df, "JetLearn Deal Source", f"{name}_src")
    with c3: ctry_all, ctry_sel, s3 = smart_multiselect("Country", df, "Country", f"{name}_ctry")
    with c4: cslr_all,  cslr_sel, s4 = smart_multiselect("Counsellor", df, "Student/Academic Counsellor", f"{name}_cslr")
    with c5:
        if st.button("Clear", key=f"{name}_clear", use_container_width=True):
            for prefix, col in [(f"{name}_pipe","Pipeline"), (f"{name}_src","JetLearn Deal Source"),
                                (f"{name}_ctry","Country"), (f"{name}_cslr","Student/Academic Counsellor")]:
                opts = sorted([v for v in df[col].dropna().astype(str).unique()])
                st.session_state[f"{prefix}_all"] = True
                st.session_state[f"{prefix}_sel"] = opts
                st.session_state[f"{prefix}_ms"]  = opts
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Filters — " + " · ".join([s1, s2, s3, s4]))

    mask = (
        in_filter(df["Pipeline"], pipe_all,  pipe_sel) &
        in_filter(df["JetLearn Deal Source"], src_all,   src_sel) &
        in_filter(df["Country"], ctry_all,  ctry_sel) &
        in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
    )
    base = df[mask].copy()
    return base, dict(pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
                      ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel)

def ensure_month_cols(base: pd.DataFrame, measures):
    for m in measures:
        col = f"{m}_Month"
        if m in base.columns and col not in base.columns:
            base[col] = base[m].dt.to_period("M")
    return base

# ---------- Panel ----------
def panel_controls(name: str, df: pd.DataFrame, date_like_cols):
    st.markdown(f"<div class='section-title'>Scenario {name} <span class='badge'>independent</span></div>", unsafe_allow_html=True)
    base, gstate = filters_toolbar(name, df)

    row = st.columns([4,2,2])
    measures = row[0].multiselect(f"[{name}] Measure date(s)", options=date_like_cols,
                                  default=[date_like_cols[0]] if date_like_cols else [],
                                  key=f"{name}_measures")
    mode = row[1].radio(f"[{name}] Mode", ["MTD","Cohort","Both"], horizontal=True, key=f"{name}_mode")
    mtd = mode in ("MTD","Both"); cohort = mode in ("Cohort","Both")

    if not measures:
        st.warning("Pick at least one Measure date.")
        measures = []
    base = ensure_month_cols(base, measures)

    mtd_from = mtd_to = coh_from = coh_to = None
    mtd_grain = coh_grain = "Month"
    c1, c2 = st.columns(2)
    if mtd:
        with c1:
            st.caption("Create-Date window (MTD)")
            mtd_from, mtd_to, _, mtd_grain = date_range_from_preset(f"[{name}] MTD Range", base["Create Date"], f"{name}_mtd")
    if cohort:
        with c2:
            st.caption("Measure-Date window (Cohort)")
            first_series = base[measures[0]] if measures else base["Create Date"]
            coh_from, coh_to, _, coh_grain = date_range_from_preset(f"[{name}] Cohort Range", first_series, f"{name}_coh")

    with st.expander(f"[{name}] Splits & leaderboards (optional)", expanded=False):
        srow = st.columns([3,2,2])
        split_dims = srow[0].multiselect(f"[{name}] Split by", ["JetLearn Deal Source", "Country"], default=[], key=f"{name}_split")
        show_top_countries = srow[1].toggle("Top 5 Countries", value=True, key=f"{name}_top_ctry")
        show_top_sources   = srow[2].toggle("Top 3 Deal Sources", value=True, key=f"{name}_top_src")
        show_combo_pairs   = st.toggle("Country × Deal Source (Top 10)", value=False, key=f"{name}_pair")

    return dict(name=name, base=base, measures=measures, mtd=mtd, cohort=cohort,
                mtd_from=mtd_from, mtd_to=mtd_to, coh_from=coh_from, coh_to=coh_to,
                mtd_grain=mtd_grain, coh_grain=coh_grain,
                split_dims=split_dims, show_top_countries=show_top_countries,
                show_top_sources=show_top_sources, show_combo_pairs=show_combo_pairs, **gstate)

# ---------- Engine ----------
def group_label_from_series(s: pd.Series, grain: str):
    if grain == "Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain == "Week":
        iso = pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str) + "-W" + iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

def compute_outputs(meta):
    base = meta["base"]; measures = meta["measures"]
    mtd = meta["mtd"]; cohort = meta["cohort"]
    mtd_from, mtd_to = meta["mtd_from"], meta["mtd_to"]
    coh_from, coh_to = meta["coh_from"], meta["coh_to"]
    mtd_grain, coh_grain = meta.get("mtd_grain","Month"), meta.get("coh_grain","Month")
    split_dims = meta["split_dims"]
    show_top_countries = meta["show_top_countries"]; show_top_sources = meta["show_top_sources"]
    show_combo_pairs = meta["show_combo_pairs"]

    metrics_rows, tables, charts = [], {}, {}

    # --------- MTD ---------
    if mtd and mtd_from and mtd_to and len(measures)>0:
        in_cre = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
        sub = base[in_cre].copy()
        flags = []
        for m in measures:
            if m not in sub.columns: continue
            flg = f"__MTD__{m}"
            sub[flg] = ((sub[m].notna()) & (sub[f"{m}_Month"] == sub["Create_Month"])).astype(int)
            flags.append(flg)
            metrics_rows.append({"Scope":"MTD","Metric":f"Count on '{m}'","Window":f"{mtd_from} → {mtd_to}","Value":int(sub[flg].sum())})
        metrics_rows.append({"Scope":"MTD","Metric":"Create Count in window","Window":f"{mtd_from} → {mtd_to}","Value":int(len(sub))})

        if flags:
            if split_dims:
                sub["_CreateCount"]=1
                grp = sub.groupby(split_dims, dropna=False)[flags+["_CreateCount"]].sum().reset_index()
                rename_map = {"_CreateCount":"Create Count in window"}
                rename_map.update({f: f"MTD: {m}" for f,m in zip(flags, measures)})
                grp = grp.rename(columns=rename_map).sort_values(by=f"MTD: {measures[0]}", ascending=False)
                tables[f"MTD split by {', '.join(split_dims)}"] = grp

            if show_top_countries and "Country" in sub.columns:
                g = sub.groupby("Country", dropna=False)[flags].sum().reset_index().rename(columns={f: f"MTD: {m}" for f,m in zip(flags, measures)})
                tables["Top 5 Countries — MTD"] = g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)

            if show_top_sources and "JetLearn Deal Source" in sub.columns:
                g = sub.groupby("JetLearn Deal Source", dropna=False)[flags].sum().reset_index().rename(columns={f: f"MTD: {m}" for f,m in zip(flags, measures)})
                tables["Top 3 Deal Sources — MTD"] = g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(3)

            trend = sub.copy(); trend["Bucket"] = group_label_from_series(trend["Create Date"], mtd_grain)
            t = trend.groupby("Bucket")[flags].sum().reset_index().rename(columns={f: m for f,m in zip(flags, measures)})
            long = t.melt(id_vars="Bucket", var_name="Measure", value_name="Count")
            charts["MTD Trend"] = alt_line(long, "Bucket:O", "Count:Q", color="Measure:N", tooltip=["Bucket","Measure","Count"])

    # --------- Cohort ---------
    if cohort and coh_from and coh_to and len(measures)>0:
        tmp = base.copy(); ch_flags=[]
        for m in measures:
            if m not in tmp.columns: continue
            flg = f"__COH__{m}"
            tmp[flg] = tmp[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both").astype(int)
            ch_flags.append(flg)
            metrics_rows.append({"Scope":"Cohort","Metric":f"Count on '{m}'","Window":f"{coh_from} → {coh_to}","Value":int(tmp[flg].sum())})
        in_cre_coh = base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
        metrics_rows.append({"Scope":"Cohort","Metric":"Create Count in Cohort window","Window":f"{coh_from} → {coh_to}","Value":int(in_cre_coh.sum())})

        if ch_flags:
            if split_dims:
                tmp["_CreateInCohort"] = in_cre_coh.astype(int)
                grp2 = tmp.groupby(split_dims, dropna=False)[ch_flags+["_CreateInCohort"]].sum().reset_index()
                rename_map2 = {"_CreateInCohort":"Create Count in Cohort window"}
                rename_map2.update({f: f"Cohort: {m}" for f,m in zip(ch_flags, measures)})
                grp2 = grp2.rename(columns=rename_map2).sort_values(by=f"Cohort: {measures[0]}", ascending=False)
                tables[f"Cohort split by {', '.join(split_dims)}"] = grp2

            if show_top_countries and "Country" in base.columns:
                g = tmp.groupby("Country", dropna=False)[ch_flags].sum().reset_index().rename(columns={f: f"Cohort: {m}" for f,m in zip(ch_flags, measures)})
                tables["Top 5 Countries — Cohort"] = g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)

            if show_top_sources and "JetLearn Deal Source" in base.columns:
                g = tmp.groupby("JetLearn Deal Source", dropna=False)[ch_flags].sum().reset_index().rename(columns={f: f"Cohort: {m}" for f,m in zip(ch_flags, measures)})
                tables["Top 3 Deal Sources — Cohort"] = g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(3)

            frames=[]
            for m in measures:
                mask = base[m].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
                sel = base.loc[mask, [m]].copy()
                if sel.empty: continue
                sel["Bucket"] = group_label_from_series(sel[m], coh_grain)
                t = sel.groupby("Bucket")[m].count().reset_index(name="Count"); t["Measure"]=m; frames.append(t)
            if frames:
                trend = pd.concat(frames, ignore_index=True)
                charts["Cohort Trend"] = alt_line(trend, "Bucket:O", "Count:Q", color="Measure:N", tooltip=["Bucket","Measure","Count"])

    return metrics_rows, tables, charts

def kpi_grid(dfk, label_prefix=""):
    if dfk.empty: st.info("No KPIs yet."); return
    cols = st.columns(4)
    for i,row in dfk.iterrows():
        with cols[i%4]:
            st.markdown(f"""
<div class="kpi">
  <div class="label">{label_prefix}{row['Scope']} — {row['Metric']}</div>
  <div class="value">{row['Value']:,}</div>
  <div class="delta">{row['Window']}</div>
</div>
""", unsafe_allow_html=True)

def build_compare_delta(dfA, dfB):
    if dfA.empty or dfB.empty: return pd.DataFrame()
    key = ["Scope","Metric"]
    a = dfA[key+["Value"]].copy().rename(columns={"Value":"A"}); a["A"]=pd.to_numeric(a["A"], errors="coerce")
    b = dfB[key+["Value"]].copy().rename(columns={"Value":"B"}); b["B"]=pd.to_numeric(b["B"], errors="coerce")
    out = pd.merge(a,b,on=key,how="inner")
    out["Δ"] = pd.to_numeric(out["B"] - out["A"], errors="coerce")
    denom = out["A"].astype(float); denom = denom.where(~(denom.isna() | (denom==0)))
    out["Δ%"] = ((out["Δ"].astype(float) / denom) * 100).round(1)
    return out

def mk_caption(meta):
    return (f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
            f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(coerce_list(meta['pipe_sel'])) or 'None'} · "
            f"Deal Source: {'All' if meta['src_all'] else ', '.join(coerce_list(meta['src_sel'])) or 'None'} · "
            f"Country: {'All' if meta['ctry_all'] else ', '.join(coerce_list(meta['ctry_sel'])) or 'None'} · "
            f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(coerce_list(meta['cslr_sel'])) or 'None'}")

# ---------- Panels ----------
left, right = st.columns(2, gap="large")
with left:  metaA = panel_controls("A", df, date_like_cols)
with right: metaB = panel_controls("B", df, date_like_cols)

# ---------- Results toggles ----------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
r1,r2,r3,r4 = st.columns([2,2,2,2])
show_kpis    = r1.toggle("Show KPIs", True)
show_splits  = r2.toggle("Show Splits & Leaderboards", False)
show_trends  = r3.toggle("Show Trends", True)
show_compare = r4.toggle("Show Smart Compare", True)

# ---------- Compute ----------
with st.spinner("Crunching numbers…"):
    metricsA, tablesA, chartsA = compute_outputs(metaA)
    metricsB, tablesB, chartsB = compute_outputs(metaB)

# ---------- KPIs ----------
if show_kpis:
    st.markdown("### 📌 KPI Overview")
    kc1,kc2 = st.columns(2)
    with kc1: st.markdown("**Scenario A**"); dfA = pd.DataFrame(metricsA); kpi_grid(dfA,"A · ")
    with kc2: st.markdown("**Scenario B**"); dfB = pd.DataFrame(metricsB); kpi_grid(dfB,"B · ")

# ---------- Splits ----------
if show_splits:
    st.markdown("### 🧩 Splits & Leaderboards")
    tabA, tabB = st.tabs(["Scenario A","Scenario B"])
    with tabA:
        if not tablesA: st.info("No tables — enable splits/leaderboards in A.")
        else:
            for name,frame in tablesA.items():
                st.subheader("A · "+name); st.dataframe(frame, use_container_width=True)
                st.download_button("Download CSV (A · "+name+")", to_csv_bytes(frame),
                                   file_name=f"A_{name.replace(' ','_')}.csv", mime="text/csv")
    with tabB:
        if not tablesB: st.info("No tables — enable splits/leaderboards in B.")
        else:
            for name,frame in tablesB.items():
                st.subheader("B · "+name); st.dataframe(frame, use_container_width=True)
                st.download_button("Download CSV (B · "+name+")", to_csv_bytes(frame),
                                   file_name=f"B_{name.replace(' ','_')}.csv", mime="text/csv")

# ---------- Trends ----------
if show_trends:
    st.markdown("### 📈 Trends")
    t1,t2 = st.columns(2)
    with t1:
        if "MTD Trend" in chartsA or "Cohort Trend" in chartsA:
            st.markdown("**Scenario A**")
            if "MTD Trend" in chartsA: st.altair_chart(chartsA["MTD Trend"], use_container_width=True)
            if "Cohort Trend" in chartsA: st.altair_chart(chartsA["Cohort Trend"], use_container_width=True)
        else: st.info("Enable MTD/Cohort in A and set ranges.")
    with t2:
        if "MTD Trend" in chartsB or "Cohort Trend" in chartsB:
            st.markdown("**Scenario B**")
            if "MTD Trend" in chartsB: st.altair_chart(chartsB["MTD Trend"], use_container_width=True)
            if "Cohort Trend" in chartsB: st.altair_chart(chartsB["Cohort Trend"], use_container_width=True)
        else: st.info("Enable MTD/Cohort in B and set ranges.")

# ---------- Smart Compare ----------
if show_compare:
    st.markdown("### 🧠 Smart Compare (A vs B)")
    dA,dB = pd.DataFrame(metricsA), pd.DataFrame(metricsB)
    if not dA.empty and not dB.empty:
        cmp = build_compare_delta(dA, dB)
        if cmp.empty:
            st.info("Adjust scenarios to produce comparable KPIs.")
        else:
            st.dataframe(cmp, use_container_width=True)
            try:
                if set(metaA["measures"]) == set(metaB["measures"]):
                    sub = cmp[cmp["Metric"].str.startswith("Count on '")].copy()
                    if not sub.empty:
                        sub["Measure"] = sub["Metric"].str.extract(r"Count on '(.+)'")
                        a_long = sub.rename(columns={"A":"Value"})[["Measure","Scope","Value"]]; a_long["Scenario"]="A"
                        b_long = sub.rename(columns={"B":"Value"})[["Measure","Scope","Value"]]; b_long["Scenario"]="B"
                        long = pd.concat([a_long,b_long], ignore_index=True)
                        ch = alt.Chart(long).mark_bar().encode(
                            x=alt.X("Scope:N", title=None), y=alt.Y("Value:Q"),
                            color=alt.Color("Scenario:N", scale=alt.Scale(range=PALETTE[:2])),
                            column=alt.Column("Measure:N", header=alt.Header(title=None, labelAngle=0)),
                            tooltip=["Measure","Scenario","Scope","Value"]
                        ).properties(height=260)
                        st.altair_chart(ch, use_container_width=True)
            except Exception:
                pass
    else:
        st.info("Turn on KPIs for both scenarios to enable compare.")

# ---------- Foot ----------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
def mk_caption(meta):
    return (f"Measures: {', '.join(meta['measures']) if meta['measures'] else '—'} · "
            f"Pipeline: {'All' if meta['pipe_all'] else ', '.join(coerce_list(meta['pipe_sel'])) or 'None'} · "
            f"Deal Source: {'All' if meta['src_all'] else ', '.join(coerce_list(meta['src_sel'])) or 'None'} · "
            f"Country: {'All' if meta['ctry_all'] else ', '.join(coerce_list(meta['ctry_sel'])) or 'None'} · "
            f"Counsellor: {'All' if meta['cslr_all'] else ', '.join(coerce_list(meta['cslr_sel'])) or 'None'}")
st.caption("**Scenario A** — " + mk_caption(metaA))
st.caption("**Scenario B** — " + mk_caption(metaB))
st.caption("Excluded globally: 1.2 Invalid Deal")
