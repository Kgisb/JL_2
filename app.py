# app_drawer_ui.py
# Engaging UI: top app bar with hamburger → filters drawer (sidebar) + back button.
# A/B scenarios, multi-measure, MTD/Cohort presets + granularity, splits incl. Counsellor,
# smart searchable multiselects, safe clone A↔B, minimal aesthetic.

import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# ---------- Page ----------
st.set_page_config(page_title="MTD vs Cohort — Drawer UI",
                   layout="wide", page_icon="📊")
st.markdown("""
<style>
:root{
  --text:#0f172a; --muted:#64748b; --blue-600:#2563eb; --blue-700:#1d4ed8;
  --border: rgba(15,23,42,.10); --card:#fff; --bg:#f8fafc;
}
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.4rem; padding-bottom:.8rem; }
.topbar {
  position:sticky; top:0; z-index:50; display:flex; gap:10px; align-items:center;
  padding:8px 12px; background:linear-gradient(90deg,var(--blue-700),var(--blue-600));
  color:#fff; border-radius:12px; margin-bottom:10px;
}
.topbar .title { font-weight:800; font-size:1.05rem; margin-right:auto; }
.topbar .btn {
  background:rgba(255,255,255,.12); color:#fff; padding:6px 10px; border-radius:10px; cursor:pointer;
  border:1px solid rgba(255,255,255,.18);
}
.section-title { font-weight:800; margin:.25rem 0 .6rem; color:var(--text); }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
.kpi .delta { font-size:.84rem; color: var(--blue-600); }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.filters-note { color:var(--muted); font-size:.82rem; margin:4px 0 8px; }
.sidebar-head { display:flex; align-items:center; gap:8px; }
.sidebar-back { background:#fff; border:1px solid var(--border); color:#111; padding:6px 10px; border-radius:10px; cursor:pointer; }
.sidebar-caption { color:#111; font-weight:700; }
.popcap { font-size:.78rem; color:var(--muted); margin-top:2px; }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country",
                 "Student/Academic Counsellor","Deal Stage","Create Date"]
PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# ---------- Clone helpers (safe) ----------
def _request_clone(direction: str):
    st.session_state["__clone_request__"] = direction

def _perform_clone_if_requested():
    direction = st.session_state.get("__clone_request__")
    if not direction:
        return
    src_prefix, dst_prefix = ("A_", "B_") if direction == "A2B" else ("B_", "A_")
    for k in list(st.session_state.keys()):
        if not k.startswith(src_prefix):
            continue
        if k.endswith("_select_all") or k.endswith("_clear"):
            continue
        st.session_state[k.replace(src_prefix, dst_prefix, 1)] = st.session_state[k]
    st.session_state["__clone_request__"] = None
    st.rerun()

# Early clone pass
_perform_clone_if_requested()

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
def quarter_end(y, q):
    if q == 4: return date(y,12,31)
    return quarter_start(y, q+1) - timedelta(days=1)
def last_quarter_bounds():
    t = pd.Timestamp.today().date(); q = (t.month - 1)//3 + 1
    if q == 1: y, lq = t.year - 1, 4
    else:      y, lq = t.year, q - 1
    return quarter_start(y, lq), quarter_end(y, lq)
def this_year_so_far_bounds(): t = pd.Timestamp.today().date(); return date(t.year,1,1), t

def date_range_from_preset(label, series: pd.Series, key_prefix: str):
    presets = ["Today","This month so far","Last month","Last quarter","This year","Custom"]
    default_grain_map = {
        "Today": "Day", "This month so far": "Day", "Last month": "Month",
        "Last quarter": "Month", "This year": "Month", "Custom": "Month",
    }
    c1, c2 = st.columns([3,2])
    with c1:
        choice = st.radio(label, presets, horizontal=True, key=f"{key_prefix}_preset")
    with c2:
        default_grain = default_grain_map.get(choice, "Month")
        grain = st.radio("Granularity", ["Day","Week","Month"], horizontal=True,
                         index=["Day","Week","Month"].index(default_grain),
                         key=f"{key_prefix}_grain")

    if choice == "Today":               f,t = today_bounds()
    elif choice == "This month so far": f,t = this_month_so_far_bounds()
    elif choice == "Last month":        f,t = last_month_bounds()
    elif choice == "Last quarter":      f,t = last_quarter_bounds()
    elif choice == "This year":         f,t = this_year_so_far_bounds()
    else:
        dmin,dmax = safe_minmax_date(series)
        rng = st.date_input("Custom range", (dmin,dmax), key=f"{key_prefix}_custom")
        f,t = (rng[0], rng[1]) if isinstance(rng,(tuple,list)) and len(rng)==2 else (dmin, dmax)

    if f > t:
        st.error("'From' is after 'To'. Please adjust.")
    return f, t, choice, grain

def to_csv_bytes(df: pd.DataFrame) -> bytes: return df.to_csv(index=False).encode("utf-8")

def alt_line(df, x, y, color=None, tooltip=None, height=260):
    enc = dict(x=alt.X(x, title=None), y=alt.Y(y, title=None), tooltip=tooltip or [])
    if color: enc["color"] = alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

# ---------- Global state init ----------
def init_defaults(name: str, df: pd.DataFrame, date_like_cols):
    # Drawer visibility
    if "filters_open" not in st.session_state:
        st.session_state["filters_open"] = True

    # Select-all defaults
    for prefix, col in [
        (f"{name}_pipe","Pipeline"),
        (f"{name}_src","JetLearn Deal Source"),
        (f"{name}_ctry","Country"),
        (f"{name}_cslr","Student/Academic Counsellor"),
    ]:
        opts = sorted([v for v in df[col].dropna().astype(str).unique()])
        if f"{prefix}_all" not in st.session_state: st.session_state[f"{prefix}_all"] = True
        if f"{prefix}_ms"  not in st.session_state: st.session_state[f"{prefix}_ms"]  = opts

    # Measures + mode
    if f"{name}_measures" not in st.session_state:
        st.session_state[f"{name}_measures"] = [date_like_cols[0]] if date_like_cols else []
    if f"{name}_mode" not in st.session_state:
        st.session_state[f"{name}_mode"] = "Both"

    # Presets/grain
    for kind in ["mtd","coh"]:
        if f"{name}_{kind}_preset" not in st.session_state: st.session_state[f"{name}_{kind}_preset"] = "This month so far"
        if f"{name}_{kind}_grain"  not in st.session_state: st.session_state[f"{name}_{kind}_grain"]  = "Month"
        if f"{name}_{kind}_custom" not in st.session_state:
            dmin,dmax = safe_minmax_date(df["Create Date"])
            st.session_state[f"{name}_{kind}_custom"] = (dmin,dmax)

    # Splits
    if f"{name}_split" not in st.session_state: st.session_state[f"{name}_split"] = []
    if f"{name}_top_ctry" not in st.session_state: st.session_state[f"{name}_top_ctry"] = True
    if f"{name}_top_src"  not in st.session_state: st.session_state[f"{name}_top_src"]  = True
    if f"{name}_top_cslr" not in st.session_state: st.session_state[f"{name}_top_cslr"] = False
    if f"{name}_pair"     not in st.session_state: st.session_state[f"{name}_pair"]     = False

def _summary(values, all_flag, max_items=2):
    vals = coerce_list(values)
    if all_flag: return "All"
    if not vals: return "None"
    s = ", ".join(map(str, vals[:max_items]))
    if len(vals) > max_items: s += f" +{len(vals)-max_items} more"
    return s

# Safe callbacks
def _toggle_filters(open_state: bool): st.session_state["filters_open"] = open_state
def _select_all_cb(ms_key: str, all_key: str, options: list):
    st.session_state[ms_key] = options; st.session_state[all_key] = True
def _clear_cb(ms_key: str, all_key: str):
    st.session_state[ms_key] = []; st.session_state[all_key] = False
def _reset_all_cb(): st.session_state.clear()

# ---------- Smart multiselect widget ----------
def unified_multifilter(label: str, df: pd.DataFrame, colname: str, key_prefix: str):
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])
    all_key = f"{key_prefix}_all"; ms_key  = f"{key_prefix}_ms"
    if all_key not in st.session_state: st.session_state[all_key] = True
    if ms_key  not in st.session_state: st.session_state[ms_key]  = options.copy()

    all_flag = bool(st.session_state[all_key])
    selected = coerce_list(st.session_state.get(ms_key, []))
    effective = options if all_flag else selected

    header = f"{label}: {_summary(effective, all_flag)}"
    ctx = st.popover(header) if hasattr(st, "popover") else st.expander(header, expanded=False)
    with ctx:
        left, right = st.columns([1, 3])
        with left:
            st.checkbox("All", value=all_flag, key=all_key)
        with right:
            disabled = st.session_state[all_key]
            st.multiselect(label, options=options, default=selected, key=ms_key,
                           placeholder=f"Type to search {label.lower()}…",
                           label_visibility="collapsed", disabled=disabled)
            c1, c2 = st.columns(2)
            with c1:
                st.button("Select all", key=f"{key_prefix}_select_all", use_container_width=True,
                          on_click=_select_all_cb, args=(ms_key, all_key, options))
            with c2:
                st.button("Clear", key=f"{key_prefix}_clear", use_container_width=True,
                          on_click=_clear_cb, args=(ms_key, all_key))
    all_flag = bool(st.session_state[all_key])
    selected = coerce_list(st.session_state.get(ms_key, []))
    effective = options if all_flag else selected
    return all_flag, effective, f"{label}: {_summary(effective, all_flag)}"

def filters_block(name: str, df: pd.DataFrame):
    st.markdown(f"**Scenario {name}** <span class='badge'>independent</span>", unsafe_allow_html=True)
    pipe_all, pipe_sel, s1 = unified_multifilter("Pipeline", df, "Pipeline", f"{name}_pipe")
    src_all,  src_sel,  s2 = unified_multifilter("Deal Source", df, "JetLearn Deal Source", f"{name}_src")
    ctry_all, ctry_sel, s3 = unified_multifilter("Country", df, "Country", f"{name}_ctry")
    cslr_all, cslr_sel, s4 = unified_multifilter("Counsellor", df, "Student/Academic Counsellor", f"{name}_cslr")
    st.markdown(f"<div class='popcap'>Filters — {s1} · {s2} · {s3} · {s4}</div>", unsafe_allow_html=True)
    return dict(pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
                ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel)

def ensure_month_cols(base: pd.DataFrame, measures):
    for m in measures:
        col = f"{m}_Month"
        if m in base.columns and col not in base.columns:
            base[col] = base[m].dt.to_period("M")
    return base

def group_label_from_series(s: pd.Series, grain: str):
    if grain == "Day":  return pd.to_datetime(s).dt.date.astype(str)
    if grain == "Week":
        iso = pd.to_datetime(s).dt.isocalendar()
        return (iso['year'].astype(str) + "-W" + iso['week'].astype(str).str.zfill(2))
    return pd.to_datetime(s).dt.to_period("M").astype(str)

# ---------- Engine ----------
def compute_outputs(meta):
    base = meta["base"]; measures = meta["measures"]
    mtd = meta["mtd"]; cohort = meta["cohort"]
    mtd_from, mtd_to = meta["mtd_from"], meta["mtd_to"]
    coh_from, coh_to = meta["coh_from"], meta["coh_to"]
    mtd_grain, coh_grain = meta.get("mtd_grain","Month"), meta.get("coh_grain","Month")
    split_dims = meta["split_dims"]
    show_top_countries   = meta["show_top_countries"]
    show_top_sources     = meta["show_top_sources"]
    show_top_counsellors = meta["show_top_counsellors"]
    show_combo_pairs     = meta["show_combo_pairs"]

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

            if show_top_counsellors and "Student/Academic Counsellor" in sub.columns:
                g = sub.groupby("Student/Academic Counsellor", dropna=False)[flags].sum().reset_index().rename(columns={f: f"MTD: {m}" for f,m in zip(flags, measures)})
                tables["Top 5 Counsellors — MTD"] = g.sort_values(by=f"MTD: {measures[0]}", ascending=False).head(5)

            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(sub.columns):
                both = sub.groupby(["Country","JetLearn Deal Source"], dropna=False)[flags].sum().reset_index().rename(
                    columns={f: f"MTD: {m}" for f,m in zip(flags, measures)}
                ).sort_values(by=f"MTD: {measures[0]}", ascending=False).head(10)
                tables["Top Country × Deal Source — MTD"] = both

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

            if show_top_counsellors and "Student/Academic Counsellor" in base.columns:
                g = tmp.groupby("Student/Academic Counsellor", dropna=False)[ch_flags].sum().reset_index().rename(columns={f: f"Cohort: {m}" for f,m in zip(ch_flags, measures)})
                tables["Top 5 Counsellors — Cohort"] = g.sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(5)

            if show_combo_pairs and {"Country","JetLearn Deal Source"}.issubset(base.columns):
                both2 = tmp.groupby(["Country","JetLearn Deal Source"], dropna=False)[ch_flags].sum().reset_index().rename(
                    columns={f: f"Cohort: {m}" for f,m in zip(ch_flags, measures)}
                ).sort_values(by=f"Cohort: {measures[0]}", ascending=False).head(10)
                tables["Top Country × Deal Source — Cohort"] = both2

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

# ---------- Top Bar ----------
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([5, 2, 2, 2])
    with c1:
        st.markdown('<div class="title">☰ MTD vs Cohort — Drawer UI</div>', unsafe_allow_html=True)
    with c2:
        st.button("☰ Filters", key="open_filters", on_click=_toggle_filters, args=(True,),
                  use_container_width=True)
    with c3:
        st.button("Clone A → B", key="clone_ab_btn", on_click=_request_clone, args=("A2B",),
                  use_container_width=True)
    with c4:
        b1,b2 = st.columns(2)
        with b1:
            st.button("Clone B → A", key="clone_ba_btn", on_click=_request_clone, args=("B2A",),
                      use_container_width=True)
        with b2:
            st.button("Reset", key="reset_all", on_click=_reset_all_cb, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Data source ----------
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

# Initialize defaults for both scenarios
init_defaults("A", df, date_like_cols)
init_defaults("B", df, date_like_cols)

# ---------- Sidebar (drawer) ----------
if st.session_state.get("filters_open", True):
    with st.sidebar:
        st.markdown("<div class='sidebar-head'>"
                    "<button class='sidebar-back' onclick='window.parent.postMessage({type:\"streamlit:setComponentValue\", value:\"close\"}, \"*\")'>← Back</button>"
                    "<span class='sidebar-caption'>Filters</span></div>", unsafe_allow_html=True)
        # Back button (real Streamlit state toggle)
        st.button("← Back (close drawer)", use_container_width=True,
                  on_click=_toggle_filters, args=(False,), key="close_filters_btn")

        st.markdown("##### Scenario A")
        gA = filters_block("A", df)
        # Measure & mode for A
        mrowA = st.columns([3, 1.4, 1.4])
        with mrowA[0]:
            st.multiselect("[A] Measure date(s)", options=date_like_cols,
                           key="A_measures")
        with mrowA[1]:
            st.radio("[A] Mode", ["MTD","Cohort","Both"], horizontal=True, key="A_mode")
        with mrowA[2]:
            st.empty()

        # MTD/Cohort ranges for A
        if st.session_state["A_mode"] in ("MTD","Both"):
            st.caption("Create-Date window (MTD)")
            date_range_from_preset("[A] MTD Range", df["Create Date"], "A_mtd")
        if st.session_state["A_mode"] in ("Cohort","Both"):
            st.caption("Measure-Date window (Cohort)")
            # use first selected measure to hint range
            measA = st.session_state.get("A_measures", [])
            base_series = df[measA[0]] if measA else df["Create Date"]
            date_range_from_preset("[A] Cohort Range", base_series, "A_coh")

        st.markdown("---")
        st.markdown("##### Scenario B")
        gB = filters_block("B", df)
        mrowB = st.columns([3, 1.4, 1.4])
        with mrowB[0]:
            st.multiselect("[B] Measure date(s)", options=date_like_cols,
                           key="B_measures")
        with mrowB[1]:
            st.radio("[B] Mode", ["MTD","Cohort","Both"], horizontal=True, key="B_mode")
        with mrowB[2]:
            st.empty()

        if st.session_state["B_mode"] in ("MTD","Both"):
            st.caption("Create-Date window (MTD)")
            date_range_from_preset("[B] MTD Range", df["Create Date"], "B_mtd")
        if st.session_state["B_mode"] in ("Cohort","Both"):
            st.caption("Measure-Date window (Cohort)")
            measB = st.session_state.get("B_measures", [])
            base_series = df[measB[0]] if measB else df["Create Date"]
            date_range_from_preset("[B] Cohort Range", base_series, "B_coh")

        # Splits & leaderboards toggles live with each scenario
        st.markdown("---")
        st.toggle("[A] Top 5 Countries", value=st.session_state["A_top_ctry"], key="A_top_ctry")
        st.toggle("[A] Top 3 Deal Sources", value=st.session_state["A_top_src"], key="A_top_src")
        st.toggle("[A] Top 5 Counsellors", value=st.session_state["A_top_cslr"], key="A_top_cslr")
        st.multiselect("[A] Split by", ["JetLearn Deal Source","Country","Student/Academic Counsellor"],
                       key="A_split")
        st.toggle("[A] Country × Deal Source (Top 10)", value=st.session_state["A_pair"], key="A_pair")

        st.markdown("---")
        st.toggle("[B] Top 5 Countries", value=st.session_state["B_top_ctry"], key="B_top_ctry")
        st.toggle("[B] Top 3 Deal Sources", value=st.session_state["B_top_src"], key="B_top_src")
        st.toggle("[B] Top 5 Counsellors", value=st.session_state["B_top_cslr"], key="B_top_cslr")
        st.multiselect("[B] Split by", ["JetLearn Deal Source","Country","Student/Academic Counsellor"],
                       key="B_split")
        st.toggle("[B] Country × Deal Source (Top 10)", value=st.session_state["B_pair"], key="B_pair")

# ---------- Assemble meta from state ----------
def assemble_meta(name: str, df: pd.DataFrame):
    # categorical mask
    pipe_all = st.session_state.get(f"{name}_pipe_all", True)
    pipe_sel = st.session_state.get(f"{name}_pipe_ms", [])
    src_all  = st.session_state.get(f"{name}_src_all", True)
    src_sel  = st.session_state.get(f"{name}_src_ms", [])
    ctry_all = st.session_state.get(f"{name}_ctry_all", True)
    ctry_sel = st.session_state.get(f"{name}_ctry_ms", [])
    cslr_all = st.session_state.get(f"{name}_cslr_all", True)
    cslr_sel = st.session_state.get(f"{name}_cslr_ms", [])

    mask = (
        in_filter(df["Pipeline"], pipe_all,  pipe_sel) &
        in_filter(df["JetLearn Deal Source"], src_all,   src_sel) &
        in_filter(df["Country"], ctry_all,  ctry_sel) &
        in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
    )
    base = df[mask].copy()

    measures = st.session_state.get(f"{name}_measures", [])
    mode = st.session_state.get(f"{name}_mode", "Both")
    mtd = mode in ("MTD","Both")
    cohort = mode in ("Cohort","Both")

    # derive ranges & grains from state
    def fetch_range(kind):
        preset  = st.session_state.get(f"{name}_{kind}_preset","This month so far")
        grain   = st.session_state.get(f"{name}_{kind}_grain","Month")
        if kind == "mtd":
            series = base["Create Date"]
        else:
            series = base[measures[0]] if measures else base["Create Date"]
        # compute the actual range using the same logic as the widget function
        if preset == "Today":               f,t = today_bounds()
        elif preset == "This month so far": f,t = this_month_so_far_bounds()
        elif preset == "Last month":        f,t = last_month_bounds()
        elif preset == "Last quarter":      f,t = last_quarter_bounds()
        elif preset == "This year":         f,t = this_year_so_far_bounds()
        else:
            dmin,dmax = safe_minmax_date(series)
            custom = st.session_state.get(f"{name}_{kind}_custom", (dmin,dmax))
            f,t = custom if isinstance(custom,(tuple,list)) and len(custom)==2 else (dmin,dmax)
        return f,t,grain

    mtd_from=mtd_to=mtd_grain=None
    coh_from=coh_to=coh_grain=None
    if mtd:    mtd_from, mtd_to, mtd_grain = fetch_range("mtd")
    if cohort: coh_from, coh_to, coh_grain = fetch_range("coh")

    base = ensure_month_cols(base, measures)

    return dict(
        name=name, base=base, measures=measures, mtd=mtd, cohort=cohort,
        mtd_from=mtd_from, mtd_to=mtd_to, coh_from=coh_from, coh_to=coh_to,
        mtd_grain=mtd_grain, coh_grain=coh_grain,
        split_dims=st.session_state.get(f"{name}_split", []),
        show_top_countries=st.session_state.get(f"{name}_top_ctry", True),
        show_top_sources=st.session_state.get(f"{name}_top_src", True),
        show_top_counsellors=st.session_state.get(f"{name}_top_cslr", False),
        show_combo_pairs=st.session_state.get(f"{name}_pair", False),
        pipe_all=pipe_all, pipe_sel=pipe_sel, src_all=src_all, src_sel=src_sel,
        ctry_all=ctry_all, ctry_sel=ctry_sel, cslr_all=cslr_all, cslr_sel=cslr_sel
    )

metaA = assemble_meta("A", df)
metaB = assemble_meta("B", df)

# ---------- Compute ----------
with st.spinner("Crunching numbers…"):
    metricsA, tablesA, chartsA = compute_outputs(metaA)
    metricsB, tablesB, chartsB = compute_outputs(metaB)

# ---------- Main tabs ----------
tabA, tabB, tabC = st.tabs(["Scenario A", "Scenario B", "Compare"])

with tabA:
    st.markdown("<div class='section-title'>📌 KPI Overview — A</div>", unsafe_allow_html=True)
    dfA = pd.DataFrame(metricsA); kpi_grid(dfA, "A · ")

    st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — A</div>", unsafe_allow_html=True)
    if not tablesA: st.info("No tables — open Filters and enable splits/leaderboards.")
    else:
        for name,frame in tablesA.items():
            st.subheader(name); st.dataframe(frame, use_container_width=True)
            st.download_button("Download CSV — " + name, to_csv_bytes(frame),
                               file_name=f"A_{name.replace(' ','_')}.csv", mime="text/csv")
    st.markdown("<div class='section-title'>📈 Trends — A</div>", unsafe_allow_html=True)
    if "MTD Trend" in chartsA: st.altair_chart(chartsA["MTD Trend"], use_container_width=True)
    if "Cohort Trend" in chartsA: st.altair_chart(chartsA["Cohort Trend"], use_container_width=True)

with tabB:
    st.markdown("<div class='section-title'>📌 KPI Overview — B</div>", unsafe_allow_html=True)
    dfB = pd.DataFrame(metricsB); kpi_grid(dfB, "B · ")

    st.markdown("<div class='section-title'>🧩 Splits & Leaderboards — B</div>", unsafe_allow_html=True)
    if not tablesB: st.info("No tables — open Filters and enable splits/leaderboards.")
    else:
        for name,frame in tablesB.items():
            st.subheader(name); st.dataframe(frame, use_container_width=True)
            st.download_button("Download CSV — " + name, to_csv_bytes(frame),
                               file_name=f"B_{name.replace(' ','_')}.csv", mime="text/csv")
    st.markdown("<div class='section-title'>📈 Trends — B</div>", unsafe_allow_html=True)
    if "MTD Trend" in chartsB: st.altair_chart(chartsB["MTD Trend"], use_container_width=True)
    if "Cohort Trend" in chartsB: st.altair_chart(chartsB["Cohort Trend"], use_container_width=True)

with tabC:
    st.markdown("<div class='section-title'>🧠 Smart Compare (A vs B)</div>", unsafe_allow_html=True)
    dfA = pd.DataFrame(metricsA); dfB = pd.DataFrame(metricsB)
    if dfA.empty or dfB.empty:
        st.info("Turn on KPIs for both scenarios to enable compare.")
    else:
        cmp = build_compare_delta(dfA, dfB)
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

# ---------- Foot ----------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.caption("**Scenario A** — " + mk_caption(metaA))
st.caption("**Scenario B** — " + mk_caption(metaB))
st.caption("Excluded globally: 1.2 Invalid Deal")
