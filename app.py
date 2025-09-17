# app.py — JetLearn Insights + Predictability (robust & data-agnostic)
# Run: streamlit run app.py
pip install -U "streamlit==1.37.1" "pandas==2.2.2" "numpy==1.26.4" "altair==5.2.0" "scikit-learn==1.3.2"

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
:root{ --text:#0f172a; --muted:#64748b; --border: rgba(15,23,42,.10); --card:#fff; }
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.6rem; padding-bottom:.85rem; }
.head { position:sticky; top:0; z-index:50; display:flex; gap:10px; align-items:center;
  padding:10px 12px; background:#0b1220; color:#fff; border-radius:12px; margin-bottom:10px; }
.head .title { font-weight:800; font-size:1.02rem; margin-right:auto; }
.section-title { font-weight:800; margin:.25rem 0 .6rem; color:var(--text); }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.badge { display:inline-block; padding:2px 8px; font-size:.72rem; border-radius:999px; border:1px solid var(--border); background:#fff; }
.warn { background:#fff7ed; border:1px solid #fed7aa; padding:6px 8px; border-radius:8px; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]

# --------------------- Helpers ---------------------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try: return pd.read_csv(file_or_path, encoding=enc)
        except Exception: pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def best_parse(series):
    """Pick dd/mm or mm/dd automatically by maximizing valid parses."""
    s1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    s2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return s1 if s1.notna().sum() >= s2.notna().sum() else s2

def detect_measure_date_columns(df: pd.DataFrame, create_col: str):
    """Find date-like columns except the create_col; coerce to datetime."""
    date_like=[]
    for col in df.columns:
        if col == create_col: continue
        cl = str(col).lower()
        if any(k in cl for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum()>0:
                df[col] = parsed
                date_like.append(col)
    # Prefer Payment first if present
    for pref in ["Payment Received Date","Payment Date","Payment"]:
        if pref in date_like:
            date_like = [pref] + [c for c in date_like if c!=pref]
            break
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

def group_label_from_series(s: pd.Series, grain: str):
    s = pd.to_datetime(s, errors="coerce")
    if grain=="Day":  return s.dt.date.astype(str)
    if grain=="Week":
        iso=s.dt.isocalendar()
        return (iso['year'].astype(str)+"-W"+iso['week'].astype(str).str.zfill(2))
    return s.dt.to_period("M").astype(str)

def pick_by_alias(df, possible_names):
    lower = {str(c).lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower: return lower[name.lower()]
    for c in df.columns:
        cl = str(c).lower()
        if any(name.lower() in cl for name in possible_names):
            return c
    return None

def cap_categories(df, col, top_k=50):
    """Map infrequent categories to '__OTHER__' to keep OHE small & robust."""
    if col is None or col not in df.columns: return df, None
    s = df[col].astype(str).fillna("NA")
    vc = s.value_counts()
    keep = set(vc.head(top_k).index)
    new = str(col) + "_capped"
    df[new] = s.where(s.isin(keep), "__OTHER__")
    return df, new

def alt_line(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

# --------------------- Header ---------------------
st.markdown('<div class="head"><div class="title">📊 JetLearn — Insights & Predictability</div></div>', unsafe_allow_html=True)

# --------------------- Data Source ---------------------
ds = st.expander("Data source", expanded=True)
with ds:
    c1, c2, c3 = st.columns([3,2,2])
    with c1: uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with c2: path = st.text_input("…or CSV path", value="Master_sheet_DB_10percent.csv")
    with c3:
        exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal'", value=True)
        safe_mode = st.checkbox("Safe Mode (baseline if ML can’t train)", value=True)

try:
    if uploaded is not None:
        df = robust_read_csv(BytesIO(uploaded.getvalue()))
    else:
        df = robust_read_csv(path)
    st.success("Data loaded ✅")
except Exception as e:
    st.error(str(e)); st.stop()

df.columns = [str(c).strip() for c in df.columns]

# --- Column mapping (auto + pickers) ---
colmap = {str(c).lower(): c for c in df.columns}
def pick_exact(name_like): return colmap.get(name_like.lower())

COL_CREATE = pick_exact("create date") or pick_by_alias(df, ["Create Date","Created Date","Deal Create Date"])
# detect payment-like
COL_PAY = None
for c in df.columns:
    cl=str(c).lower()
    if "payment" in cl and "date" in cl:
        COL_PAY=c; break

if COL_CREATE is None:
    st.warning("Create Date not detected. Pick it below.")
    COL_CREATE = st.selectbox("Create Date column", options=df.columns)
if COL_PAY is None:
    st.warning("Payment Received Date not detected. Pick it below.")
    COL_PAY = st.selectbox("Payment Date column", options=df.columns)

COL_COUNTRY  = pick_exact("country") or pick_by_alias(df, ["Country","Country Name"])
COL_SOURCE   = pick_exact("jetlearn deal source") or pick_by_alias(df, ["JetLearn Deal Source","Deal Source","Source"])
COL_CSL      = pick_exact("student/academic counsellor") or pick_by_alias(df, ["Student/Academic Counsellor","Student/Academic Counselor","Academic Counsellor","Academic Counselor","Counsellor","Counselor"])
COL_TIMES    = pick_by_alias(df, ["Number of times contacted","Times Contacted","Times Contacted Count"])
COL_SALES    = pick_by_alias(df, ["Number of Sales Activities","Sales Activities"])
COL_LAST_ACT = pick_by_alias(df, ["Last Activity Date","Last Activity"])
COL_LAST_CNT = pick_by_alias(df, ["Last Contacted","Last Contacted Date"])

# Exclude invalid deals (optional)
if exclude_invalid and "Deal Stage" in df.columns:
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

# Parse dates robustly
df[COL_CREATE] = best_parse(df[COL_CREATE])
df[COL_PAY]    = best_parse(df[COL_PAY])
if COL_LAST_ACT and COL_LAST_ACT in df.columns: df[COL_LAST_ACT] = best_parse(df[COL_LAST_ACT])
if COL_LAST_CNT and COL_LAST_CNT in df.columns: df[COL_LAST_CNT] = best_parse(df[COL_LAST_CNT])

# Numeric coercion
if COL_TIMES and COL_TIMES in df.columns: df[COL_TIMES] = pd.to_numeric(df[COL_TIMES], errors="coerce").fillna(0)
if COL_SALES and COL_SALES in df.columns: df[COL_SALES] = pd.to_numeric(df[COL_SALES], errors="coerce").fillna(0)

# Date-like columns for Insights
df["Create_Month"] = df[COL_CREATE].dt.to_period("M")
date_like_cols = detect_measure_date_columns(df, COL_CREATE)

# IST cutoff
tz = "Asia/Kolkata"
now_ist = pd.Timestamp.now(tz=tz)
cm_start = pd.Timestamp(year=now_ist.year, month=now_ist.month, day=1, tz=tz).tz_convert(None)
today = pd.Timestamp.now(tz=tz).tz_convert(None).normalize()
end_this = today.to_period("M").to_timestamp("M")
end_next = (today.to_period("M") + 1).to_timestamp("M")

# --------------------- Quick Integrity ---------------------
with st.expander("🔎 Data integrity check", expanded=False):
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Create non-null:", int(df[COL_CREATE].notna().sum()))
    st.write("Payment non-null:", int(df[COL_PAY].notna().sum()))
    st.write("Payments before current month:", int(((df[COL_PAY].notna()) & (df[COL_PAY] < cm_start)).sum()))
    st.write("Min/Max Create:", str(df[COL_CREATE].min()), str(df[COL_CREATE].max()))
    st.write("Min/Max Payment:", str(df[COL_PAY].min()), str(df[COL_PAY].max()))

# --------------------- Tabs ---------------------
tab_insights, tab_predict = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictability"])

# =========================================================
# ===============  INSIGHTS (compact)  ====================
# =========================================================
with tab_insights:
    st.markdown("### MTD vs Cohort — Payment Received Date")
    # Filters (only if columns exist)
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
        all_key = f"{key_prefix}_all"; ms_key  = f"{key_prefix}_ms"
        header = f"{label}: " + (summary_label(options, True) if options else "—")
        with st.expander(header, expanded=False):
            c1,c2 = st.columns([1,3])
            all_flag = c1.checkbox("All", value=True, key=all_key, disabled=(len(options)==0))
            disabled = st.session_state.get(all_key, True) or (len(options)==0)
            _sel = st.multiselect(label, options=options, default=options, key=ms_key,
                                  placeholder=f"Type to search {label.lower()}…",
                                  label_visibility="collapsed", disabled=disabled)
        all_flag = bool(st.session_state.get(all_key, True)) if len(options)>0 else True
        selected = [v for v in coerce_list(st.session_state.get(ms_key, options)) if v in options] if len(options)>0 else []
        effective = options if all_flag else selected
        st.markdown(f"<div class='popcap'>{label}: {summary_label(effective, all_flag)}</div>", unsafe_allow_html=True)
        return all_flag, effective

    p_all, p_sel = unified_multifilter("Pipeline", df, "Pipeline", "ins_pipe")
    s_all, s_sel = unified_multifilter("Deal Source", df, "JetLearn Deal Source", "ins_src")
    c_all, c_sel = unified_multifilter("Country", df, "Country", "ins_cty")
    u_all, u_sel = unified_multifilter("Counsellor", df, "Student/Academic Counsellor", "ins_csl")

    mask = pd.Series(True, index=df.index)
    if "Pipeline" in df.columns: mask &= in_filter(df["Pipeline"], p_all, p_sel)
    if "JetLearn Deal Source" in df.columns: mask &= in_filter(df["JetLearn Deal Source"], s_all, s_sel)
    if "Country" in df.columns: mask &= in_filter(df["Country"], c_all, c_sel)
    if "Student/Academic Counsellor" in df.columns: mask &= in_filter(df["Student/Academic Counsellor"], u_all, u_sel)
    base = df[mask].copy()

    # Windows
    presets=["Today","This month so far","Last month","This year","Custom"]
    c1,c2 = st.columns([3,2])
    with c1:
        choice = st.radio("MTD Window", presets, horizontal=True, key="mtd_win")
    with c2:
        grain = st.radio("Granularity", ["Day","Week","Month"], horizontal=True, key="mtd_grain")

    def bounds(choice, series):
        t=pd.Timestamp.today().date()
        if choice=="Today": return t,t
        if choice=="This month so far": return t.replace(day=1),t
        if choice=="Last month":
            first_this=t.replace(day=1); last_prev=first_this - timedelta(days=1)
            return last_prev.replace(day=1), last_prev
        if choice=="This year": return date(t.year,1,1),t
        dmin,dmax=safe_minmax_date(series)
        rng=st.date_input("Custom range",(dmin,dmax),key="mtd_custom")
        return (rng if isinstance(rng,(tuple,list)) and len(rng)==2 else (dmin,dmax))

    mtd_from, mtd_to = bounds(choice, base[COL_CREATE])

    # MTD metrics (Payment within same Create_Month)
    sub = base[base[COL_CREATE].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")].copy()
    sub["Payment_Month"] = pd.to_datetime(sub[COL_PAY]).dt.to_period("M")
    same_month = sub[(sub[COL_PAY].notna()) & (sub["Payment_Month"]==sub["Create_Month"])]

    cols = st.columns(3)
    with cols[0]: st.markdown(f"<div class='kpi'><div class='label'>Creates in window</div><div class='value'>{len(sub):,}</div></div>", unsafe_allow_html=True)
    with cols[1]: st.markdown(f"<div class='kpi'><div class='label'>Payments in same month</div><div class='value'>{len(same_month):,}</div></div>", unsafe_allow_html=True)
    with cols[2]:
        conv = (len(same_month)/len(sub)*100) if len(sub)>0 else 0.0
        st.markdown(f"<div class='kpi'><div class='label'>Same-month %</div><div class='value'>{conv:.1f}%</div></div>", unsafe_allow_html=True)

    # Simple trend of payments (by chosen grain)
    trend = base[base[COL_PAY].notna()].copy()
    if not trend.empty:
        trend["Bucket"] = group_label_from_series(trend[COL_PAY], grain)
        t = trend.groupby("Bucket")[COL_PAY].count().reset_index(name="Payments")
        try:
            st.altair_chart(alt_line(t, "Bucket:O", "Payments:Q", tooltip=["Bucket","Payments"]), use_container_width=True)
        except Exception:
            st.line_chart(t.set_index("Bucket")["Payments"])

# =========================================================
# ==============  PREDICTABILITY (robust)  =================
# =========================================================
with tab_predict:
    st.markdown("### 🔮 Predictability — Count of Payments (leak-free)")
    st.caption("Trains on **all history strictly before the running month (IST)**. If data isn’t enough, it falls back to a baseline but still runs.")

    # Feature selection
    available_cats = { "Country": COL_COUNTRY, "JetLearn Deal Source": COL_SOURCE, "Student/Academic Counsellor": COL_CSL }
    available_nums = { "Number of times contacted": COL_TIMES, "Number of Sales Activities": COL_SALES }
    available_dates= { "Last Activity Date": COL_LAST_ACT, "Last Contacted": COL_LAST_CNT }

    use_cat = st.multiselect("Categorical", [k for k,v in available_cats.items() if v is not None],
                             default=[k for k,v in available_cats.items() if v is not None])
    use_num = st.multiselect("Numeric", [k for k,v in available_nums.items() if v is not None],
                             default=[k for k,v in available_nums.items() if v is not None])
    use_dt  = st.multiselect("Date/Recency", [k for k,v in available_dates.items() if v is not None],
                             default=[k for k,v in available_dates.items() if v is not None])

    include_inflow = st.toggle("Include new-deal inflow (M0/M1)", value=True)
    fast_mode = st.toggle("Fast mode (only This Month + Tomorrow)", value=False)
    if fast_mode: end_next = end_this

    # Prepare dataset
    X = df.copy()
    X["Create"] = pd.to_datetime(X[COL_CREATE]).dt.normalize()
    X["Pay"]    = pd.to_datetime(X[COL_PAY]).dt.normalize()

    # Cap categories
    cap_map = {}
    for label in use_cat:
        base_col = available_cats[label]
        X, capped = cap_categories(X, base_col, top_k=50)
        cap_map[label] = capped

    # Build training data (positives + sampled negatives), excluding current month
    NEG_PER_POS = 5
    MAX_AGE_DAYS = 150

    def build_training(X_: pd.DataFrame) -> pd.DataFrame:
        D = X_[X_["Create"].notna()].copy().reset_index(drop=True)
        if D.empty: return pd.DataFrame()
        D["deal_id"] = np.arange(len(D))

        base_cols = ["deal_id","Create"]
        for label in use_cat:
            if cap_map.get(label): base_cols.append(cap_map[label])
        for label in use_num:
            col = available_nums[label]
            if col and col in D.columns: base_cols.append(col)
        for label in use_dt:
            col = available_dates[label]
            if col and col in D.columns: base_cols.append(col)

        pos_cols = base_cols + ["Pay"]
        pos = D[D["Pay"].notna() & (D["Pay"] < cm_start)][pos_cols].copy()
        if pos.empty: return pd.DataFrame()
        pos.rename(columns={"Pay":"day"}, inplace=True); pos["y"]=1.0

        rng = np.random.default_rng(42); neg_rows = []

        # negatives from paid deals pre-pay & < cutoff
        for _, r in pos.iterrows():
            d0 = pd.to_datetime(r["Create"]).normalize()
            dp = pd.to_datetime(r["day"]).normalize()
            d1 = min(dp - pd.Timedelta(days=1), cm_start - pd.Timedelta(days=1))
            if d1 < d0: continue
            span = (d1.date() - d0.date()).days + 1
            take = int(min(span, NEG_PER_POS))
            if take <= 0: continue
            offs = rng.choice(span, size=take, replace=False)
            rr=r.to_dict()
            for o in offs:
                row=dict(rr); row["day"]=(d0 + pd.Timedelta(days=int(o))); row["y"]=0.0; neg_rows.append(row)

        # negatives from unpaid deals as of cutoff
        unpaid = D[D["Pay"].isna() & (D["Create"] < cm_start)][base_cols].copy()
        for _, r in unpaid.iterrows():
            d0 = pd.to_datetime(r["Create"]).normalize()
            d1 = cm_start - pd.Timedelta(days=1)
            if d1 < d0: continue
            span = (d1.date() - d0.date()).days + 1
            take = int(min(span, NEG_PER_POS))
            if take <= 0: continue
            offs = rng.choice(span, size=take, replace=False)
            rr=r.to_dict()
            for o in offs:
                row=dict(rr); row["day"]=(d0 + pd.Timedelta(days=int(o))); row["y"]=0.0; neg_rows.append(row)

        neg = pd.DataFrame(neg_rows) if neg_rows else pd.DataFrame(columns=pos.columns)
        return pd.concat([pos, neg], ignore_index=True)

    reason = []
    model_ok = False
    try:
        train = build_training(X)
        if train.empty:
            reason.append("No train rows (need payments before current month).")
            raise RuntimeError("empty-train")

        # features
        train["day"] = pd.to_datetime(train["day"]).dt.normalize()
        train["age"] = (train["day"] - pd.to_datetime(train["Create"]).dt.normalize()).dt.days.clip(lower=0, upper=MAX_AGE_DAYS).astype(int)
        train["moy"] = train["day"].dt.month.astype(int)
        train["dow"] = train["day"].dt.dayofweek.astype(int)
        train["dom"] = train["day"].dt.day.astype(int)

        def recency(colname):
            if (colname is None) or (colname not in train.columns): return 365
            d = pd.to_datetime(train[colname], errors="coerce")
            return (train["day"] - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        # chosen recencies
        act_col = available_dates.get("Last Activity Date")
        cnt_col = available_dates.get("Last Contacted")
        train["rec_act"] = recency(act_col)
        train["rec_cnt"] = recency(cnt_col)

        if available_nums.get("Number of times contacted") in train.columns:
            train["times"] = pd.to_numeric(train[available_nums["Number of times contacted"]], errors="coerce").fillna(0)
        else: train["times"]=0
        if available_nums.get("Number of Sales Activities") in train.columns:
            train["sales"] = pd.to_numeric(train[available_nums["Number of Sales Activities"]], errors="coerce").fillna(0)
        else: train["sales"]=0

        num_cols = [c for c in ["age","moy","dow","dom","rec_act","rec_cnt","times","sales"] if c in train.columns]
        cat_cols = []
        for label in use_cat:
            col = cap_map.get(label)
            if col and col in train.columns: cat_cols.append(col)

        # model
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import HistGradientBoostingClassifier

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        pre = ColumnTransformer([("num","passthrough", num_cols), ("cat", ohe, cat_cols)], remainder="drop")
        clf = HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=31, max_iter=250, early_stopping=True, random_state=42)
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        pos_count = int((train["y"] == 1).sum()); neg_count = int((train["y"] == 0).sum())
        if pos_count == 0 or neg_count == 0:
            reason.append(f"One-class training (pos={pos_count}, neg={neg_count}).")
            raise RuntimeError("one-class")

        pipe.fit(train[num_cols + cat_cols], train["y"])
        model_ok = True
    except Exception as e:
        if not reason: reason = [str(e)]

    # ---------- Scoring existing open deals per day ----------
    date_range = pd.date_range(start=today, end=end_next, freq="D")
    deals = X.copy(); deals["deal_id"] = np.arange(len(deals))
    rows = []
    for day in date_range:
        active = deals[(deals["Create"] <= day) & (deals["Pay"].isna() | (deals["Pay"] >= day))].copy()
        if active.empty: continue
        F = active.copy()
        F["age"] = (day - F["Create"]).dt.days.clip(lower=0, upper=MAX_AGE_DAYS).astype(int)
        F["moy"] = day.month; F["dow"] = day.dayofweek; F["dom"] = day.day

        # recencies for scoring
        if COL_LAST_ACT and COL_LAST_ACT in F.columns:
            d = pd.to_datetime(F[COL_LAST_ACT], errors="coerce")
            F["rec_act"] = (day - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        else: F["rec_act"]=365
        if COL_LAST_CNT and COL_LAST_CNT in F.columns:
            d = pd.to_datetime(F[COL_LAST_CNT], errors="coerce")
            F["rec_cnt"] = (day - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        else: F["rec_cnt"]=365

        if COL_TIMES and COL_TIMES in F.columns:
            F["times"] = pd.to_numeric(F[COL_TIMES], errors="coerce").fillna(0)
        else: F["times"]=0
        if COL_SALES and COL_SALES in F.columns:
            F["sales"] = pd.to_numeric(F[COL_SALES], errors="coerce").fillna(0)
        else: F["sales"]=0

        # ensure capped cat columns exist
        for label in use_cat:
            col = cap_map.get(label)
            if col and col not in F.columns: F[col] = "__OTHER__"

        num_cols_score = [c for c in ["age","moy","dow","dom","rec_act","rec_cnt","times","sales"] if c in F.columns]
        cat_cols_score = [cap_map.get(label) for label in use_cat if cap_map.get(label) in F.columns]

        if model_ok:
            feats = F[num_cols_score + cat_cols_score] if (num_cols_score or cat_cols_score) else pd.DataFrame(index=F.index)
            p = np.clip(pipe.predict_proba(feats)[:,1], 0, 1)
        else:
            # Baseline: average daily payments historically / active deals
            hist = X[(X["Pay"].notna()) & (X["Pay"] < cm_start)]
            if hist.empty:
                p = np.zeros(len(F))
                if "No historical payments before current month." not in reason:
                    reason.append("No historical payments before current month.")
            else:
                per_day = hist["Pay"].dt.date.value_counts().mean()
                base_p = float(per_day) / max(1, len(F))
                p = np.full(len(F), max(0.0, min(0.2, base_p)))
        rows.append(pd.DataFrame({"Day": str(day.date()), "Day of Week": day.day_name(), "p": p}))
    pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Day","Day of Week","p"])

    # ---------- New-deal inflow (global M0/M1) ----------
    inflow = pd.DataFrame(columns=["Day","p"])
    if include_inflow:
        try:
            lm_start = (cm_start - pd.offsets.MonthBegin(1))
            lm_end   = (cm_start - pd.Timedelta(days=1))
            lastm = X[(X["Create"] >= lm_start) & (X["Create"] <= lm_end)].copy()

            def month_code(ts): return ts.dt.year * 12 + ts.dt.month
            dfC = X.copy(); dfC["Create_M"] = dfC["Create"].dt.to_period("M").dt.to_timestamp()
            dfC["Pay_M"] = dfC["Pay"].dt.to_period("M").dt.to_timestamp()
            hist = dfC[dfC["Create_M"].notna() & (dfC["Create_M"] < cm_start)]

            if not hist.empty:
                cmc = month_code(hist["Create_M"]); pmc = month_code(hist["Pay_M"])
                lag = (pmc - cmc)
                g_r0 = (lag==0).mean()
                g_r1 = (lag==1).mean()
            else:
                g_r0 = g_r1 = 0.0

            if not lastm.empty:
                days_last_month = pd.date_range(lm_start, lm_end, freq="D")
                dow_counts = pd.Series(days_last_month.dayofweek).value_counts().to_dict()
                for d in range(7): dow_counts.setdefault(d, 0)
                lastm["dow"] = lastm["Create"].dt.dayofweek
                creates_dow = lastm.groupby("dow")["Create"].count().rename("creates").reset_index()
                creates_dow["rate_per_day"] = creates_dow.apply(lambda r: (r["creates"] / max(1, dow_counts[int(r["dow"])])), axis=1)

                future_days = pd.date_range(start=today, end=end_next, freq="D")
                E = pd.DataFrame({"Day":[str(d.date()) for d in future_days], "dow":[d.dayofweek for d in future_days]})
                E = E.merge(creates_dow[["dow","rate_per_day"]], how="left", on="dow").fillna({"rate_per_day":0.0})

                days_this = [str(d.date()) for d in pd.date_range(start=today, end=end_this, freq="D")]
                days_next = [str(d.date()) for d in pd.date_range(start=end_this+pd.Timedelta(days=1), end=end_next, freq="D")]
                n_this = max(1, len(days_this)); n_next = max(1, len(days_next))

                m0_total  = E[E["Day"].isin(days_this)]["rate_per_day"].sum() * g_r0
                m1_total  = E[E["Day"].isin(days_this)]["rate_per_day"].sum() * g_r1
                m0n_total = E[E["Day"].isin(days_next)]["rate_per_day"].sum() * g_r0

                inflow_rows=[]
                if n_this>0:
                    for d in days_this: inflow_rows.append((d, m0_total / n_this))
                if (not fast_mode) and n_next>0:
                    for d in days_next: inflow_rows.append((d, m1_total / n_next))
                if n_next>0:
                    for d in days_next: inflow_rows.append((d, m0n_total / n_next))
                inflow = pd.DataFrame(inflow_rows, columns=["Day","p"])
        except Exception as e:
            st.markdown(f"<div class='warn'>Inflow disabled: {str(e)}</div>", unsafe_allow_html=True)

    # Combine (existing + inflow)
    pred_all = pred.copy()
    if not inflow.empty:
        pred_all = pd.concat([pred_all, inflow.assign(**{"Day of Week": pd.to_datetime(inflow["Day"]).dt.day_name()})], ignore_index=True)

    # Grouping options (optional)
    group_opts = ["Day","Day of Week"]
    if COL_SOURCE: group_opts.append("JetLearn Deal Source")
    if COL_COUNTRY: group_opts.append("Country")
    if COL_CSL: group_opts.append("Student/Academic Counsellor")
    group_by = st.multiselect("Breakdowns (group by)", options=group_opts, default=["Day"])

    # Summaries
    if pred_all.empty:
        st.info("No active deals to score in the forecast window.")
    else:
        by = group_by if group_by else ["Day"]
        out = pred_all.groupby(by, dropna=False)["p"].sum().reset_index()
        # derive totals
        today_s = str(today.date()); tom_s = str((today+pd.Timedelta(days=1)).date())
        end_m_s = str(end_this.date()); end_n_s = str(end_next.date())

        def sum_mask(df_, f, t=None):
            if t is None:
                sub = df_[df_["Day"]==f]
            else:
                sub = df_[(df_["Day"]>=f) & (df_["Day"]<=t)]
            if group_by:
                g = sub.groupby(by)["p"].sum().reset_index()
                return g
            return pd.DataFrame({"p":[float(sub["p"].sum())]})

        base_cols = by + ["p"] if group_by else ["p"]
        g_today = sum_mask(pred_all, today_s).rename(columns={"p":"Today"})
        g_tom   = sum_mask(pred_all, tom_s).rename(columns={"p":"Tomorrow"})
        g_m     = sum_mask(pred_all, today_s, end_m_s).rename(columns={"p":"This Month"})
        g_n     = sum_mask(pred_all, str((end_this+pd.Timedelta(days=1)).date()), end_n_s).rename(columns={"p":"Next Month"})

        def smart_merge(a,b):
            common = [c for c in a.columns if c in b.columns and c not in ["Today","Tomorrow","This Month","Next Month","p"]]
            return a.merge(b, on=common, how="outer")

        outm = g_today
        for part in [g_tom, g_m, g_n]:
            outm = smart_merge(outm, part)

        for c in ["Today","Tomorrow","This Month","Next Month"]:
            if c in outm.columns: outm[c] = outm[c].fillna(0).round(0).astype(int)

        # KPIs (totals)
        tot_today   = int(outm["Today"].sum()) if "Today" in outm.columns else 0
        tot_tom     = int(outm["Tomorrow"].sum()) if "Tomorrow" in outm.columns else 0
        tot_this    = int(outm["This Month"].sum()) if "This Month" in outm.columns else 0
        tot_next    = int(outm["Next Month"].sum()) if "Next Month" in outm.columns else 0

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Today", f"{tot_today:,}")
        k2.metric("Tomorrow", f"{tot_tom:,}")
        k3.metric("This Month", f"{tot_this:,}")
        k4.metric("Next Month", f"{tot_next:,}")

        # Table
        order_cols = [c for c in ["Day","Day of Week","JetLearn Deal Source","Country","Student/Academic Counsellor","Today","Tomorrow","This Month","Next Month"] if c in outm.columns]
        st.markdown("#### Forecast breakdown")
        st.dataframe(outm[order_cols].sort_values(by=[c for c in ["Today","This Month","Next Month"] if c in outm.columns], ascending=False), use_container_width=True)
        st.download_button("Download — Predictability CSV", outm[order_cols].to_csv(index=False).encode("utf-8"),
                           file_name="predictability_payment_count.csv", mime="text/csv")

        # Day chart
        try:
            day_agg = pred_all.groupby(["Day"])["p"].sum().reset_index()
            day_agg["Count"] = day_agg["p"].round(0).astype(int)
            st.altair_chart(
                alt.Chart(day_agg).mark_bar().encode(
                    x=alt.X("Day:N", sort=None, title=None),
                    y=alt.Y("Count:Q", title=None),
                    tooltip=["Day","Count"]
                ).properties(height=260),
                use_container_width=True
            )
        except Exception:
            pass

    if not model_ok:
        st.markdown("<div class='warn'><b>ML fallback active</b>: " + "; ".join(reason) + "</div>", unsafe_allow_html=True)
    elif safe_mode:
        st.caption("ML trained successfully. Safe Mode will still fallback to baseline if a swapped CSV is tricky.")
