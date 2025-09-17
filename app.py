# ============================== 
# ==== Predictability (ML) =====
# ==== Payment Count Target =====
# ==============================
with st.expander("🔮 Predictability (ML) — Payment Count (excludes running month)", expanded=True):

    # ---------- Inputs / column detection ----------
    # Expected feature columns (case-insensitive match)
    colmap = {c.lower(): c for c in df.columns}
    def pick(name_like, required=False):
        for k,v in colmap.items():
            if name_like.lower() == k:
                return v
        if required:
            st.error(f"Missing required column: '{name_like}'"); st.stop()
        return None

    COL_CREATE   = pick("create date", required=True)
    COL_PAY      = None
    # detect payment column by keywords
    for c in df.columns:
        cl = c.lower()
        if "payment" in cl and "received" in cl and "date" in cl:
            COL_PAY = c; break
    if COL_PAY is None:
        for c in df.columns:
            cl = c.lower()
            if "payment" in cl and "date" in cl:
                COL_PAY = c; break
    if COL_PAY is None:
        st.error("Couldn't find a Payment Received Date column."); st.stop()

    COL_COUNTRY  = pick("country")
    COL_SOURCE   = pick("jetlearn deal source")
    COL_CSL      = pick("student/academic counsellor")
    COL_TIMES    = None
    for c in df.columns:
        if "number of times contacted" in c.lower():
            COL_TIMES = c; break
    COL_SALES    = None
    for c in df.columns:
        if "number of sales activities" in c.lower():
            COL_SALES = c; break
    COL_LAST_ACT = None
    for c in df.columns:
        if "last activity date" in c.lower():
            COL_LAST_ACT = c; break
    COL_LAST_CNT = None
    for c in df.columns:
        if "last contacted" in c.lower():
            COL_LAST_CNT = c; break

    # ---------- Parse dates & basic coercions ----------
    X = df.copy()
    X[COL_CREATE] = pd.to_datetime(X[COL_CREATE], errors="coerce", dayfirst=True)
    X[COL_PAY]    = pd.to_datetime(X[COL_PAY],    errors="coerce", dayfirst=True)
    if COL_LAST_ACT is not None:
        X[COL_LAST_ACT] = pd.to_datetime(X[COL_LAST_ACT], errors="coerce", dayfirst=True)
    if COL_LAST_CNT is not None:
        X[COL_LAST_CNT] = pd.to_datetime(X[COL_LAST_CNT], errors="coerce", dayfirst=True)

    for num_col in [COL_TIMES, COL_SALES]:
        if num_col is not None:
            X[num_col] = pd.to_numeric(X[num_col], errors="coerce")

    # ---------- Leak-free cutoff (Asia/Kolkata) ----------
    tz = "Asia/Kolkata"
    now_ist = pd.Timestamp.now(tz=tz)
    cm_start = pd.Timestamp(year=now_ist.year, month=now_ist.month, day=1, tz=tz).tz_convert(None)

    # ---------- Controls ----------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        max_age_days = st.slider("Max age days (cap)", 30, 365, 150, 10,
                                 help="Cap days-since-create when building features.")
    with c2:
        neg_pos = st.slider("Negatives per positive (train)", 1, 20, 5, 1,
                            help="Subsample negative days to control dataset size.")
    with c3:
        lr = st.slider("Learning rate", 0.01, 0.5, 0.08, 0.01)
    with c4:
        n_estimators = st.slider("Estimators", 50, 800, 250, 50)

    gb1, gb2 = st.columns([2,3])
    with gb1:
        group_by = st.multiselect(
            "Breakdowns (group by)",
            options=[opt for opt in [
                "Student/Academic Counsellor" if COL_CSL else None,
                "JetLearn Deal Source" if COL_SOURCE else None,
                "Country" if COL_COUNTRY else None,
                "Day", "Day of Week"
            ] if opt is not None],
            default=[x for x in ["JetLearn Deal Source","Country"] if x in ["JetLearn Deal Source","Country"]],
            help="Choose how forecasts are summarized."
        )
    with gb2:
        st.caption("Target = expected **count of payments** per day; trained on all history **excluding the running month**.")

    # ---------- Helper: build training samples ----------
    @st.cache_data(show_spinner=False)
    def build_training(X: pd.DataFrame, cm_start: pd.Timestamp,
                       max_age_days: int, neg_pos_ratio: int) -> pd.DataFrame:
        """Return a training dataframe with columns:
           [deal_id, day, y, age, moy, dow, dom, country, source, counsellor,
            times, sales, rec_act, rec_cnt]"""
        D = X.copy()
        D = D[D[COL_CREATE].notna()].reset_index(drop=True)
        D["deal_id"] = np.arange(len(D))
        train_mask = ((D[COL_CREATE] < cm_start))
        # Positives: payments strictly before current month
        pos = D[D[COL_PAY].notna() & (D[COL_PAY] < cm_start)][
            ["deal_id", COL_CREATE, COL_PAY, COL_COUNTRY, COL_SOURCE, COL_CSL, COL_TIMES, COL_SALES, COL_LAST_ACT, COL_LAST_CNT]
        ].copy()
        pos.rename(columns={COL_PAY: "day"}, inplace=True)
        pos["y"] = 1.0

        # Negatives: sample non-payment days before cm_start
        rng = np.random.default_rng(42)
        neg_rows = []
        # For paid deals: sample days from [create .. min(pay-1, cm_start-1)]
        for _, r in pos.iterrows():
            d0 = pd.to_datetime(r[COL_CREATE])
            d1 = min(pd.Timestamp(r["day"]) - pd.Timedelta(days=1), cm_start - pd.Timedelta(days=1))
            if d1 < d0:
                continue
            span = (d1.date() - d0.date()).days + 1
            if span <= 0: 
                continue
            take = min(span, neg_pos_ratio)
            offs = rng.choice(span, size=take, replace=False)
            for o in offs:
                neg_rows.append({**r.to_dict(), "day": (d0 + pd.Timedelta(days=int(o))), "y": 0.0})

        # For unpaid-as-of-cutoff deals: sample days from [create .. cm_start-1]
        unpaid = D[(D[COL_PAY].isna()) & (D[COL_CREATE] < cm_start)][
            ["deal_id", COL_CREATE, COL_COUNTRY, COL_SOURCE, COL_CSL, COL_TIMES, COL_SALES, COL_LAST_ACT, COL_LAST_CNT]
        ]
        for _, r in unpaid.iterrows():
            d0 = pd.to_datetime(r[COL_CREATE])
            d1 = cm_start - pd.Timedelta(days=1)
            span = (d1.date() - d0.date()).days + 1
            if span <= 0: 
                continue
            take = min(span, neg_pos_ratio)
            offs = rng.choice(span, size=take, replace=False)
            for o in offs:
                neg_rows.append({**r.to_dict(), "day": (d0 + pd.Timedelta(days=int(o))), "y": 0.0})

        neg = pd.DataFrame(neg_rows) if neg_rows else pd.DataFrame(columns=pos.columns)

        train = pd.concat([pos, neg], ignore_index=True)
        if train.empty:
            return train

        # Feature engineering
        train["day"] = pd.to_datetime(train["day"]).dt.normalize()
        train["age"] = (train["day"] - pd.to_datetime(train[COL_CREATE]).dt.normalize()).dt.days.clip(lower=0, upper=max_age_days).astype(int)
        train["moy"] = pd.to_datetime(train["day"]).dt.month.astype(int)
        train["dow"] = pd.to_datetime(train["day"]).dt.dayofweek.astype(int)  # 0=Mon
        train["dom"] = pd.to_datetime(train["day"]).dt.day.astype(int)
        # recencies
        def recency(col):
            if col is None: return 9999
            d = pd.to_datetime(train[col], errors="coerce")
            return (train["day"] - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
        train["rec_act"] = recency(COL_LAST_ACT)
        train["rec_cnt"] = recency(COL_LAST_CNT)
        # numeric fills
        for c in [COL_TIMES, COL_SALES]:
            if c is not None:
                train[c] = pd.to_numeric(train[c], errors="coerce").fillna(0)
            else:
                train[c] = 0

        # trim columns
        keep = ["deal_id","day","y","age","moy","dow","dom","rec_act","rec_cnt",
                COL_COUNTRY, COL_SOURCE, COL_CSL, COL_TIMES, COL_SALES]
        keep = [k for k in keep if k is not None]
        return train[keep]

    # ---------- Build / cache training ----------
    with st.spinner("Preparing training set…"):
        train = build_training(X, cm_start, max_age_days=max_age_days, neg_pos_ratio=neg_pos)
    if train.empty:
        st.info("Not enough data to train (no historical payments before current month)."); st.stop()

    # ---------- Train model ----------
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Features
    num_cols = ["age","moy","dow","dom","rec_act","rec_cnt", COL_TIMES or "times_fallback", COL_SALES or "sales_fallback"]
    num_cols = [c for c in num_cols if c in train.columns]
    cat_cols = [c for c in [COL_COUNTRY, COL_SOURCE, COL_CSL] if c is not None]

    pre = ColumnTransformer(
        transformers=[
            ("num","passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=float(lr),
        max_depth=None,
        max_leaf_nodes=31,
        n_estimators=int(n_estimators),
        l2_regularization=0.0,
        early_stopping=True,
        random_state=42
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    with st.spinner("Training model (leak-free)…"):
        pipe.fit(train.drop(columns=["y","deal_id","day"], errors="ignore"), train["y"])

    # ---------- Build scoring frame (today..end of next month) ----------
    today = pd.Timestamp.now(tz=tz).tz_convert(None).normalize()
    end_this = (today.to_period("M").to_timestamp("M"))
    end_next = ((today.to_period("M") + 1).to_timestamp("M"))

    dates = pd.date_range(start=today, end=end_next, freq="D")

    # Active deals for a given day: created <= day and (unpaid before day)
    deals = X.copy()
    deals["deal_id"] = np.arange(len(deals))
    deals["Create"] = pd.to_datetime(deals[COL_CREATE]).dt.normalize()
    deals["Pay"]    = pd.to_datetime(deals[COL_PAY]).dt.normalize()

    # Cartesian product (deals x dates) but filter quickly with masks
    cart = deals[["deal_id","Create","Pay", COL_COUNTRY, COL_SOURCE, COL_CSL, COL_TIMES, COL_SALES, COL_LAST_ACT, COL_LAST_CNT]].assign(key=1)\
            .merge(pd.DataFrame({"day": dates, "key":1}), on="key").drop(columns=["key"])
    # keep rows where Create <= day and (Pay is NaT or Pay >= day)
    cart = cart[(cart["Create"] <= cart["day"]) & (cart["Pay"].isna() | (cart["Pay"] >= cart["day"]))].copy()

    # features
    cart["age"] = (cart["day"] - cart["Create"]).dt.days.clip(lower=0, upper=max_age_days).astype(int)
    cart["moy"] = cart["day"].dt.month.astype(int)
    cart["dow"] = cart["day"].dt.dayofweek.astype(int)
    cart["dom"] = cart["day"].dt.day.astype(int)

    def recency_score(col):
        if col not in cart.columns: 
            return 9999
        d = pd.to_datetime(cart[col], errors="coerce")
        return (cart["day"] - d).dt.days.clip(lower=0, upper=365).fillna(365).astype(int)
    cart["rec_act"] = recency_score(COL_LAST_ACT)
    cart["rec_cnt"] = recency_score(COL_LAST_CNT)

    for c in [COL_TIMES, COL_SALES]:
        if c in cart.columns:
            cart[c] = pd.to_numeric(cart[c], errors="coerce").fillna(0)
        else:
            cart[c] = 0

    # predict probabilities and treat as expected counts per deal-day
    with st.spinner("Scoring future days…"):
        proba = pipe.predict_proba(cart.drop(columns=["deal_id","Create","Pay","day"], errors="ignore"))[:,1]
    cart["p"] = proba

    # ---------- Aggregations ----------
    cart["Day"] = cart["day"].dt.date.astype(str)
    cart["Day of Week"] = cart["day"].dt.day_name()

    # Build grouping keys from UI
    key_cols = []
    lbl_map = {}
    if "Student/Academic Counsellor" in group_by and COL_CSL is not None:
        key_cols.append(COL_CSL); lbl_map[COL_CSL] = "Counsellor"
    if "JetLearn Deal Source" in group_by and COL_SOURCE is not None:
        key_cols.append(COL_SOURCE); lbl_map[COL_SOURCE] = "Deal Source"
    if "Country" in group_by and COL_COUNTRY is not None:
        key_cols.append(COL_COUNTRY); lbl_map[COL_COUNTRY] = "Country"
    if "Day" in group_by:
        key_cols.append("Day")
    if "Day of Week" in group_by:
        key_cols.append("Day of Week")

    # Helper to summarize any subset of days
    def summarize(mask, label):
        subset = cart.loc[mask]
        if len(key_cols)==0:
            val = subset["p"].sum()
            frame = pd.DataFrame({label:[int(round(val))]})
            return frame
        g = subset.groupby(key_cols, dropna=False)["p"].sum().reset_index()
        g.rename(columns=lbl_map, inplace=True)
        g[label] = g["p"].round(0).astype(int)
        return g.drop(columns=["p"])

    today_mask = (cart["day"] == today)
    tom_mask   = (cart["day"] == (today + pd.Timedelta(days=1)))
    month_mask = (cart["day"] >= today) & (cart["day"] <= end_this)
    next_mask  = (cart["day"] >= (end_this + pd.Timedelta(days=1))) & (cart["day"] <= end_next)

    g_today = summarize(today_mask, "Today")
    g_tom   = summarize(tom_mask, "Tomorrow")
    g_this  = summarize(month_mask, "This Month")
    g_next  = summarize(next_mask, "Next Month")

    # Merge all summaries on the grouping keys
    def smart_merge(a,b):
        common = [c for c in a.columns if c in b.columns and c not in ["Today","Tomorrow","This Month","Next Month"]]
        return a.merge(b, on=common, how="outer")

    out = g_today
    for part in [g_tom, g_this, g_next]:
        out = smart_merge(out, part)
    for col in ["Today","Tomorrow","This Month","Next Month"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    # ---------- Display ----------
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Today", f"{int(out['Today'].sum()) if 'Today' in out.columns else 0:,}")
    k2.metric("Tomorrow", f"{int(out['Tomorrow'].sum()) if 'Tomorrow' in out.columns else 0:,}")
    k3.metric("This Month", f"{int(out['This Month'].sum()) if 'This Month' in out.columns else 0:,}")
    k4.metric("Next Month", f"{int(out['Next Month'].sum()) if 'Next Month' in out.columns else 0:,}")

    st.markdown("#### Forecast breakdown")
    st.dataframe(out.sort_values(by=[c for c in ["Today","This Month","Next Month"] if c in out.columns], ascending=False),
                 use_container_width=True)
    st.download_button("Download — ML Predictability CSV",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="ml_predictability_payment_count.csv",
                       mime="text/csv")

    # Optional day-wise chart if 'Day' included
    if "Day" in group_by:
        try:
            import altair as alt
            long = out.melt(id_vars=[c for c in out.columns if c not in ["Today","Tomorrow","This Month","Next Month"]],
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

    st.caption("Training uses full history **excluding current month** (Asia/Kolkata). "
               "Model = HistGradientBoostingClassifier (log-loss) on deal-day samples with negative subsampling. "
               "Predictions are summed per day to get expected payment counts.")
