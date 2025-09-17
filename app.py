# Drop-in replacement for the entire `with tab_predict:` block
# This replaces the old Poisson time-series forecast with a cohort-based conversion model
# that estimates M0 (same-month) and M1 (next-month) conversions by Deal Source × Country × Counsellor,
# with empirical-Bayes smoothing and daily/hourly distribution for Today/Tomorrow.

with tab_predict:
    # ----------------------------
    # Utilities specific to cohort forecast
    # ----------------------------
    def detect_payment_col(cols):
        for c in cols:
            cl = c.lower()
            if "payment" in cl and "received" in cl and "date" in cl:
                return c
        for c in cols:
            cl = c.lower()
            if "payment" in cl and "date" in cl:
                return c
        return None

    def to_month(dtser: pd.Series) -> pd.Series:
        return pd.to_datetime(dtser, errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()

    def month_add(ts: pd.Timestamp, k: int) -> pd.Timestamp:
        return (ts.to_period("M") + k).to_timestamp()

    def month_start(d: date) -> pd.Timestamp:
        return pd.Timestamp(d).to_period("M").to_timestamp()

    def month_days(ts: pd.Timestamp) -> int:
        m0 = ts
        m1 = (m0 + pd.offsets.MonthBegin(1))
        return int((m1 - m0).days)

    def safe_div(a, b):
        return (a / b) if b else 0.0

    # Empirical-Bayes smoothing for binomial rate
    def eb_rate(success: float, trials: float, prior_rate: float, prior_strength: float) -> float:
        return (success + prior_strength * prior_rate) / (trials + prior_strength) if (trials + prior_strength) > 0 else prior_rate

    # Day-of-month profile with EB smoothing and seasonal pooling (same month-of-year preferred)
    def day_of_month_profile(df_paid: pd.DataFrame, target_month: pd.Timestamp, group_mask: pd.Series, prior_strength: float = 5.0) -> pd.Series:
        dfp = df_paid.loc[group_mask].copy()
        if dfp.empty:
            days = month_days(target_month)
            return pd.Series(1.0 / days, index=pd.Index(range(1, days + 1), name="day"))
        dfp["d"] = dfp["Payment Received Date"].dt.day
        dfp["moy"] = dfp["Payment Received Date"].dt.month
        moy = int(target_month.month)
        pool = dfp[dfp["moy"] == moy]
        if pool.empty:
            pool = dfp
        # Raw counts by day 1..N
        days = month_days(target_month)
        idx = pd.Index(range(1, days + 1), name="day")
        cnt = pool["d"].value_counts().reindex(idx, fill_value=0).astype(float)
        total = cnt.sum()
        # Uniform prior
        prior = pd.Series(1.0 / days, index=idx)
        smoothed = (cnt + prior_strength * prior) / (total + prior_strength)
        return smoothed / smoothed.sum()

    # Hour-of-day profile (0..23) with EB smoothing; used to split Today's prediction by hour
    def hour_of_day_profile(df_paid: pd.DataFrame, group_mask: pd.Series, prior_strength: float = 10.0) -> pd.Series:
        dfp = df_paid.loc[group_mask].copy()
        if dfp.empty or dfp["Payment Received Date"].isna().all():
            return pd.Series(1/24, index=pd.Index(range(24), name="hour"))
        hrs = dfp["Payment Received Date"].dt.hour
        idx = pd.Index(range(24), name="hour")
        cnt = hrs.value_counts().reindex(idx, fill_value=0).astype(float)
        total = cnt.sum()
        prior = pd.Series(1/24, index=idx)
        smoothed = (cnt + prior_strength * prior) / (total + prior_strength)
        return smoothed / smoothed.sum()

    # Backoff helpers for priors
    def compute_priors(df_cre: pd.DataFrame, df_paid_cohort: pd.DataFrame):
        """
        Returns dictionaries with prior rates for:
        - global
        - by Deal Source
        - by (Deal Source, Country)
        Keys:
          priors_global = {"r0": float, "r1": float, "n": int}
          priors_src[(src)] = {"r0": float, "r1": float, "n": int}
          priors_src_cty[(src, cty)] = {"r0": float, "r1": float, "n": int}
        """
        # df_cre: rows = deals with Create_Month set; one row per deal
        # df_paid_cohort: df_cre merged with PaymentMonth + Lag, one row per deal (payments may be NaT)
        # Global
        g_trials = len(df_cre)
        g_succ0 = int(((df_paid_cohort["Lag"] == 0)).sum())
        g_succ1 = int(((df_paid_cohort["Lag"] == 1)).sum())
        priors_global = {"r0": safe_div(g_succ0, g_trials), "r1": safe_div(g_succ1, g_trials), "n": g_trials}

        # By source
        priors_src = {}
        if "JetLearn Deal Source" in df_cre.columns:
            for src, grp in df_cre.groupby("JetLearn Deal Source", dropna=False):
                gidx = grp.index
                sub = df_paid_cohort.loc[gidx]
                trials = len(grp)
                succ0 = int(((sub["Lag"] == 0)).sum())
                succ1 = int(((sub["Lag"] == 1)).sum())
                priors_src[src] = {"r0": safe_div(succ0, trials), "r1": safe_div(succ1, trials), "n": trials}

        # By source×country
        priors_src_cty = {}
        if {"JetLearn Deal Source", "Country"}.issubset(df_cre.columns):
            for (src, cty), grp in df_cre.groupby(["JetLearn Deal Source", "Country"], dropna=False):
                gidx = grp.index
                sub = df_paid_cohort.loc[gidx]
                trials = len(grp)
                succ0 = int(((sub["Lag"] == 0)).sum())
                succ1 = int(((sub["Lag"] == 1)).sum())
                priors_src_cty[(src, cty)] = {"r0": safe_div(succ0, trials), "r1": safe_div(succ1, trials), "n": trials}

        return priors_global, priors_src, priors_src_cty

    def estimate_group_rates(df_cre: pd.DataFrame, df_paid_cohort: pd.DataFrame,
                             lookback_months: int, prior_strength_base: float = 25.0,
                             min_trials_for_local: int = 30) -> pd.DataFrame:
        """
        Compute EB-smoothed M0/M1 rates per finest group (Source×Country×Counsellor) with backoff priors.
        """
        # Restrict to lookback window based on Create_Month
        last_month = df_cre["Create_Month"].max()
        if pd.isna(last_month):
            return pd.DataFrame(columns=["JetLearn Deal Source","Country","Student/Academic Counsellor","r0","r1","trials"])  # empty
        first_lb = month_add(last_month, -lookback_months + 1)
        mask_lb = (df_cre["Create_Month"] >= first_lb) & (df_cre["Create_Month"] <= last_month)
        cre_lb = df_cre.loc[mask_lb]
        paid_lb = df_paid_cohort.loc[cre_lb.index]

        # Priors
        pg, psrc, psrccty = compute_priors(cre_lb, paid_lb)

        recs = []
        grp_cols = ["JetLearn Deal Source", "Country", "Student/Academic Counsellor"]
        for keys, grp in cre_lb.groupby(grp_cols, dropna=False):
            gidx = grp.index
            sub = paid_lb.loc[gidx]
            trials = len(grp)
            succ0 = int(((sub["Lag"] == 0)).sum())
            succ1 = int(((sub["Lag"] == 1)).sum())

            # Choose prior hierarchy: src×cty -> src -> global
            src = keys[0]
            cty = keys[1]
            prior_r0, prior_r1, prior_n = pg["r0"], pg["r1"], max(pg["n"], 1)
            if (src, cty) in psrccty and psrccty[(src, cty)]["n"] >= 10:
                pr = psrccty[(src, cty)]
                prior_r0, prior_r1, prior_n = pr["r0"], pr["r1"], pr["n"]
            elif src in psrc and psrc[src]["n"] >= 10:
                pr = psrc[src]
                prior_r0, prior_r1, prior_n = pr["r0"], pr["r1"], pr["n"]

            # Scale prior strength: stronger prior when local trials are small
            prior_strength = prior_strength_base * (1.0 if trials < min_trials_for_local else 0.4)

            r0 = eb_rate(succ0, trials, prior_r0, prior_strength)
            r1 = eb_rate(succ1, trials, prior_r1, prior_strength)

            recs.append({
                "JetLearn Deal Source": keys[0],
                "Country": keys[1],
                "Student/Academic Counsellor": keys[2],
                "r0": float(r0),
                "r1": float(r1),
                "trials": int(trials)
            })

        rates = pd.DataFrame(recs)
        return rates

    def forecast_month_groupwise(df: pd.DataFrame, df_paid: pd.DataFrame, rates: pd.DataFrame,
                                 target_month: pd.Timestamp) -> pd.DataFrame:
        """
        Forecast payments for target_month from:
          - M0 conversions of deals created in target_month, and
          - M1 conversions of deals created in previous month.
        Returns a table at finest granularity (Source×Country×Counsellor) with columns:
          [Deal Source, Country, Counsellor, Forecast]
        """
        prev_month = month_add(target_month, -1)
        # Count creates this and prev month at finest level
        grp_cols = ["JetLearn Deal Source", "Country", "Student/Academic Counsellor"]
        dfc = df.copy()
        dfc["Create_Month"] = to_month(dfc["Create Date"])  # already present but ensure

        cur_cre = dfc[dfc["Create_Month"] == target_month].groupby(grp_cols, dropna=False)["Create Date"].count().rename("C_cur")
        prev_cre = dfc[dfc["Create_Month"] == prev_month].groupby(grp_cols, dropna=False)["Create Date"].count().rename("C_prev")
        base = pd.concat([cur_cre, prev_cre], axis=1).fillna(0)

        out = base.reset_index().merge(rates, on=grp_cols, how="left")
        out[["r0","r1"]] = out[["r0","r1"]].fillna(out[["r0","r1"]].median().fillna(0))
        out["Forecast"] = out["r0"] * out["C_cur"] + out["r1"] * out["C_prev"]
        out["Forecast"] = out["Forecast"].clip(lower=0)
        out["Month"] = target_month
        return out[grp_cols + ["C_cur","C_prev","r0","r1","Forecast","Month"]]

    # ----------------------------
    # Detect payments and prep cohort frame
    # ----------------------------
    PAYMENT_COL = detect_payment_col(df.columns)
    if PAYMENT_COL is None:
        st.error("Couldn't find a payment date column. Add one like 'Payment Received Date'.")
        st.stop()

    dfX = df.copy()
    dfX["Create Date"] = pd.to_datetime(dfX["Create Date"], errors="coerce", dayfirst=True)
    dfX["Create_Month"] = to_month(dfX["Create Date"])  # ensure set
    dfX["Payment Received Date"] = pd.to_datetime(dfX[PAYMENT_COL], errors="coerce", dayfirst=True)
    paid = dfX[dfX["Payment Received Date"].notna()].copy()
    paid["PaymentMonth"] = to_month(paid["Payment Received Date"])  

    if paid.empty:
        st.error("No non-empty payment dates found.")
        st.stop()

    # Cohort merge: align each deal with its payment month (if any) and compute Lag
    df_cohort = dfX[["Create Date","Create_Month","JetLearn Deal Source","Country","Student/Academic Counsellor"]].copy()
    df_cohort["PaymentMonth"] = to_month(dfX["Payment Received Date"])  # keeps NaT for non-paid
    # Lag in months: PM - CM (NaN -> large positive that won't match 0/1)
    cm_code = df_cohort["Create_Month"].dt.year * 12 + df_cohort["Create_Month"].dt.month
    pm_code = df_cohort["PaymentMonth"].dt.year * 12 + df_cohort["PaymentMonth"].dt.month
    df_cohort["Lag"] = (pm_code - cm_code)

    # ----------------------------
    # Controls
    # ----------------------------
    st.markdown("### Cohort-based Predictability")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lookback = st.slider("Lookback window (months)", 3, 12, 6, 1)
    with c2:
        min_trials = st.slider("Min. creates for local rates", 10, 100, 30, 5)
    with c3:
        prior_strength = st.slider("Prior strength", 5, 100, 25, 5)
    with c4:
        group_view = st.selectbox("View level", [
            "Deal Source",
            "Country",
            "Counsellor",
            "Deal Source × Country",
            "Deal Source × Country × Counsellor"
        ])

    today_dt = pd.Timestamp.today().date()
    cm = month_start(today_dt)
    nm = month_add(cm, 1)

    # ----------------------------
    # Estimate rates (r0/r1) with EB
    # ----------------------------
    with st.spinner("Estimating conversion rates by cohort…"):
        rates = estimate_group_rates(
            df_cre=df_cohort,
            df_paid_cohort=df_cohort,
            lookback_months=lookback,
            prior_strength_base=float(prior_strength),
            min_trials_for_local=int(min_trials)
        )

    # ----------------------------
    # Forecast this & next month at finest level
    # ----------------------------
    fc_this = forecast_month_groupwise(dfX, paid, rates, cm)
    fc_next = forecast_month_groupwise(dfX, paid, rates, nm)

    # Helper to aggregate to view level
    def aggregate_view(df_fc: pd.DataFrame, view: str) -> pd.DataFrame:
        if view == "Deal Source":
            keys = ["JetLearn Deal Source"]
        elif view == "Country":
            keys = ["Country"]
        elif view == "Counsellor":
            keys = ["Student/Academic Counsellor"]
        elif view == "Deal Source × Country":
            keys = ["JetLearn Deal Source","Country"]
        else:
            keys = ["JetLearn Deal Source","Country","Student/Academic Counsellor"]
        agg = df_fc.groupby(keys, dropna=False)["Forecast"].sum().reset_index()
        return agg

    v_this = aggregate_view(fc_this, group_view).rename(columns={"Forecast":"This Month"})
    v_next = aggregate_view(fc_next, group_view).rename(columns={"Forecast":"Next Month"})
    merged_months = v_this.merge(v_next, on=v_this.columns[:-1].tolist(), how="outer").fillna(0)

    # ----------------------------
    # Today / Tomorrow using day-of-month + hour-of-day profiles per group
    # ----------------------------
    # Build group masks for profiles only when requested (expensive)
    st.markdown("### Today & Tomorrow (day/time aware)")
    with st.expander("Daily & hourly allocation settings", expanded=False):
        daily_prior_strength = st.slider("Daily profile prior strength", 1, 20, 5, 1)
        hourly_prior_strength = st.slider("Hourly profile prior strength", 5, 50, 10, 5)

    # Compute day-of-month allocation per group in the selected view
    # We'll apportion the 'This Month' forecast to today and tomorrow
    dom_today = today_dt.day
    dom_tom = (today_dt + timedelta(days=1)).day

    # Function to get a boolean mask for paid rows matching a group row
    def mask_for_row(row: pd.Series) -> pd.Series:
        m = pd.Series(True, index=paid.index)
        if "JetLearn Deal Source" in row.index and not pd.isna(row["JetLearn Deal Source"]):
            m &= (paid["JetLearn Deal Source"].astype(str) == str(row["JetLearn Deal Source"]))
        if "Country" in row.index and not pd.isna(row.get("Country", np.nan)):
            m &= (paid["Country"].astype(str) == str(row["Country"]))
        if "Student/Academic Counsellor" in row.index and not pd.isna(row.get("Student/Academic Counsellor", np.nan)):
            m &= (paid["Student/Academic Counsellor"].astype(str) == str(row["Student/Academic Counsellor"]))
        return m

    # Prepare table for Today/Tomorrow
    vt = v_this.copy()
    vt["Today"] = 0.0
    vt["Tomorrow"] = 0.0

    for i, row in vt.iterrows():
        m = mask_for_row(row)
        dom_profile = day_of_month_profile(paid, cm, m, prior_strength=daily_prior_strength)
        p_today = float(dom_profile.reindex(range(1, month_days(cm) + 1)).get(dom_today, 0.0))
        p_tom = float(dom_profile.reindex(range(1, month_days(cm) + 1)).get(dom_tom, 0.0))
        vt.at[i, "Today"] = row["This Month"] * p_today
        vt.at[i, "Tomorrow"] = row["This Month"] * p_tom

    # Show KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>Today ({pd.Timestamp(today_dt):%d %b})</div>
          <div class='value'>{int(round(vt['Today'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>Tomorrow ({pd.Timestamp(today_dt + timedelta(days=1)):%d %b})</div>
          <div class='value'>{int(round(vt['Tomorrow'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>This Month ({cm:%b %Y})</div>
          <div class='value'>{int(round(v_this['This Month'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class='kpi'>
          <div class='label'>Next Month ({nm:%b %Y})</div>
          <div class='value'>{int(round(v_next['Next Month'].sum())):,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    # ----------------------------
    # Tables & Downloads
    # ----------------------------
    st.markdown("#### Month-level Forecast (cohort-based)")
    st.dataframe(merged_months.sort_values("This Month", ascending=False), use_container_width=True)
    st.download_button("Download — Month Forecast CSV",
                       merged_months.to_csv(index=False).encode("utf-8"),
                       file_name="cohort_month_forecast.csv", mime="text/csv")

    st.markdown("#### Today & Tomorrow (by group)")
    st.dataframe(vt.sort_values("Today", ascending=False), use_container_width=True)
    st.download_button("Download — Today/Tomorrow CSV",
                       vt.to_csv(index=False).encode("utf-8"),
                       file_name="cohort_today_tomorrow.csv", mime="text/csv")

    # Optional hourly split for Today for the top groups (limit to 10 for clarity)
    st.markdown("#### Optional: Today's hourly breakdown (top 10 groups by Today forecast)")
    top_today = vt.sort_values("Today", ascending=False).head(10)
    if not top_today.empty:
        for _, row in top_today.iterrows():
            m = mask_for_row(row)
            hprof = hour_of_day_profile(paid, m, prior_strength=hourly_prior_strength)
            hour_df = hprof.rename("Prop").reset_index()
            hour_df["Forecast"] = (row["Today"] * hour_df["Prop"]).round(1)
            title_bits = []
            if "JetLearn Deal Source" in row.index and not pd.isna(row["JetLearn Deal Source"]):
                title_bits.append(str(row["JetLearn Deal Source"]))
            if "Country" in row.index and not pd.isna(row.get("Country", np.nan)):
                title_bits.append(str(row["Country"]))
            if "Student/Academic Counsellor" in row.index and not pd.isna(row.get("Student/Academic Counsellor", np.nan)):
                title_bits.append(str(row["Student/Academic Counsellor"]))
            st.caption(" — ".join(title_bits) or "Group")
            ch = alt.Chart(hour_df).mark_bar().encode(
                x=alt.X("hour:O", title=None),
                y=alt.Y("Forecast:Q", title=None),
                tooltip=["hour","Forecast","Prop"]
            ).properties(height=160)
            st.altair_chart(ch, use_container_width=True)

    st.caption("Model: Cohort-based (M0/M1) using EB-smoothed rates by Deal Source × Country × Counsellor; daily/hourly allocation from historical payment patterns of the same month-of-year. Excludes '1.2 Invalid Deal'.")
