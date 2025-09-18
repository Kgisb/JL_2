# MTD vs Cohort Analyzer (Streamlit)

Interactive app to compare **MTD** (Create-date window; measure-month == create-month) vs **Cohort** (Measure-date window) metrics with global filters and breakdowns.

## Features
- Global filters (Pipeline, JetLearn Deal Source, Country, Counsellor) each with an **All** checkbox
- Auto-detected **Measure date** field (e.g., _Payment Received Date_)
- **MTD** window (Create Date) and **Cohort** window (Measure Date)
- Cohort also shows **Create Count in Cohort window**
- **Split by** Country / JetLearn Deal Source (or both)
- Leaderboards: **Top 5 Countries**, **Top 3 Deal Sources**, and **Country × Deal Source** (top 10 pairs)
- Excludes `1.2 Invalid Deal` globally

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

- Upload your CSV in the sidebar **or** place `master_DB_JL.csv` next to `app.py`.

## Data requirements
Your CSV must include these columns:

- `Pipeline`
- `JetLearn Deal Source`
- `Country`
- `Student/Academic Counsellor`
- `Deal Stage`
- `Create Date`

Other date-like columns (names containing `date`, `time`, `timestamp`) will be auto-parsed and offered as **Measure date** options (e.g., `Payment Received Date`).

## Notes
- The app **excludes** rows where `Deal Stage == "1.2 Invalid Deal"`.
- If your CSV is large on GitHub (>100 MB), use **Git LFS** or keep your data local and upload via the app.
