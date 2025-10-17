# ISTAT Unemployment Nowcasting Lab

A modular, research‑grade Streamlit app for **interactive nowcasting** of Italian unemployment. It supports **daily→monthly** aggregation, **feature engineering**, **walk‑forward backtesting** (baselines, Ridge‑ARX, U‑MIDAS, ARIMAX/SARIMAX), **results analytics** (DM test), **SHAP & events**, **news impact**, a **multi‑provider AI assistant**, and **report export**.

---

## 🔧 Quick Start

```bash
# 1) Create env (Python ≥ 3.10 recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install
pip install -U pip
pip install -r requirements.txt

# 3) (Optional) Setup API keys for page 6 & 8
# Create .streamlit/secrets.toml with your keys (see below)

# 4) Run
streamlit run app.py
```

**Default entry**: the sidebar navigator (from `app.py`) routes to all pages under `/pages` automatically.

---

## 🧭 Project Structure

```
unemployment_lab/
├── app.py                     # Slim shell: theme, navigation, session bootstrap
├── pages/
│   ├── 1_Dashboard.py         # KPIs, recent results, quick insights
│   ├── 2_Data_Aggregation.py  # Upload daily/GT; build monthly/quarterly panels
│   ├── 3_Feature_Engineering.py
│   ├── 4_Backtesting.py       # Baselines, Ridge‑ARX, U‑MIDAS, ARIMAX/SARIMAX
│   ├── 5_Results.py           # Metrics, charts, DM test, exports
│   ├── 6_AI_Assistant.py      # 4‑panel chat: OpenAI/Claude/Gemini/Local
│   ├── 7_SHAP_Events.py       # SHAP global/local + timeline
│   ├── 8_News_Impact.py       # RSS/NewsAPI/GDELT → monthly signals
│   └── 9_Report.py            # Markdown & single‑file HTML report
│
├── utils/
│   ├── state.py               # AppState (central session storage)
│   ├── io_ops.py              # Safe loaders/savers (optional)
│   ├── time_ops.py            # tz‑naive + end‑of‑month helpers
│   ├── feature_ops.py         # diff/pct/log, lags, rolling stats
│   ├── viz.py                 # Plotly helpers
│   ├── models/
│   │   ├── baselines.py       # NAIVE/SNAIVE/MA/ETS
│   │   ├── ridge_arx.py
│   │   ├── ensembles.py
│   │   ├── backtest.py        # walk‑forward engine & metrics
│   │   └── metrics.py         # MAE/RMSE/SMAPE/MASE
│   └── news/
│       ├── sources.py         # RSS/NewsAPI/GDELT fetchers (optional)
│       ├── scoring.py         # lexicon/ML scoring
│       └── aggregation.py     # daily→monthly, smoothing, shifts
│
├── assets/
│   ├── style.css
│   └── logo.svg (optional)
├── configs/
│   ├── labels.json
│   ├── demo_config.json
│   └── app.toml (optional)
├── data/sample/
│   ├── target_istat.csv       # monthly target sample
│   └── daily_sample.csv       # daily features sample
├── .streamlit/
│   └── config.toml            # global theme/layout
├── requirements.txt
└── README.md
```

> **Note**: Many `utils/*` are optional because each page contains safe fallbacks; you can progressively extract helpers into `utils/`.

---

## 📦 Data Contracts (Expected Formats)

### Monthly target (`target_istat.csv`)

* Columns: `date, y` (end‑of‑month or any day per month)
* Example:

```csv
date,y
2016-01-31,11.50
2016-02-29,11.42
...
```

### Daily panel (`daily_sample.csv`)

* Columns: `date, <series_1>, <series_2>, ...` (UTC or local; any tz)
* The app makes everything **tz‑naive** and aligns to **EOM** when aggregating.

```csv
date,google_trend_unemployment,stock_ret
2019-01-02,43,0.004
2019-01-03,45,-0.006
...
```

> For Google Trends, keep original daily format; the app bins/aggregates correctly.

---

## 🔐 Secrets / API Keys (optional)

Create `.streamlit/secrets.toml` if you want to use news/APIs or page‑6 chat:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "..."
GOOGLE_API_KEY = "..."
NEWSAPI_KEY = "..."
# Local LLM endpoint for page 6 (optional)
# OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
```

And a simple theme in `.streamlit/config.toml` (optional):

```toml
[server]
runOnSave = true

[theme]
base = "light"
primaryColor = "#0EA5E9"
backgroundColor = "#ffffff"
textColor = "#111827"
```

---

## ✨ Major Features

* **Daily→Monthly/Quarterly**: robust EOM alignment; safe datetime coercion.
* **Feature Engineering**: transforms (`diff/pct/log`), winsorize, lags/rolling, expanding z‑score.
* **Backtesting**: walk‑forward; NAIVE/SNAIVE/MA/ETS; **Ridge‑ARX**; **U‑MIDAS** (binning last D daily lags); **ARIMAX/SARIMAX**.
* **Results & Stats**: metrics table, **DM test** (HAC), rolling/cumulative errors.
* **Explainability**: **SHAP** (fallback to β·x contributions), local & global.
* **Events**: timeline overlay; pre/post error shifts.
* **News Impact**: RSS/NewsAPI/GDELT; keyword/VADER scoring; monthly signals; cross‑corr, rolling corr, (optional) Granger.
* **AI Assistant**: four parallel chat panels; OpenAI/Claude/Gemini/Local + lightweight RAG.
* **Report**: executive markdown + single‑file HTML with embedded Plotly.

---

## 🧪 Minimal Smoke Test (5 min)

1. **Load sample data** (page 2) → Build monthly panel.
2. **Feature Engineering** (page 3) → add a few lags/rolling stats.
3. **Backtest** (page 4) → pick NAIVE + Ridge‑ARX, `horizon=1`, run.
4. **Results** (page 5) → verify metrics & charts; run a DM test.
5. **News** (page 8) → upload a small CSV, build monthly signals; append to panel.
6. **Report** (page 9) → export `report.html` and open in browser.

> If any step fails, see Troubleshooting.

---

## 🛡️ Troubleshooting (Top 5)

1. **`ValueError` on `merge` / monthly join**
   Root cause: mismatched `datetime64[ns]` vs `object`/timezone.
   **Fix**: always coerce & EOM‑align before merges:

```python
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
df["date_eom"] = (df["date"] + pd.offsets.MonthEnd(0)).dt.normalize()
```

2. **No predictions** in backtest
   Too few training months or NA rows after lagging.
   **Action**: increase `min_train`, reduce lags, enable standardization; check that `panel_monthly` has numeric columns.

3. **U‑MIDAS empty series**
   Your chosen daily column isn’t found in uploaded daily frames, or vintage cutoff removes all rows.
   **Action**: confirm column name & dates; lower cutoff; reduce `D/B`.

4. **ARIMAX/SARIMAX horizon>1**
   Future exogenous variables are unknown.
   **Action**: keep `horizon ≤ 1` or provide your own exog forecasts.

5. **SHAP too slow / not installed**
   **Action**: uncheck SHAP (fallback β·x), or install `shap` with extra wheels; consider limiting features.

---

## ⚙️ Performance Tips

* Prefer **expanding windows** initially; switch to rolling after sanity checks.
* Keep feature set **compact**; remove collinear, quasi‑constant, or duplicate columns.
* Cache heavy steps with `@st.cache_data` (already used where safe).
* For news scoring, start **upload CSV** path before enabling live APIs.

---

## 📜 License & Attribution

* Choose a license (MIT/BSD-3/Apache-2.0). Add `LICENSE` file.
* Data sources (ISTAT, GDELT, NewsAPI, RSS feeds) retain their own terms.

---

## 🤝 Contributing

* Use feature branches; open PRs with focused changes.
* Add unit tests for utilities where feasible (e.g., `time_ops`, `feature_ops`).
* Document new features in this README and the relevant page header.

---

## 🔭 Roadmap (suggested)

* Proper **restricted MIDAS** (Almon/Beta weights) in `utils/models/midas.py`.
* Robust **exog forecaster** for ARIMAX/SARIMAX at `h>1`.
* Optional **SQLite** persistence for runs & configs.
* **Dockerfile** for reproducible deployments.

---

## ✅ Final Checklists

**Before pushing to prod:**

* [ ] `requirements.txt` installed cleanly on fresh env
* [ ] `streamlit run app.py` starts with no exceptions
* [ ] Page 4 creates metrics; Page 5 renders charts; Page 9 exports HTML
* [ ] Optional APIs tested (keys present)

**Sample `.gitignore` additions:**

```
.venv/
__pycache__/
.streamlit/secrets.toml
*.pyc
*.ipynb_checkpoints/
```
