# 🏏 IPL Analytics Platform

> AI-Powered IPL Match Prediction & Analytics Platform  
> Built with **Python · Streamlit · FastAPI · SQLite · XGBoost · LightGBM**

### 🚀 Live Demo
**[https://iplanalytics-tizbfs5v2tjgu8wr4vkye7.streamlit.app/](https://iplanalytics-tizbfs5v2tjgu8wr4vkye7.streamlit.app/)**

---

##  Kaggle Dataset — Download Instructions

This project uses the **IPL Complete Dataset** from Kaggle.

### Step 1 — Get the Dataset

**Option A (Recommended — Direct Download):**

1. Go to: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020  
2. Click **Download** (top-right button)  
3. Unzip the downloaded file  
4. Copy **`matches.csv`** and **`deliveries.csv`** into the **`data/`** folder

**Option B — Using Kaggle CLI:**
```bash
pip install kaggle
kaggle datasets download -d patrickb1912/ipl-complete-dataset-20082020
unzip ipl-complete-dataset-20082020.zip -d data/
```

**Option C — Alternative Dataset (if above unavailable):**
- https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set
- https://www.kaggle.com/datasets/manasgarg/ipl

### Step 2 — What You Need

After downloading, your `data/` folder should look like:
```
data/
├── matches.csv       ← Match-level data (2008–2022)
└── deliveries.csv    ← Ball-by-ball data
```

The **matches.csv** must have these columns:
```
id, season, city, date, team1, team2, toss_winner, toss_decision,
result, dl_applied, winner, win_by_runs, win_by_wickets,
player_of_match, venue, umpire1, umpire2
```

The **deliveries.csv** must have these columns:
```
match_id, inning, batting_team, bowling_team, over, ball,
batsman, non_striker, bowler, is_super_over, wide_runs, bye_runs,
legbye_runs, noball_runs, penalty_runs, batsman_runs, extra_runs,
total_runs, player_dismissed, dismissal_kind, fielder
```

---

##  Setup & Run (Local)

### Prerequisites
- Python 3.9 or higher
- pip

### Step-by-step Setup

```bash
# 1. Clone or download this project
git clone https://github.com/yourusername/ipl-predictor
cd ipl-predictor

# 2. Create virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Copy environment variables
cp .env.example .env

# 5. Place your Kaggle CSVs (from above) into the data/ folder
# data/matches.csv
# data/deliveries.csv

# 6. Run the training pipeline (creates model + database)
python train.py

# 7. Launch the Streamlit dashboard
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

##  Run the REST API (Optional)

```bash
# Start FastAPI backend (in a separate terminal)
uvicorn api:app --reload --port 8000

# API Docs: http://localhost:8000/docs
# Interactive: http://localhost:8000/redoc
```

### Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Health check |
| POST   | `/api/v1/predict/match` | Predict match winner |
| POST   | `/api/v1/predict/live-win-probability` | Live win probability |
| GET    | `/api/v1/teams` | All team stats |
| GET    | `/api/v1/teams/h2h/{team1}/{team2}` | Head-to-head |
| GET    | `/api/v1/players/top-batsmen` | Top batsmen |
| GET    | `/api/v1/players/top-bowlers` | Top bowlers |
| GET    | `/api/v1/players/impact-scores` | Player Impact Scores |
| GET    | `/api/v1/model/info` | Model evaluation metrics |

---

##  Docker Setup

```bash
# Build and run everything with Docker Compose
docker-compose up --build

# Frontend: http://localhost:8501
# API:      http://localhost:8000
```

---

## ☁️ Deploy Online (Free — Streamlit Cloud)

1. Push code to GitHub (without the `data/` folder in .gitignore)
2. Go to https://share.streamlit.io
3. Click **New app** → Connect your repo
4. Set **Main file path**: `app.py`
5. In **Advanced settings**, add secrets:
   ```
   DATABASE_PATH = "./data/ipl_analytics.db"
   ```
6. Upload your trained data files via the app or commit `data/processed/` to git

---

##  Deploy on Render (Free Tier)

1. Push to GitHub
2. Go to https://render.com → **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt && python train.py`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Environment**: Python 3.11
5. Add environment variable: `DATABASE_PATH = /opt/render/project/src/data/ipl_analytics.db`

---

##  Project Structure

```
ipl-predictor/
│
├── app.py                    ← Streamlit frontend (all pages)
├── api.py                    ← FastAPI REST backend
├── train.py                  ← Training pipeline (run this first!)
├── database.py               ← SQLite database models
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Docker container
├── docker-compose.yml        ← Docker Compose (frontend + API)
├── Procfile                  ← Render/Heroku deployment
├── runtime.txt               ← Python version for cloud
│
├── ml/
│   ├── __init__.py
│   ├── features.py           ← Feature engineering pipeline
│   ├── predictor.py          ← XGBoost + LightGBM + RF ensemble
│   └── analytics.py         ← Advanced analytics functions
│
├── data/                     ← (Create this folder)
│   ├── matches.csv           ← Download from Kaggle ← PUT HERE
│   ├── deliveries.csv        ← Download from Kaggle ← PUT HERE
│   ├── processed/            ← Auto-created by train.py
│   └── models/               ← Auto-created by train.py
│
└── .streamlit/
    └── config.toml           ← Dark theme config
```

---

##  ML Model Details

### Architecture: Stacking Ensemble
- **Level 0 (Base models)**: XGBoost + LightGBM + Random Forest
- **Level 1 (Meta-learner)**: Logistic Regression
- **Cross-validation**: Stratified 5-fold
- **Target**: Binary — Team 1 wins (1) or Team 2 wins (0)

### Features Used
| Feature | Description |
|---------|-------------|
| `t1_win_rate` | Team 1 all-time win rate |
| `t2_win_rate` | Team 2 all-time win rate |
| `t1_strength` | Team 1 Strength Index (composite) |
| `t2_strength` | Team 2 Strength Index (composite) |
| `t1_recent_form` | Team 1 win rate in last 10 matches |
| `t2_recent_form` | Team 2 win rate in last 10 matches |
| `h2h_win_rate_t1` | Head-to-head win rate (team 1 vs team 2) |
| `toss_winner_is_team1` | Did team 1 win the toss? |
| `toss_decision_bat` | Did they choose to bat? |
| `venue_bat_win_rate` | Venue bat-first win rate |
| `win_rate_diff` | Difference in win rates |
| `strength_diff` | Difference in strength indices |
| `form_diff` | Difference in recent form |

### Custom Metrics
- **Player Impact Score (PIS)**: Batting (55%) + Bowling (45%) composite
- **Team Strength Index**: Win rate + Recent form + Toss advantage + Dominance
- **Win Probability Engine**: Run-rate + wicket-based live calculator

---

##  Dashboard Pages

| Page | Description |
|------|-------------|
|  Dashboard | KPIs, team strength bar chart, season-wise wins |
|  Match Predictor | Input match details → AI predicts winner with probabilities |
|  Player Analytics | Batting, bowling, Player Impact Score |
|  Team Analysis | Radar charts, H2H records, venue stats, toss impact |
|  Live Match Simulator | Real-time win probability calculator over overs |
|  Season Stats | Orange Cap, Purple Cap, phase analysis |

---

##  Run Tests

```bash
python -m pytest tests/ -v
```

---

##  Troubleshooting

**Q: `FileNotFoundError: matches.csv not found`**  
A: Download the Kaggle dataset and place both CSVs in the `data/` folder.

**Q: `Model not found. Run: python train.py`**  
A: Run `python train.py` after placing the CSV files.

**Q: `ModuleNotFoundError: No module named 'xgboost'`**  
A: Run `pip install -r requirements.txt` again.

**Q: Slow loading on first run**  
A: The data is loading and being cached. It will be fast from the second load onwards.

---

##  License
MIT License — Free to use, modify, and deploy.