"""
app.py
=======
IPL Analytics Platform - Main Streamlit Application.
Fully self-contained, no React, pure Python.

Run:  streamlit run app.py  
"""

import os
import sys
import json
import time
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(229,57,53,0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: 2px;
}
.main-header p {
    color: rgba(255,255,255,0.65);
    font-size: 1rem;
    margin: 0.4rem 0 0;
}
.accent { color: #e53935; }

.metric-card {
    background: linear-gradient(145deg, #1e1e2e, #16213e);
    border: 1px solid rgba(229,57,53,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform .2s, border-color .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(229,57,53,0.6);
}
.metric-card .metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e53935;
}
.metric-card .metric-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.pred-box {
    background: linear-gradient(135deg, #0f3460, #16213e);
    border: 2px solid #e53935;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.pred-winner {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e53935;
}
.pred-conf {
    color: rgba(255,255,255,0.7);
    font-size: 0.95rem;
}
.team-badge {
    display: inline-block;
    background: rgba(229,57,53,0.15);
    border: 1px solid rgba(229,57,53,0.4);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    color: #e88;
    font-weight: 600;
    font-size: 0.9rem;
}
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    border-left: 4px solid #e53935;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}
hr.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 1.5rem 0;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(229,57,53,0.2);
}
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #e53935 !important;
    background: rgba(229,57,53,0.15) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Cached data loaders ───────────────────────────────────

@st.cache_data(ttl=600)
def load_matches():
    p = "data/matches.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_deliveries():
    p = "data/deliveries.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_team_stats():
    p = "data/processed/team_stats.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_batting_stats():
    p = "data/processed/batting_stats.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_bowling_stats():
    p = "data/processed/bowling_stats.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_h2h():
    p = "data/processed/h2h_stats.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_venue_stats():
    p = "data/processed/venue_stats.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_player_impact():
    p = "data/processed/player_impact.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_model_cached():
    from ml.predictor import load_model, model_exists
    if not model_exists():
        return None
    return load_model()

def data_ready():
    return os.path.exists("data/processed/team_stats.csv")

# ── Plotly theme ──────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(13,13,26,0)",
    plot_bgcolor="rgba(13,13,26,0)",
    font=dict(family="Inter", color="#ccc"),
)

def styled_fig(fig, title=""):
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text=title, font=dict(size=16, family="Rajdhani"), x=0.01),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Sidebar Navigation ────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 1.5rem;">
        <div style="font-family:Rajdhani;font-size:1.4rem;font-weight:700;
                    color:#fff;letter-spacing:1px;">
            IPL Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Match Predictor", "Player Analytics",
         "Team Analysis", "Live Match Simulator", "Season Stats"],
        label_visibility="collapsed",
    )

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.1);'>",
        unsafe_allow_html=True,
    )

    if not data_ready():
        st.warning("Run python train.py first to generate analytics.")
    else:
        st.success("Data ready")
        meta_path = "data/models/winner_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            st.markdown(f"""
            <div style="background:rgba(229,57,53,0.1);border:1px solid rgba(229,57,53,0.3);
                        border-radius:8px;padding:0.8rem;font-size:0.82rem;">
                <b style="color:#e88;">Model Stats</b><br>
                Accuracy: <b style="color:#4fc3f7;">{meta.get('accuracy', 0):.1%}</b><br>
                AUC-ROC:  <b style="color:#4fc3f7;">{meta.get('roc_auc', 0):.4f}</b><br>
                F1 Score: <b style="color:#4fc3f7;">{meta.get('f1_score', 0):.4f}</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.1);'>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style="color:rgba(255,255,255,0.35);font-size:0.75rem;text-align:center;">
        IPL Analytics<br>Built with Python + ML
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>IPL <span class="accent">Analytics</span> Platform</h1>
        <p>Data-driven insights across every IPL season</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready():
        st.info("Download the Kaggle dataset and run python train.py to populate the dashboard.")
        st.code("# Setup steps\n1. Download data from Kaggle (see README)\n"
                "2. Place matches.csv + deliveries.csv in data/\n"
                "3. python train.py\n4. streamlit run app.py")
        st.stop()

    matches    = load_matches()
    team_stats = load_team_stats()

    # KPI row
    total_matches = len(matches)
    total_seasons = matches["season"].nunique() if "season" in matches.columns else 0
    total_teams   = len(team_stats)
    valid_matches = (
        matches[matches["winner"].notna()].copy()
        if "winner" in matches.columns else matches.copy()
    )
    toss_wins = (
        (valid_matches["toss_winner"] == valid_matches["winner"]).mean() * 100
        if "toss_winner" in valid_matches.columns else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, total_matches,        "Total Matches"),
        (c2, total_seasons,        "IPL Seasons"),
        (c3, total_teams,          "Active Teams"),
        (c4, f"{toss_wins:.1f}%",  "Toss Win Rate"),
        (c5, valid_matches["player_of_match"].nunique()
             if "player_of_match" in valid_matches.columns else "-",
             "Unique POMs"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Team Strength Index</div>", unsafe_allow_html=True)

    fig = px.bar(
        team_stats.sort_values("team_strength_index"),
        x="team_strength_index", y="team",
        orientation="h",
        color="team_strength_index",
        color_continuous_scale=["#1a1a2e", "#e53935"],
        labels={"team_strength_index": "Strength Index", "team": ""},
    )
    fig = styled_fig(fig, "Team Strength Index (All-time)")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>Wins by Season</div>", unsafe_allow_html=True)
        if "season" in matches.columns and "winner" in matches.columns:
            season_wins = (
                valid_matches.groupby(["season", "winner"]).size()
                .reset_index(name="wins")
            )
            top_teams   = team_stats.head(6)["team"].tolist()
            season_wins = season_wins[season_wins["winner"].isin(top_teams)]
            fig2 = px.line(
                season_wins, x="season", y="wins", color="winner",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig2 = styled_fig(fig2, "Season-wise Wins (Top Teams)")
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>Win Method Distribution</div>",
                    unsafe_allow_html=True)
        bat_wins, chase_wins = 0, 0
        # Support both schema variants
        if "win_by_runs" in valid_matches.columns and "win_by_wickets" in valid_matches.columns:
            bat_wins   = int((valid_matches["win_by_runs"] > 0).sum())
            chase_wins = int((valid_matches["win_by_wickets"] > 0).sum())
        elif "result" in valid_matches.columns:
            bat_wins   = int((valid_matches["result"] == "runs").sum())
            chase_wins = int((valid_matches["result"] == "wickets").sum())

        if bat_wins + chase_wins > 0:
            fig3 = go.Figure(go.Pie(
                labels=["Bat First (Won by Runs)", "Chase (Won by Wickets)"],
                values=[bat_wins, chase_wins],
                hole=0.5,
                marker_colors=["#e53935", "#1565c0"],
            ))
            fig3 = styled_fig(fig3, "How Matches Are Won")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Team Performance Table</div>",
                unsafe_allow_html=True)
    display_cols   = ["team", "total_matches", "total_wins", "win_rate",
                      "recent_form", "team_strength_index"]
    available_cols = [c for c in display_cols if c in team_stats.columns]
    df_display     = team_stats[available_cols].copy()
    df_display.columns = [c.replace("_", " ").title() for c in available_cols]
    st.dataframe(df_display, use_container_width=True, height=420)


# ─────────────────────────────────────────────────────────
#  PAGE 2 — MATCH PREDICTOR
# ─────────────────────────────────────────────────────────
elif page == "Match Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>Match <span class="accent">Predictor</span></h1>
        <p>Ensemble model: XGBoost + LightGBM + RandomForest stacking</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready():
        st.warning("Run python train.py first.")
        st.stop()

    team_stats  = load_team_stats()
    h2h_stats   = load_h2h()
    venue_stats = load_venue_stats()
    model       = load_model_cached()

    teams  = sorted(team_stats["team"].tolist()) if not team_stats.empty else []
    venues = sorted(venue_stats["venue"].tolist()) if not venue_stats.empty else []

    if not teams:
        st.error("No team data. Run python train.py first.")
        st.stop()

    st.markdown("<div class='section-header'>Configure Match</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0)
    with col2:
        other_teams = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", other_teams, index=min(1, len(other_teams) - 1))

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    with col4:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    with col5:
        venue = st.selectbox("Venue", venues if venues else ["Unknown"])
    with col6:
        season = st.number_input("Season", min_value=2008, max_value=2030, value=2024)

    if st.button("Predict Winner", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not found. Run: python train.py")
        else:
            with st.spinner("Running prediction..."):
                from ml.predictor import predict_match as pm
                result = pm(
                    model=model,
                    team1=team1, team2=team2,
                    toss_winner=toss_winner, toss_decision=toss_decision,
                    venue=str(venue), season=int(season),
                    team_stats=team_stats, h2h_stats=h2h_stats,
                    venue_stats=venue_stats,
                )
                time.sleep(0.3)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            r_col1, r_col2, r_col3 = st.columns([1, 1.5, 1])

            t1_pct = result["team1_win_probability"] * 100
            t2_pct = result["team2_win_probability"] * 100

            with r_col1:
                is_w1  = result["predicted_winner"] == team1
                border = "2px solid #e53935" if is_w1 else "1px solid rgba(255,255,255,0.1)"
                tag    = ("<div style='margin-top:0.5rem;color:#e53935;font-weight:600;'>"
                          "PREDICTED WINNER</div>") if is_w1 else ""
                st.markdown(f"""
                <div style="text-align:center;padding:1.5rem;
                            background:rgba(229,57,53,0.05);
                            border:{border};border-radius:12px;">
                    <div style="font-family:Rajdhani;font-size:1.3rem;
                                font-weight:700;color:#fff;">{team1}</div>
                    <div style="font-size:2.5rem;font-weight:700;
                                color:{'#e53935' if is_w1 else '#aaa'};">
                        {t1_pct:.1f}%
                    </div>
                    <div style="color:rgba(255,255,255,0.5);font-size:0.85rem;">
                        Win Probability
                    </div>
                    {tag}
                </div>
                """, unsafe_allow_html=True)

            with r_col2:
                st.markdown(f"""
                <div class="pred-box">
                    <div style="color:rgba(255,255,255,0.5);font-size:0.85rem;
                                text-transform:uppercase;letter-spacing:1px;">
                        Prediction
                    </div>
                    <div class="pred-winner" style="margin:0.8rem 0;">
                        {result['predicted_winner']}
                    </div>
                    <div class="pred-conf">
                        Confidence:
                        <b style="color:#4fc3f7;">{result['confidence']*100:.1f}%</b>
                    </div>
                    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);
                               margin:1rem 0;">
                    <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);">
                        Toss: <b style="color:#fff;">{toss_winner}</b>
                        chose to <b style="color:#fff;">{toss_decision}</b><br>
                        Venue: <b style="color:#fff;">{venue}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r_col3:
                is_w2   = result["predicted_winner"] == team2
                border2 = "2px solid #e53935" if is_w2 else "1px solid rgba(255,255,255,0.1)"
                tag2    = ("<div style='margin-top:0.5rem;color:#e53935;font-weight:600;'>"
                           "PREDICTED WINNER</div>") if is_w2 else ""
                st.markdown(f"""
                <div style="text-align:center;padding:1.5rem;
                            background:rgba(229,57,53,0.05);
                            border:{border2};border-radius:12px;">
                    <div style="font-family:Rajdhani;font-size:1.3rem;
                                font-weight:700;color:#fff;">{team2}</div>
                    <div style="font-size:2.5rem;font-weight:700;
                                color:{'#e53935' if is_w2 else '#aaa'};">
                        {t2_pct:.1f}%
                    </div>
                    <div style="color:rgba(255,255,255,0.5);font-size:0.85rem;">
                        Win Probability
                    </div>
                    {tag2}
                </div>
                """, unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=[t1_pct, t2_pct],
                y=[team1, team2],
                orientation="h",
                marker_color=["#e53935" if result["predicted_winner"] == team1 else "#1565c0",
                              "#e53935" if result["predicted_winner"] == team2 else "#1565c0"],
                text=[f"{t1_pct:.1f}%", f"{t2_pct:.1f}%"],
                textposition="outside",
            ))
            fig = styled_fig(fig, "Win Probability Comparison")
            fig.update_layout(xaxis_range=[0, 100], height=200)
            st.plotly_chart(fig, use_container_width=True)

            # H2H record
            h2h     = load_h2h()
            h2h_row = h2h[
                ((h2h["team1"] == team1) & (h2h["team2"] == team2)) |
                ((h2h["team1"] == team2) & (h2h["team2"] == team1))
            ]
            if not h2h_row.empty:
                row     = h2h_row.iloc[0]
                t1w     = int(row["team1_wins"]) if row["team1"] == team1 else int(row["team2_wins"])
                t2w     = int(row["team2_wins"]) if row["team1"] == team1 else int(row["team1_wins"])
                total_m = int(row["total_meetings"])
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);
                            border:1px solid rgba(255,255,255,0.1);
                            border-radius:10px;padding:1rem;
                            text-align:center;margin-top:1rem;">
                    <b>Head-to-Head Record</b>&nbsp;&nbsp;
                    <span class="team-badge">{team1}</span>
                    <b style="color:#e53935;margin:0 0.8rem;">{t1w} - {t2w}</b>
                    <span class="team-badge">{team2}</span>
                    <span style="color:rgba(255,255,255,0.4);font-size:0.85rem;
                                 margin-left:1rem;">
                        ({total_m} meetings)
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Key factors
            ts = team_stats.set_index("team")

            def tstat(t, c):
                try:
                    return float(ts.loc[t, c])
                except Exception:
                    return 0.0

            factors = {
                "Win Rate":         (tstat(team1, "win_rate") * 100,
                                     tstat(team2, "win_rate") * 100),
                "Recent Form (10)": (tstat(team1, "recent_form") * 100,
                                     tstat(team2, "recent_form") * 100),
                "Toss Win Rate":    (tstat(team1, "toss_win_rate") * 100,
                                     tstat(team2, "toss_win_rate") * 100),
                "Strength Index":   (tstat(team1, "team_strength_index"),
                                     tstat(team2, "team_strength_index")),
            }

            st.markdown("<div class='section-header'>Key Contributing Factors</div>",
                        unsafe_allow_html=True)
            cats    = list(factors.keys())
            t1_vals = [v[0] for v in factors.values()]
            t2_vals = [v[1] for v in factors.values()]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name=team1, x=cats, y=t1_vals, marker_color="#e53935"))
            fig2.add_trace(go.Bar(name=team2, x=cats, y=t2_vals, marker_color="#1565c0"))
            fig2 = styled_fig(fig2, "Factor Comparison")
            fig2.update_layout(barmode="group", height=320)
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  PAGE 3 — PLAYER ANALYTICS
# ─────────────────────────────────────────────────────────
elif page == "Player Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>Player <span class="accent">Analytics</span></h1>
        <p>Batting · Bowling · Player Impact Score</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready():
        st.warning("Run python train.py first.")
        st.stop()

    batting = load_batting_stats()
    bowling = load_bowling_stats()
    impact  = load_player_impact()

    tab1, tab2, tab3 = st.tabs(["Batting", "Bowling", "Impact Score"])

    # ── Batting ───────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_innings = st.slider("Min Innings", 1, 50, 10)
        with col2:
            top_n = st.slider("Show Top N Players", 5, 50, 20)
        with col3:
            metric = st.selectbox("Rank By",
                                  ["total_runs", "batting_avg", "strike_rate", "batting_impact"])

        df_bat = (
            batting[batting["innings"] >= min_innings].nlargest(top_n, metric)
            if "innings" in batting.columns else batting.head(top_n)
        )
        player_col = (
            "batter"  if "batter"  in df_bat.columns else
            "batsman" if "batsman" in df_bat.columns else df_bat.columns[0]
        )

        fig = px.bar(
            df_bat.sort_values(metric), x=metric, y=player_col,
            orientation="h", color=metric,
            color_continuous_scale=["#1a1a2e", "#e53935"],
        )
        fig = styled_fig(fig, f"Top {top_n} Batsmen by {metric.replace('_', ' ').title()}")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Batting Average vs Strike Rate</div>",
                    unsafe_allow_html=True)
        fig2 = px.scatter(
            df_bat, x="batting_avg", y="strike_rate",
            size="total_runs", color="batting_impact",
            hover_name=player_col, text=player_col,
            color_continuous_scale="RdYlGn",
        )
        fig2 = styled_fig(fig2, "Batting Average vs Strike Rate (bubble = total runs)")
        fig2.update_traces(textposition="top center", textfont_size=9)
        st.plotly_chart(fig2, use_container_width=True)

        disp = [player_col, "total_runs", "innings", "batting_avg",
                "strike_rate", "fours", "sixes", "batting_impact"]
        disp = [c for c in disp if c in df_bat.columns]
        st.dataframe(df_bat[disp].reset_index(drop=True), use_container_width=True, height=350)

    # ── Bowling ───────────────────────────────────────────
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_wkts = st.slider("Min Wickets", 1, 50, 5)
        with col2:
            top_bowl = st.slider("Show Top N Bowlers", 5, 50, 20)
        with col3:
            bowl_metric = st.selectbox(
                "Rank By", ["wickets", "economy_rate", "bowling_impact", "bowling_avg"]
            )

        df_bowl = (
            bowling[bowling["wickets"] >= min_wkts].nlargest(top_bowl, bowl_metric)
            if "wickets" in bowling.columns else bowling.head(top_bowl)
        )
        if bowl_metric in ["economy_rate", "bowling_avg"]:
            df_bowl = df_bowl.nsmallest(top_bowl, bowl_metric)

        fig3 = px.bar(
            df_bowl.sort_values(bowl_metric,
                                ascending=(bowl_metric in ["economy_rate", "bowling_avg"])),
            x=bowl_metric, y="bowler",
            orientation="h", color=bowl_metric,
            color_continuous_scale=(
                ["#e53935", "#1a1a2e"] if bowl_metric != "economy_rate"
                else ["#1a1a2e", "#e53935"]
            ),
        )
        fig3 = styled_fig(fig3, f"Top Bowlers by {bowl_metric.replace('_', ' ').title()}")
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        disp2 = ["bowler", "wickets", "matches", "economy_rate",
                 "bowling_avg", "bowling_sr", "dot_ball_pct", "bowling_impact"]
        disp2 = [c for c in disp2 if c in df_bowl.columns]
        st.dataframe(df_bowl[disp2].reset_index(drop=True), use_container_width=True, height=350)

    # ── Impact Score ──────────────────────────────────────
    with tab3:
        if impact.empty:
            st.info("Player impact data not available.")
        else:
            top_impact = impact.head(30)
            fig4 = px.bar(
                top_impact.sort_values("player_impact_score"),
                x="player_impact_score", y="player",
                color="role", orientation="h",
                color_discrete_map={
                    "Batsman": "#e53935", "Bowler": "#1565c0",
                    "All-Rounder": "#f57c00", "Unknown": "#555",
                },
            )
            fig4 = styled_fig(fig4, "Player Impact Score - Top 30")
            st.plotly_chart(fig4, use_container_width=True)

            role_dist = impact["role"].value_counts().reset_index()
            role_dist.columns = ["role", "count"]
            fig5 = go.Figure(go.Pie(
                labels=role_dist["role"], values=role_dist["count"],
                hole=0.55,
                marker_colors=["#e53935", "#1565c0", "#f57c00", "#666"],
            ))
            fig5 = styled_fig(fig5, "Player Role Distribution")
            st.plotly_chart(fig5, use_container_width=True)

            disp3 = ["player", "player_impact_score", "role",
                     "batting_impact", "bowling_impact", "total_runs", "wickets"]
            disp3 = [c for c in disp3 if c in top_impact.columns]
            st.dataframe(top_impact[disp3].reset_index(drop=True),
                         use_container_width=True, height=400)


# ─────────────────────────────────────────────────────────
#  PAGE 4 — TEAM ANALYSIS
# ─────────────────────────────────────────────────────────
elif page == "Team Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>Team <span class="accent">Analysis</span></h1>
        <p>Strength index · Head-to-head · Venue performance</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready():
        st.warning("Run python train.py first.")
        st.stop()

    team_stats = load_team_stats()
    h2h        = load_h2h()
    venue      = load_venue_stats()
    matches    = load_matches()

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Head-to-Head", "Venues", "Toss Impact"])

    with tab1:
        st.markdown("<div class='section-header'>Team Strength Radar</div>",
                    unsafe_allow_html=True)
        top6       = team_stats.head(6)
        categories = ["win_rate", "recent_form", "toss_win_rate"]
        cats_clean = ["Win Rate", "Recent Form", "Toss Win Rate"]
        cats_close = cats_clean + [cats_clean[0]]

        fig    = go.Figure()
        colors = ["#e53935", "#1565c0", "#f57c00", "#2e7d32", "#6a1b9a", "#00695c"]
        for i, (_, row) in enumerate(top6.iterrows()):
            vals       = [float(row.get(c, 0)) for c in categories]
            vals_close = vals + [vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals_close, theta=cats_close, fill="toself",
                name=str(row["team"]),
                line_color=colors[i % len(colors)],
            ))
        fig = styled_fig(fig, "Team Performance Radar (Top 6)")
        fig.update_layout(polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="rgba(0,0,0,0.2)",
        ), height=480)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            team_stats, x="win_rate", y="recent_form",
            size="total_matches", color="team_strength_index",
            hover_name="team", text="team",
            color_continuous_scale="RdYlGn",
        )
        fig2 = styled_fig(fig2, "Win Rate vs Recent Form")
        fig2.update_traces(textposition="top center", textfont_size=9)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-header'>Head-to-Head</div>", unsafe_allow_html=True)
        teams_list = sorted(team_stats["team"].tolist())
        col1, col2 = st.columns(2)
        with col1:
            t1 = st.selectbox("Team A", teams_list, index=0, key="h2h_t1")
        with col2:
            t2_opts = [t for t in teams_list if t != t1]
            t2 = st.selectbox("Team B", t2_opts, index=0, key="h2h_t2")

        row = h2h[
            ((h2h["team1"] == t1) & (h2h["team2"] == t2)) |
            ((h2h["team1"] == t2) & (h2h["team2"] == t1))
        ]
        if not row.empty:
            r    = row.iloc[0]
            t1w  = int(r["team1_wins"]) if r["team1"] == t1 else int(r["team2_wins"])
            t2w  = int(r["team2_wins"]) if r["team1"] == t1 else int(r["team1_wins"])
            total = int(r["total_meetings"])
            t1_wr = t1w / total if total > 0 else 0
            t2_wr = t2w / total if total > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{t1} Wins",    t1w,   f"{t1_wr:.1%} win rate")
            c2.metric("Total Matches", total)
            c3.metric(f"{t2} Wins",    t2w,   f"{t2_wr:.1%} win rate")

            fig3 = go.Figure(go.Pie(
                labels=[f"{t1} Wins", f"{t2} Wins"],
                values=[t1w, t2w], hole=0.6,
                marker_colors=["#e53935", "#1565c0"],
            ))
            fig3.update_layout(**PLOTLY_DARK, height=300)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No head-to-head record found for this pair.")

    with tab3:
        top_venues     = venue.head(15)
        chase_wins_col = top_venues["total_matches"] - top_venues["bat_first_wins"]
        fig4           = go.Figure()
        fig4.add_trace(go.Bar(
            name="Bat First Wins", x=top_venues["venue"],
            y=top_venues["bat_first_wins"], marker_color="#e53935",
        ))
        fig4.add_trace(go.Bar(
            name="Chase Wins", x=top_venues["venue"],
            y=chase_wins_col, marker_color="#1565c0",
        ))
        fig4 = styled_fig(fig4, "Bat-first vs Chase Wins by Venue")
        fig4.update_layout(barmode="stack", xaxis_tickangle=-35, height=450)
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(
            venue[["venue", "total_matches", "bat_first_wins", "bat_first_win_rate"]].head(20),
            use_container_width=True,
        )

    with tab4:
        if not matches.empty and "toss_winner" in matches.columns:
            valid = matches[matches["winner"].notna()].copy()
            valid["toss_win"] = valid["toss_winner"] == valid["winner"]
            season_toss = valid.groupby("season")["toss_win"].mean().reset_index()
            season_toss.columns = ["season", "toss_win_pct"]
            season_toss["toss_win_pct"] *= 100

            fig5 = px.area(
                season_toss, x="season", y="toss_win_pct",
                color_discrete_sequence=["#e53935"],
            )
            fig5 = styled_fig(fig5, "Toss Win to Match Win Rate by Season (%)")
            fig5.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                           annotation_text="50% baseline")
            st.plotly_chart(fig5, use_container_width=True)

            bat_wins_toss   = (valid[valid["toss_decision"] == "bat"]["toss_win"]).mean() * 100
            field_wins_toss = (valid[valid["toss_decision"] == "field"]["toss_win"]).mean() * 100

            c1, c2 = st.columns(2)
            c1.metric("Bat-first after winning toss win %", f"{bat_wins_toss:.1f}%")
            c2.metric("Field after winning toss win %",     f"{field_wins_toss:.1f}%")


# ─────────────────────────────────────────────────────────
#  PAGE 5 — LIVE MATCH SIMULATOR
# ─────────────────────────────────────────────────────────
elif page == "Live Match Simulator":
    st.markdown("""
    <div class="main-header">
        <h1>Live Match <span class="accent">Simulator</span></h1>
        <p>Real-time win probability as the match progresses ball by ball</p>
    </div>
    """, unsafe_allow_html=True)

    from ml.features import live_win_probability

    team_stats_sim = load_team_stats()
    teams_sim = sorted(team_stats_sim["team"].tolist()) if not team_stats_sim.empty else [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
        "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
        "Punjab Kings", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants",
    ]

    col1, col2 = st.columns(2)
    with col1:
        mi_idx       = teams_sim.index("Mumbai Indians") if "Mumbai Indians" in teams_sim else 0
        batting_team = st.selectbox("Batting Team", options=teams_sim, index=mi_idx)
    with col2:
        inning = st.radio("Innings", [1, 2], horizontal=True)
        if inning == 2:
            target = st.number_input("Target (runs needed)", min_value=50, max_value=300, value=170)
        else:
            target = None

    bowling_options = [t for t in teams_sim if t != batting_team]
    csk_idx         = (bowling_options.index("Chennai Super Kings")
                       if "Chennai Super Kings" in bowling_options else 0)
    bowling_team    = st.selectbox("Bowling Team", options=bowling_options, index=csk_idx)

    st.markdown("<div class='section-header'>Current Match State</div>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        runs = st.number_input("Runs Scored", min_value=0, max_value=350, value=80)
    with col4:
        wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=3)
    with col5:
        overs = st.slider("Overs Completed", 0.0, 20.0, 10.0, step=0.1)

    balls_bowled = int(overs * 6)
    prob         = live_win_probability(
        runs_scored=runs,
        wickets_fallen=wickets,
        balls_bowled=balls_bowled,
        target=target,
        inning=inning,
    )

    bat_prob  = prob["batting_team"]
    bowl_prob = prob["bowling_team"]

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);
                border-radius:16px;padding:2rem;margin:1rem 0;text-align:center;">
        <div style="font-family:Rajdhani;font-size:1.2rem;color:rgba(255,255,255,0.6);
                    text-transform:uppercase;letter-spacing:2px;">Win Probability</div>
        <div style="display:flex;justify-content:space-between;
                    align-items:center;margin:1.5rem 0;">
            <div style="text-align:left;">
                <div style="font-family:Rajdhani;font-size:1.1rem;
                            color:rgba(255,255,255,0.7);">{batting_team}</div>
                <div style="font-family:Rajdhani;font-size:3rem;
                            font-weight:700;color:#e53935;">{bat_prob*100:.1f}%</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.8rem;">BATTING</div>
            </div>
            <div style="font-family:Rajdhani;font-size:2rem;font-weight:700;
                        color:rgba(255,255,255,0.3);">VS</div>
            <div style="text-align:right;">
                <div style="font-family:Rajdhani;font-size:1.1rem;
                            color:rgba(255,255,255,0.7);">{bowling_team}</div>
                <div style="font-family:Rajdhani;font-size:3rem;
                            font-weight:700;color:#1565c0;">{bowl_prob*100:.1f}%</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.8rem;">BOWLING</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=[bat_prob * 100, bowl_prob * 100],
        y=[batting_team, bowling_team],
        orientation="h",
        marker_color=["#e53935", "#1565c0"],
        text=[f"{bat_prob*100:.1f}%", f"{bowl_prob*100:.1f}%"],
        textposition="inside",
    ))
    fig = styled_fig(fig, "Win Probability Meter")
    fig.update_layout(height=200, xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    crr       = (runs / (balls_bowled / 6)) if balls_bowled > 0 else 0
    balls_rem = 120 - balls_bowled

    st.markdown("<div class='section-header'>Match Statistics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Runs",    runs)
    c2.metric("Wickets", f"{wickets}/10")
    c3.metric("Overs",   f"{overs:.1f}/20")
    c4.metric("CRR",     f"{crr:.2f}")
    if inning == 2 and target:
        runs_needed = target - runs
        rrr = (runs_needed / (balls_rem / 6)) if balls_rem > 0 else 999
        c5.metric("RRR", f"{rrr:.2f}",
                  f"Need {runs_needed} in {balls_rem//6}.{balls_rem%6} ov")
    else:
        projected = runs + crr * (balls_rem / 6) if crr > 0 else runs
        c5.metric("Projected", f"{int(projected)}")

    st.markdown("<div class='section-header'>Projected Win Probability Curve</div>",
                unsafe_allow_html=True)
    over_points = []
    for ov in range(1, 21):
        b = ov * 6
        r = int(runs * (b / balls_bowled)) if balls_bowled > 0 else int(crr * ov)
        w = min(int(wickets * (b / balls_bowled)) if balls_bowled > 0 else 0, 10)
        p = live_win_probability(r, w, b, target, inning)
        over_points.append({"over": ov, "bat_win_prob": p["batting_team"] * 100})

    sim_df = pd.DataFrame(over_points)
    fig2   = px.area(sim_df, x="over", y="bat_win_prob",
                     color_discrete_sequence=["#e53935"])
    fig2   = styled_fig(fig2, f"Win Probability Projection - {batting_team}")
    fig2.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                   annotation_text="50% (Even)")
    fig2.add_vline(x=float(overs), line_dash="dash", line_color="#4fc3f7",
                   annotation_text="Current")
    fig2.update_yaxes(range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  PAGE 6 — SEASON STATS
# ─────────────────────────────────────────────────────────
elif page == "Season Stats":
    st.markdown("""
    <div class="main-header">
        <h1>Season <span class="accent">Statistics</span></h1>
        <p>All-time IPL records · Orange Cap · Purple Cap · Champions</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready():
        st.warning("Run python train.py first.")
        st.stop()

    matches    = load_matches()
    deliveries = load_deliveries()

    if matches.empty:
        st.error("No match data found.")
        st.stop()

    from ml.analytics import (
        season_summary, toss_impact_analysis,
        orange_cap_tracker, purple_cap_tracker, phase_analysis,
    )

    tab1, tab2, tab3 = st.tabs(["Season History", "Cap Trackers", "Phase Analysis"])

    with tab1:
        try:
            ss = season_summary(matches)
        except Exception as e:
            st.error(f"Could not build season summary: {e}")
            ss = pd.DataFrame()

        if not ss.empty:
            fig = px.bar(
                ss, x="season", y="total_matches",
                color="bat_win_pct",
                color_continuous_scale="RdYlGn",
                text="champion",
            )
            fig = styled_fig(fig, "Matches per Season (coloured by bat-first win %)")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(name="Bat First Win %", x=ss["season"],
                                  y=ss["bat_win_pct"], marker_color="#e53935"))
            fig2.add_trace(go.Scatter(name="Toss Impact %", x=ss["season"],
                                      y=ss["toss_win_match_pct"],
                                      mode="lines+markers", line_color="#4fc3f7"),
                           secondary_y=True)
            fig2.update_layout(**PLOTLY_DARK,
                               title_text="Bat-first Win % vs Toss Impact %",
                               height=380)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(ss, use_container_width=True)

    with tab2:
        if not deliveries.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='section-header'>Orange Cap (Most Runs)</div>",
                            unsafe_allow_html=True)
                try:
                    oc = orange_cap_tracker(deliveries, matches)
                except Exception as e:
                    st.error(f"Orange Cap error: {e}")
                    oc = pd.DataFrame()
                if not oc.empty:
                    fig3 = px.bar(oc, x="season", y="runs", text="orange_cap_holder",
                                  color="runs",
                                  color_continuous_scale=["#ff6d00", "#fff"])
                    fig3 = styled_fig(fig3, "Orange Cap Winners by Season")
                    fig3.update_coloraxes(showscale=False)
                    st.plotly_chart(fig3, use_container_width=True)
                    st.dataframe(oc, use_container_width=True)

            with col2:
                st.markdown("<div class='section-header'>Purple Cap (Most Wickets)</div>",
                            unsafe_allow_html=True)
                try:
                    pc = purple_cap_tracker(deliveries, matches)
                except Exception as e:
                    st.error(f"Purple Cap error: {e}")
                    pc = pd.DataFrame()
                if not pc.empty:
                    fig4 = px.bar(pc, x="season", y="wickets", text="purple_cap_holder",
                                  color="wickets",
                                  color_continuous_scale=["#7b1fa2", "#fff"])
                    fig4 = styled_fig(fig4, "Purple Cap Winners by Season")
                    fig4.update_coloraxes(showscale=False)
                    st.plotly_chart(fig4, use_container_width=True)
                    st.dataframe(pc, use_container_width=True)
        else:
            st.info("Deliveries data not available.")

    with tab3:
        if not deliveries.empty:
            try:
                phases = phase_analysis(deliveries)
            except Exception as e:
                st.error(f"Phase analysis error: {e}")
                phases = pd.DataFrame()

            if not phases.empty:
                fig5 = px.bar(
                    phases, x="phase", y="run_rate",
                    color="phase",
                    color_discrete_sequence=["#1565c0", "#f57c00", "#e53935"],
                    text="run_rate",
                )
                fig5 = styled_fig(fig5, "Average Run Rate by Match Phase")
                st.plotly_chart(fig5, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    fig6 = px.bar(phases, x="phase", y="wickets", color="phase",
                                  color_discrete_sequence=["#1565c0", "#f57c00", "#e53935"])
                    fig6 = styled_fig(fig6, "Wickets Fallen by Phase")
                    st.plotly_chart(fig6, use_container_width=True)
                with c2:
                    fig7 = px.bar(phases, x="phase", y="boundaries", color="phase",
                                  color_discrete_sequence=["#1565c0", "#f57c00", "#e53935"])
                    fig7 = styled_fig(fig7, "Boundaries Hit by Phase")
                    st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("Deliveries data not available for phase analysis.")