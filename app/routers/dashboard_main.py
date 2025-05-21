# app/routers/dashboard_main.py
from fastapi import APIRouter, Query
from app.schemas.dashboard_main import DashboardResponse
from app.services import dashboard_main as svc
from app.util.data_loader import load_events, load_matches, load_league_context
from app.util.ml_models import load_xt_model

router = APIRouter(prefix="/api/dashboard", tags=["Player"])

@router.get("/summary", response_model=DashboardResponse)
def get_dashboard_data(teamId: str = Query(..., alias="teamId")):
    events = load_events()
    matches = load_matches()
    xt_model = load_xt_model()
    league_stats = load_league_context()

    # ✅ Filter events and matches by team
    team_events = events[events["team.id"] == int(teamId)]
    team_matches = matches[
        (matches["home_team_id"] == int(teamId)) | (matches["away_team_id"] == int(teamId))
    ].copy()

    return DashboardResponse(
        kpis=svc.get_kpi_summary(team_events, team_matches),
        xg_trend=svc.get_xg_goals_trend(team_events, team_matches),
        xt_heatmap=svc.get_xt_zone_heatmap(team_events, xt_model),
        top_contributors=svc.get_top_player_contributions(team_events, model='xt'),
        radar=svc.get_team_radar(team_events, league_stats),
        flow=svc.get_possession_flow(team_events)
    )
