from fastapi import APIRouter, Query
from typing import List

from ..schemas.tactical_offensive import (
    ShotEventDTO, XgSummaryDTO, HeatPointDTO, PassNetworkDTO,
    ProgressiveActionDTO, ChanceCreatorDTO
)
from ..services import tactical_offensive as svc

router = APIRouter(
    prefix="/api/tactical-insights/offensive",
    tags=["Tactical-Insights ⇢ Offensive"]
)

# 1 ───────────────────────────────────────────────────────────────────────
@router.get(
    "/shot-map",
    response_model=List[ShotEventDTO],
    summary="Integrated shot map for one match"
)
def shot_map(teamId: int, gameId: int, playPattern: str = Query("all")):
    return svc.get_shot_map(teamId, gameId, playPattern)


# 2 ───────────────────────────────────────────────────────────────────────
@router.get(
    "/xg-summary",
    response_model=XgSummaryDTO,
    summary="xG vs Goals for a period"
)
def xg_summary(teamId: int,
               period_games: List[int] = Query(..., alias="gameIds"),
               periodDescription: str = Query("custom period")):
    return svc.get_xg_summary(teamId, period_games, periodDescription)


# 3 ───────────────────────────────────────────────────────────────────────
@router.get(
    "/attacking-third-heatmap",
    response_model=List[HeatPointDTO],
    summary="Raw point cloud for attacking third heat-map"
)
def heatmap(teamId: int, actionType: str = Query("touch")):
    return svc.get_attacking_third_points(teamId, actionType)


# 4 ───────────────────────────────────────────────────────────────────────
@router.get(
    "/final-third-pass-network",
    response_model=PassNetworkDTO,
    summary="Pass network in final third"
)
def pass_network(teamId: int, gameIds: List[int] = Query(...)):
    return svc.get_final_third_pass_network(teamId, gameIds)


# 5 ───────────────────────────────────────────────────────────────────────
@router.get(
    "/progressive-actions",
    response_model=List[ProgressiveActionDTO],
    summary="Progressive passes & carries originating from a zone"
)
def progressive_actions(teamId: int, zone: str = Query("Middle Third")):
    return svc.get_progressive_actions(teamId, zone)


# 6 ───────────────────────────────────────────────────────────────────────
@router.get(
    "/top-chance-creators",
    response_model=List[ChanceCreatorDTO],
    summary="Top chance creators table"
)
def top_creators(teamId: int, limit: int = Query(10)):
    return svc.get_top_creators(teamId, limit)
