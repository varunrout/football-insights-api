from pydantic import BaseModel
from typing import List, Dict

class KPIData(BaseModel):
    goals: int
    xg: float
    assists: int
    xa: float
    ppda: float
    xga: float
    clean_sheets: int
    possession: float
    set_piece_goals: int

class MatchTrendPoint(BaseModel):
    match_id: int
    date: str
    xg: float
    goals: int

class ZoneXT(BaseModel):
    zone: List[int]
    xt: float

class PlayerContribution(BaseModel):
    player_name: str
    value: float
    metric: str

class RadarChartData(BaseModel):
    metrics: List[str]
    values: List[float]

class PossessionFlow(BaseModel):
    source: str
    target: str
    value: int

class DashboardResponse(BaseModel):
    kpis: KPIData
    xg_trend: List[MatchTrendPoint]
    xt_heatmap: List[ZoneXT]
    top_contributors: List[PlayerContribution]
    radar: RadarChartData
    flow: List[PossessionFlow]
