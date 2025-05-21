from pydantic import BaseModel, Field
from uuid import UUID
from typing import List, Optional


# ---------- 1  Shot-map --------------------------------------------------------
class ShotEventDTO(BaseModel):
    id: UUID
    playerId: int
    playerName: str
    teamId: int
    teamName: str
    gameId: int
    x: float
    y: float
    isGoal: bool
    xg: float
    bodyPart: Optional[str] = None
    playPattern: str


# ---------- 2  xG summary ------------------------------------------------------
class XgSummaryDTO(BaseModel):
    teamId: int
    teamName: str
    periodDescription: str
    totalExpectedGoals: float
    totalActualGoals: int
    shotCount: int
    matchesPlayedInPeriod: int


# ---------- 3  Heat-map --------------------------------------------------------
class HeatPointDTO(BaseModel):
    x: float
    y: float
    value: Optional[float] = None


class HeatGridDTO(BaseModel):
    xBins: List[float]
    yBins: List[float]
    density: List[List[int]]
    minValue: int
    maxValue: int


# ---------- 4  Pass network ----------------------------------------------------
class PassNodeDTO(BaseModel):
    id: int
    name: str
    averageX: float
    averageY: float
    passVolume: int


class PassEdgeDTO(BaseModel):
    sourceNodeId: int
    targetNodeId: int
    passCount: int
    averagePassStartX: float
    averagePassStartY: float
    averagePassEndX: float
    averagePassEndY: float
    progressivePassCount: int


class PassNetworkDTO(BaseModel):
    nodes: List[PassNodeDTO]
    edges: List[PassEdgeDTO]


# ---------- 5  Progressive actions --------------------------------------------
class ProgressiveActionDTO(BaseModel):
    id: UUID
    playerId: int
    playerName: str
    actionType: str  # pass | carry
    startX: float
    startY: float
    endX: float
    endY: float
    originatingZoneName: str
    distanceProgressed: float
    isSuccessful: bool
    timestamp: str


# ---------- 6  Chance creators table ------------------------------------------
class ChanceCreatorDTO(BaseModel):
    playerId: int
    playerName: str
    teamName: str
    matchesPlayed: int
    keyPasses: int
    keyPassesPer90: float
    expectedAssists: float
    expectedAssistsPer90: float
    actualAssists: int
