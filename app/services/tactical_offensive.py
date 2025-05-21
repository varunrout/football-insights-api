import pandas as pd
import numpy as np
from typing import List

from app.util.data_loader import shot_events, games, events_df  # <- assume you expose a full events DataFrame
from ..schemas.tactical_offensive import (
    ShotEventDTO, XgSummaryDTO, HeatPointDTO, PassNetworkDTO, PassNodeDTO, PassEdgeDTO,
    ProgressiveActionDTO, ChanceCreatorDTO
)

# ──────────────────────────────────────────────────────────────────────────
# helper
def _df() -> pd.DataFrame:
    """Return shot_events as DataFrame."""
    return pd.DataFrame([e.dict() for e in shot_events])


# ---------- 1  Shot-map ---------------------------------------------------
def get_shot_map(team_id: int, game_id: int, play_pattern: str) -> List[ShotEventDTO]:
    df = _df()
    df = df[(df.teamId == team_id) & (df.gameId == game_id)]
    if play_pattern != "all":
        df = df[df.playPattern == play_pattern]
    return [ShotEventDTO(**row) for row in df.to_dict("records")]


# ---------- 2  xG summary -------------------------------------------------
def get_xg_summary(team_id: int, period_games: List[int], period_label: str) -> XgSummaryDTO:
    df = _df()
    df = df[df.gameId.isin(period_games) & (df.teamId == team_id)]
    tot_xg = df.xg.sum()
    goals  = df.isGoal.sum()
    shot_n = len(df)
    return XgSummaryDTO(
        teamId=team_id,
        teamName=next(g.name for g in games if g.id in period_games) if period_games else "",
        periodDescription=period_label,
        totalExpectedGoals=round(float(tot_xg), 2),
        totalActualGoals=int(goals),
        shotCount=shot_n,
        matchesPlayedInPeriod=len(set(period_games)),
    )


# ---------- 3  Heat-map points -------------------------------------------
def get_attacking_third_points(team_id: int, action: str) -> List[HeatPointDTO]:
    df = events_df  # full StatsBomb events DataFrame created in data_loader
    df = df[(df["team_id"] == team_id) & (df["location"].notna())]

    # filter only attacking-third rows (x > 66.6 metres ≈ last 1/3)
    df = df[df["location"].apply(lambda loc: loc[0] > 66.6)]

    if action == "touch":
        pass  # already filtered; keep all
    else:
        df = df[df["type"] == action.title()]  # e.g., "Pass", "Carry"

    pts = [
        HeatPointDTO(x=loc[0], y=loc[1])
        for loc in df["location"]
    ]
    return pts


# ---------- 4  Final-third pass network ----------------------------------
def get_final_third_pass_network(team_id: int, game_ids: List[int]) -> PassNetworkDTO:
    df = events_df
    # passes into OR inside final third by this team
    passes = df[
        (df["type"] == "Pass") &
        (df["team_id"] == team_id) &
        (df["match_id"].isin(game_ids)) &
        (df["location"].apply(lambda l: l[0] >= 66.6))
    ]

    nodes = {}
    edges = {}

    for _, r in passes.iterrows():
        src = int(r["player_id"])
        tgt = int(r["pass_recipient_id"]) if pd.notna(r["pass_recipient_id"]) else None
        if tgt is None:
            continue

        nodes.setdefault(src, {"x": [], "y": [], "vol": 0, "name": r["player"]})
        nodes.setdefault(tgt, {"x": [], "y": [], "vol": 0, "name": r["pass_recipient"]})

        nodes[src]["x"].append(r["location"][0])
        nodes[src]["y"].append(r["location"][1])
        nodes[src]["vol"] += 1
        nodes[tgt]["x"].append(r["pass_end_location"][0])
        nodes[tgt]["y"].append(r["pass_end_location"][1])

        key = (src, tgt)
        edges.setdefault(key, {"cnt": 0, "sx": [], "sy": [], "ex": [], "ey": [], "prog": 0})
        edges[key]["cnt"] += 1
        edges[key]["sx"].append(r["location"][0])
        edges[key]["sy"].append(r["location"][1])
        edges[key]["ex"].append(r["pass_end_location"][0])
        edges[key]["ey"].append(r["pass_end_location"][1])
        # progressive rule: >30% distance toward goal
        if r["pass_end_location"][0] - r["location"][0] > 15:
            edges[key]["prog"] += 1

    node_objs = [
        PassNodeDTO(
            id=nid,
            name=val["name"],
            averageX=np.mean(val["x"]),
            averageY=np.mean(val["y"]),
            passVolume=val["vol"],
        ) for nid, val in nodes.items()
    ]

    edge_objs = [
        PassEdgeDTO(
            sourceNodeId=s,
            targetNodeId=t,
            passCount=val["cnt"],
            averagePassStartX=np.mean(val["sx"]),
            averagePassStartY=np.mean(val["sy"]),
            averagePassEndX=np.mean(val["ex"]),
            averagePassEndY=np.mean(val["ey"]),
            progressivePassCount=val["prog"],
        ) for (s, t), val in edges.items()
        if val["cnt"] >= 3  # filter noise
    ]

    return PassNetworkDTO(nodes=node_objs, edges=edge_objs)


# ---------- 5  Progressive actions ---------------------------------------
def get_progressive_actions(team_id: int, zone_name: str) -> List[ProgressiveActionDTO]:
    df = events_df

    # define middle-third zone for example
    if zone_name.lower().startswith("middle"):
        zone_filter = lambda loc: 33.3 <= loc[0] <= 66.6
    else:
        zone_filter = lambda loc: True

    prog_pass = df[
        (df["type"] == "Pass") & (df["team_id"] == team_id) &
        (df["location"].apply(zone_filter)) &
        ((df["pass_end_location"].apply(lambda l: l[0]) - df["location"].apply(lambda l: l[0])) > 15)
    ]

    prog_carry = df[
        (df["type"] == "Carry") & (df["team_id"] == team_id) &
        (df["location"].apply(zone_filter)) &
        (df["carry_end_location"].apply(lambda l: l[0]) - df["location"].apply(lambda l: l[0]) > 15)
    ]

    rows = []
    for _, r in pd.concat([prog_pass, prog_carry]).iterrows():
        act_type = "pass" if r["type"] == "Pass" else "carry"
        end_loc = r["pass_end_location"] if act_type == "pass" else r["carry_end_location"]
        rows.append(
            ProgressiveActionDTO(
                id=r["id"],
                playerId=int(r["player_id"]),
                playerName=r["player"],
                actionType=act_type,
                startX=r["location"][0],
                startY=r["location"][1],
                endX=end_loc[0],
                endY=end_loc[1],
                originatingZoneName=zone_name,
                distanceProgressed=end_loc[0] - r["location"][0],
                isSuccessful=r.get("pass_outcome") in (np.nan, None),
                timestamp=r["timestamp"],
            )
        )
    return rows


# ---------- 6  Top chance creators ---------------------------------------
def get_top_creators(team_id: int, limit: int = 10) -> List[ChanceCreatorDTO]:
    df = events_df
    # key pass = pass that assists a shot
    kp = df[df["pass_shot_assist"].notna() & (df["team_id"] == team_id)]

    minutes = df[df["team_id"] == team_id].groupby("player_id")["minute"].count()  # rough
    kps = kp.groupby("player_id").size()

    xA = df[(df["type"] == "Pass") & (df["team_id"] == team_id) & (df["pass_xa"].notna())] \
            .groupby("player_id")["pass_xa"].sum()

    assists = df[(df["type"] == "Pass") & (df["team_id"] == team_id) &
                 (df["pass_goal_assist"] == True)].groupby("player_id").size()

    tbl = pd.concat([kps, xA, assists, minutes], axis=1).fillna(0)
    tbl.columns = ["kp", "xa", "ast", "mins"]
    tbl["kp90"] = tbl.kp / (tbl.mins / 90 + 1e-9)
    tbl["xa90"] = tbl.xa / (tbl.mins / 90 + 1e-9)
    tbl = tbl.sort_values("kp90", ascending=False).head(limit)

    out = []
    for pid, row in tbl.iterrows():
        out.append(
            ChanceCreatorDTO(
                playerId=int(pid),
                playerName=df[df["player_id"] == pid]["player"].iloc[0],
                teamName=df[df["player_id"] == pid]["team"].iloc[0],
                matchesPlayed=int(row.mins // 90),
                keyPasses=int(row.kp),
                keyPassesPer90=round(row.kp90, 2),
                expectedAssists=round(row.xa, 2),
                expectedAssistsPer90=round(row.xa90, 2),
                actualAssists=int(row.ast),
            )
        )
    return out
