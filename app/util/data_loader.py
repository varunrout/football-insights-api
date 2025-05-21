# # app/data_loader.py
#
# import logging
# import pandas as pd
# from statsbombpy import sb
# from app.util.pydantic_models import Team, Player, Game, ShotEvent, TeamPerformance, PlayerStats
# from app.util.config import settings
#
# # ─── Logging Setup ───────────────────────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # ─── In-Memory Stores ────────────────────────────────────────────────────────────
# teams: list[Team]             = []
# players: list[Player]         = []
# games: list[Game]             = []
# shot_events: list[ShotEvent]  = []
# team_performances: list[TeamPerformance] = []
# player_stats: list[PlayerStats]          = []
# events_df: pd.DataFrame | None = None   # type hint for IDE
#
# def load_statsbomb_data():
#     global teams, players, games, shot_events, team_performances, player_stats, events_df
#
#     # reset everything
#     teams = []
#     players = []
#     games = []
#     shot_events = []
#     team_performances = []
#     player_stats = []
#     events_frames = []
#
#     # 1) Pick competition & season
#     comp_name   = "1. Bundesliga"
#     season_name = "2023/2024"
#
#     comps = sb.competitions()
#     logger.info(f"[comps] columns: {comps.columns.tolist()}")
#     logger.info(f"[comps] sample:\n{comps.head().to_string(index=False)}")
#
#     mask     = (comps["competition_name"] == comp_name) & (comps["season_name"] == season_name)
#     filtered = comps.loc[mask]
#     if filtered.empty:
#         raise RuntimeError(
#             f"No competition+season for '{comp_name}'/'{season_name}'.\n"
#             f"Available combos:\n"
#             f"{comps[['competition_name','season_name']].drop_duplicates().to_string(index=False)}"
#         )
#     comp_row = filtered.iloc[0]
#     comp_id, season_id = int(comp_row["competition_id"]), int(comp_row["season_id"])
#
#     # 2) Fetch matches
#     df_matches = sb.matches(competition_id=comp_id, season_id=season_id)
#     logger.info(f"[matches] columns: {df_matches.columns.tolist()}")
#     logger.info(f"[matches] sample:\n{df_matches.head().to_string(index=False)}")
#
#     for _, r in df_matches.iterrows():
#         mid = int(r["match_id"])
#         ev = sb.events(match_id=mid)
#         events_frames.append(ev)
#
#         home = r.get("home_team") or r.get("home_team_name") or "Unknown"
#         away = r.get("away_team") or r.get("away_team_name") or "Unknown"
#         games.append(Game(
#             id=int(r["match_id"]),
#             name=f"{home} vs {away}",
#             date=r["match_date"]
#         ))
#     # 🆕 Build the full DataFrame once
#     events_df = pd.concat(events_frames, ignore_index=True)
#     logger.info(f"[data_loader] events_df shape = {events_df.shape}")
#
#     # 3) Fetch & parse shot events
#     for _, match in df_matches.iterrows():
#         mid = int(match["match_id"])
#         ev  = sb.events(match_id=mid)
#
#         logger.info(f"[events for match {mid}] columns: {ev.columns.tolist()}")
#         logger.info(f"[events sample]\n{ev.head(1).to_string(index=False)}")
#
#         if "type" not in ev.columns:
#             logger.error(f"'type' column missing for match {mid}; skipping.")
#             continue
#
#         shots = ev[ev["type"] == "Shot"]
#         logger.info(f"Found {len(shots)} shots for match {mid}")
#
#         for _, s in shots.iterrows():
#             # extract required fields, skipping any NaNs
#             sid = s["id"]  # UUID string
#             pid = s["player_id"] if pd.notna(s.get("player_id")) else None
#             tid = s["team_id"]   if pd.notna(s.get("team_id"))   else None
#
#             loc = s.get("location", [None, None])
#             x = float(loc[0]) if isinstance(loc, (list, tuple)) and loc[0] is not None else 0.0
#             y = float(loc[1]) if isinstance(loc, (list, tuple)) and loc[1] is not None else 0.0
#
#             outcome = s.get("shot_outcome", "")
#             xg_val  = s.get("shot_statsbomb_xg", 0.0)
#
#             # only add if we have numeric player and team IDs
#             if pid is None or tid is None:
#                 logger.warning(f"Skipping shot {sid} in match {mid}: missing player_id or team_id")
#                 continue
#
#             shot_events.append(ShotEvent(
#                 id       = sid,
#                 playerId = int(pid),
#                 teamId   = int(tid),
#                 gameId   = mid,
#                 x        = x,
#                 y        = y,
#                 isGoal   = (outcome == "Goal"),
#                 xg       = float(xg_val) if pd.notnull(xg_val) else 0.0,
#             ))
#
#     # 4) Build team & player lookup from a sample of events
#     sample_ev = sb.events(match_id=int(df_matches.iloc[0]["match_id"]))
#
#     team_lookup = {}
#     for _, row in sample_ev.iterrows():
#         tid = row.get("team_id")
#         if pd.isna(tid):
#             continue
#         tid = int(tid)
#         team_val = row.get("team")
#         name = team_val["name"] if isinstance(team_val, dict) else str(team_val)
#         team_lookup[tid] = name
#
#     player_lookup = {}
#     for _, row in sample_ev.iterrows():
#         pid = row.get("player_id")
#         if pd.isna(pid):
#             continue
#         pid = int(pid)
#         player_val = row.get("player")
#         name = player_val["name"] if isinstance(player_val, dict) else str(player_val)
#         player_lookup[pid] = name
#
#     # instantiate Team and Player objects
#     for se in shot_events:
#         if se.teamId not in [t.id for t in teams]:
#             teams.append(Team(id=se.teamId, name=team_lookup.get(se.teamId, f"Team {se.teamId}")))
#         if se.playerId not in [p.id for p in players]:
#             players.append(Player(
#                 id=se.playerId,
#                 name=player_lookup.get(se.playerId, f"Player {se.playerId}"),
#                 teamId=se.teamId
#             ))
#
#     # 5) Aggregate team performance
#     df_se = pd.DataFrame([e.dict() for e in shot_events])
#     tp = df_se.groupby("teamId").agg(
#         xg            = ("xg", "sum"),
#         shots         = ("id", "count"),
#         shotsOnTarget = ("isGoal", "sum")
#     ).reset_index()
#     for _, r in tp.iterrows():
#         team_performances.append(TeamPerformance(
#             teamId            = int(r["teamId"]),
#             teamName          = next(t.name for t in teams if t.id == r["teamId"]),
#             xg                = float(r["xg"]),
#             shots             = int(r["shots"]),
#             shotsOnTarget     = int(r["shotsOnTarget"]),
#             passCompletionRate= 0.0,
#             possession        = 0.0
#         ))
#
#     # 6) Aggregate player stats
#     ps = df_se.groupby("playerId").agg(
#         matchesPlayed = ("gameId", lambda g: g.nunique()),
#         goals         = ("isGoal", "sum"),
#         xg            = ("xg", "sum")
#     ).reset_index()
#     for _, r in ps.iterrows():
#         player_stats.append(PlayerStats(
#             playerId         = int(r["playerId"]),
#             playerName       = next(p.name   for p in players if p.id == r["playerId"]),
#             teamName         = next(t.name   for t in teams   if t.id ==
#                                      next(e.teamId for e in shot_events if e.playerId == r["playerId"])),
#             matchesPlayed    = int(r["matchesPlayed"]),
#             goals            = int(r["goals"]),
#             assists          = 0,
#             xg               = float(r["xg"]),
#             xa               = 0.0,
#             keyPasses        = 0,
#             passCompletionRate=0.0
#         ))
#
# def init_data():
#     if settings.ENV.upper() == "TEST":
#         # optionally call load_mock_data()
#         return
#     load_statsbomb_data()
#
# # run at import
# init_data()

# app/util/data_loader.py
import pandas as pd
from app.util.config import settings
from statsbombpy import sb
from functools import lru_cache

@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    comp_id = 11   # Bundesliga
    season_id = 281  # 2023/24
    all_events = []
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    for mid in matches.match_id.values:
        try:
            df = sb.events(match_id=mid)
            df["match_id"] = mid
            all_events.append(df)
        except Exception as e:
            print(f"Error loading match {mid}: {e}")
    return pd.concat(all_events, ignore_index=True)

@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    comp_id = 11
    season_id = 281
    return sb.matches(competition_id=comp_id, season_id=season_id)

@lru_cache(maxsize=1)
def load_xt_model():
    # Stub: Replace with actual model logic
    class DummyXTModel:
        def get_value(self, x, y):
            # Example: scale x from 0 to 1, y from 0 to 1 and multiply
            return round((x / 120) * (y / 80), 4)

    return DummyXTModel()

@lru_cache(maxsize=1)
def load_league_context():
    # Example league averages and stds, replace with real values from all teams
    return {
        "xG": {"mean": 50.0, "std": 10.0},
        "Shots": {"mean": 400, "std": 50},
        "xGA": {"mean": 40.0, "std": 10.0},
        "PPDA": {"mean": 10.0, "std": 2.0},
    }
