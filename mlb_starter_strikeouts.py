#!/usr/bin/env python3
"""
Compile daily MLB starting-pitcher strikeouts.

For a given date, this pulls that day's schedule from the free MLB Stats API,
then for every FINAL game grabs the box score and records each starting
pitcher (the first pitcher each team used) and how many strikeouts they got.

Output:
  - Prints a formatted list to the console.
  - Appends each starter's line to a CSV (mlb_starter_ks.csv) so the file
    builds up day over day.

No API key required. Only dependency is `requests` (pip install requests).

Usage:
  python mlb_starter_strikeouts.py                # yesterday (US/Eastern)
  python mlb_starter_strikeouts.py 2026-07-17     # a specific date
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API = "https://statsapi.mlb.com/api/v1"
CSV_PATH = Path(__file__).with_name("mlb_starter_ks.csv")
TIMEOUT = 20


def target_date() -> str:
    """Return the date to compile as YYYY-MM-DD.

    Defaults to 'yesterday' in US/Eastern, which is the natural window once a
    full slate has finished. Pass a date on the command line to override.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]
    yesterday = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    return yesterday.isoformat()


def get_games(date: str) -> list[dict]:
    """Return the list of game entries scheduled on `date`."""
    r = requests.get(
        f"{API}/schedule",
        params={"sportId": 1, "date": date},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    dates = r.json().get("dates", [])
    return dates[0]["games"] if dates else []


def starters_from_boxscore(game_pk: int) -> list[dict]:
    """Return one row per starting pitcher in a single game.

    The box score lists each team's pitchers in the order they appeared, so
    element [0] is whoever actually started (an 'opener' counts, which is what
    you want — it reflects reality, not the pre-game projection).
    """
    r = requests.get(f"{API}/game/{game_pk}/boxscore", timeout=TIMEOUT)
    r.raise_for_status()
    teams = r.json()["teams"]

    rows = []
    for side in ("away", "home"):
        team = teams[side]
        team_name = team["team"]["name"]
        pitcher_ids = team.get("pitchers", [])
        if not pitcher_ids:
            continue  # no pitchers recorded yet
        starter_id = pitcher_ids[0]
        player = team["players"][f"ID{starter_id}"]
        name = player["person"]["fullName"]
        ks = player["stats"]["pitching"].get("strikeOuts", 0)
        rows.append({"team": team_name, "starter": name, "strikeouts": int(ks)})
    return rows


def compile_day(date: str) -> list[dict]:
    games = get_games(date)
    results = []
    for g in games:
        # Only completed games have final strikeout numbers.
        if g.get("status", {}).get("abstractGameState") != "Final":
            continue

        away = g["teams"]["away"]["team"]["name"]
        home = g["teams"]["home"]["team"]["name"]
        matchup = f"{away} @ {home}"

        # Doubleheaders come through as separate games with game numbers.
        dh = g.get("gameNumber")
        if g.get("doubleHeader") in ("Y", "S") and dh:
            matchup += f" (G{dh})"

        for row in starters_from_boxscore(g["gamePk"]):
            row["date"] = date
            row["matchup"] = matchup
            results.append(row)
    return results


def write_csv(rows: list[dict]) -> None:
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "matchup", "team", "starter", "strikeouts"]
        )
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: row[k] for k in ["date", "matchup", "team", "starter", "strikeouts"]}
            )


def print_list(date: str, rows: list[dict]) -> None:
    print(f"\nStarting pitcher strikeouts — {date}")
    print("=" * 52)
    if not rows:
        print("No final games found for this date.")
        return
    current = None
    for row in rows:
        if row["matchup"] != current:
            current = row["matchup"]
            print(f"\n{current}")
        print(f"  {row['starter']:<24} ({row['team'][:3].upper()})  {row['strikeouts']:>2} K")
    print()


def main() -> None:
    date = target_date()
    try:
        rows = compile_day(date)
    except requests.RequestException as e:
        print(f"Error reaching MLB Stats API: {e}", file=sys.stderr)
        sys.exit(1)
    print_list(date, rows)
    if rows:
        write_csv(rows)
        print(f"Appended {len(rows)} rows to {CSV_PATH.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
