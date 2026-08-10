#!/usr/bin/env python3
"""
Daily MLB starting-pitcher scouting board (pitcher core).

For each of today's games it builds, per starter:
  - season line: handedness, ERA, K/9, innings, strikeouts, avg pitches/start
  - last 4 starts: IP / K / BB / pitches-strikes / hits
  - season handedness splits (BA against LHB and RHB) with sample sizes
  - short auto-generated scouting notes (durability, hot form, K streak, platoon)
  - an opponent-adjusted expected-K figure (one descriptive number, no bet odds)

Outputs: stdout text (emailed), index.html (GitHub Pages), mlb_previews.csv (log).

NOTE: this is the pitcher layer. Hitter streaks, team leans, and Statcast
whiff metrics are separate, later layers.

Only dependency: requests (pip install requests). No API key.
"""

import csv
import html
import math
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API = "https://statsapi.mlb.com/api/v1"
HERE = Path(__file__).parent
CSV_PATH = HERE / "mlb_previews.csv"
HTML_PATH = HERE / "index.html"
ET = ZoneInfo("America/New_York")
TIMEOUT = 20
LEAGUE_K_RATE = 0.22

# ---- Tunable thresholds (edit freely) ------------------------------------- #
WORKHORSE_P90 = 6      # 90+ pitches in this many of last 10 -> "workhorse"
DURABLE_P80 = 8        # 80+ pitches in this many of last 10 -> "durable"
DEEP_5IP = 8           # 5+ IP in this many of last 10 -> "works into the 6th"
K9_MISSES_BATS = 9.0   # season K/9 at/above this -> "misses bats"
HOT_ERA_LAST4 = 2.75   # ERA over last 4 starts at/below this -> "rolling"
DEEP6_LAST6 = 5        # 6+ IP in this many of last 6 -> "going deep"
TEAM_WINS_LAST10 = 7   # team won this many of last 10 starts -> note
KSTREAK_MIN = 5        # 5+ K in each of this many straight starts -> streak
TOUGH_SIDE_BA = 0.220  # opp BA at/below this (with sample) -> "tough on <side>"
TOUGH_SIDE_MIN_AB = 60


# --------------------------------------------------------------------------- #
def target_date() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return datetime.now(ET).date().isoformat()


def ip_to_float(ip) -> float:
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole or 0) + (int(frac or 0) / 3)
    except (ValueError, TypeError):
        return 0.0


def game_time_et(iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(ET)
        return dt.strftime("%-I:%M %p ET")
    except (ValueError, TypeError):
        return "time TBD"


def poisson_at_least(n, lam):
    if lam <= 0 or n <= 0:
        return 1.0 if n <= 0 else 0.0
    cum, term = 0.0, math.exp(-lam)
    for k in range(n):
        cum += term
        term *= lam / (k + 1)
    return max(0.0, min(1.0, 1 - cum))


# --------------------------------------------------------------------------- #
# Reference data (cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def team_abbr_map() -> dict:
    try:
        r = requests.get(f"{API}/teams", params={"sportId": 1}, timeout=TIMEOUT)
        r.raise_for_status()
        return {t["id"]: t["abbreviation"] for t in r.json().get("teams", [])}
    except (requests.RequestException, KeyError):
        return {}


def code_for(team_id, name):
    return team_abbr_map().get(team_id) or (name or "")[:3].upper()


@lru_cache(maxsize=64)
def team_k_rate(team_id, season):
    if not team_id:
        return None
    try:
        r = requests.get(f"{API}/teams/{team_id}/stats",
                         params={"stats": "season", "group": "hitting", "season": season},
                         timeout=TIMEOUT)
        st = r.json()["stats"][0]["splits"][0]["stat"]
        so, pa = float(st.get("strikeOuts") or 0), float(st.get("plateAppearances") or 0)
        return so / pa if pa else None
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


@lru_cache(maxsize=64)
def team_results_map(team_id, season):
    """gamePk -> True/False (did this team win) for finished games."""
    out = {}
    try:
        r = requests.get(f"{API}/schedule",
                         params={"sportId": 1, "teamId": team_id, "season": season, "gameType": "R"},
                         timeout=TIMEOUT).json()
        for d in r.get("dates", []):
            for g in d.get("games", []):
                for side in ("home", "away"):
                    t = g["teams"][side]
                    if t["team"].get("id") == team_id and t.get("isWinner") is not None:
                        out[g["gamePk"]] = bool(t["isWinner"])
    except (requests.RequestException, KeyError):
        pass
    return out


# --------------------------------------------------------------------------- #
# Per-pitcher data
# --------------------------------------------------------------------------- #
def gamelog_starts(pid, season):
    """List of this pitcher's starts, oldest -> newest."""
    try:
        g = requests.get(f"{API}/people/{pid}/stats",
                         params={"stats": "gameLog", "group": "pitching", "season": season},
                         timeout=TIMEOUT).json()
        splits = g["stats"][0]["splits"]
    except (requests.RequestException, KeyError, IndexError):
        return []
    starts = []
    for sp in splits:
        st = sp["stat"]
        if int(st.get("gamesStarted") or 0) != 1:
            continue
        opp = sp.get("opponent", {})
        starts.append({
            "gamePk": sp.get("game", {}).get("gamePk"),
            "date": sp.get("date", ""),
            "team_id": sp.get("team", {}).get("id"),
            "opp": f"{'vs' if sp.get('isHome') else '@'} {code_for(opp.get('id'), opp.get('name'))}",
            "ip_str": st.get("inningsPitched", "0.0"),
            "ip": ip_to_float(st.get("inningsPitched")),
            "k": int(st.get("strikeOuts") or 0),
            "bb": int(st.get("baseOnBalls") or 0),
            "pitches": int(st.get("numberOfPitches", st.get("pitchesThrown", 0)) or 0),
            "strikes": int(st.get("strikes") or 0),
            "hits": int(st.get("hits") or 0),
            "er": int(st.get("earnedRuns") or 0),
        })
    return starts


def pitcher_season(pid, season):
    prof = {"hand": "", "era": None, "k9": None, "ip": None, "k": None,
            "avg_pitches": None, "k_pct": None, "bf_per_start": None, "ip_per_start": None}
    try:
        p = requests.get(f"{API}/people/{pid}", timeout=TIMEOUT).json()
        prof["hand"] = p["people"][0].get("pitchHand", {}).get("code", "")
    except (requests.RequestException, KeyError, IndexError):
        pass
    try:
        s = requests.get(f"{API}/people/{pid}/stats",
                         params={"stats": "season", "group": "pitching", "season": season},
                         timeout=TIMEOUT).json()
        st = s["stats"][0]["splits"][0]["stat"]
        prof["era"] = st.get("era")
        prof["k9"] = st.get("strikeoutsPer9Inn")
        prof["ip"] = st.get("inningsPitched")
        prof["k"] = int(st.get("strikeOuts") or 0)
        gs = int(st.get("gamesStarted") or 0)
        bf = int(st.get("battersFaced") or 0)
        if gs:
            prof["ip_per_start"] = round(ip_to_float(st.get("inningsPitched")) / gs, 1)
            if bf:
                prof["bf_per_start"] = bf / gs
        if bf:
            prof["k_pct"] = int(st.get("strikeOuts") or 0) / bf
    except (requests.RequestException, KeyError, IndexError, ValueError):
        pass
    return prof


def pitcher_vshand(pid, season):
    """Season BA-against split vs LHB and RHB, with at-bats."""
    out = {"L": None, "R": None}
    try:
        s = requests.get(f"{API}/people/{pid}/stats",
                         params={"stats": "statSplits", "group": "pitching",
                                 "season": season, "sitCodes": "vl,vr"},
                         timeout=TIMEOUT).json()
        for sp in s["stats"][0]["splits"]:
            code = sp.get("split", {}).get("code", "")
            side = "L" if code == "vl" else "R" if code == "vr" else None
            if side:
                st = sp["stat"]
                out[side] = {"avg": st.get("avg"), "ab": int(st.get("atBats") or 0)}
    except (requests.RequestException, KeyError, IndexError, ValueError):
        pass
    return out


def avg_pitches_per_start(starts):
    p = [s["pitches"] for s in starts if s["pitches"] > 0]
    return round(sum(p) / len(p)) if p else None


def project_k(prof, opp_team_id, season):
    bf = prof.get("bf_per_start") or (prof.get("ip_per_start") or 5.0) * 4.3
    p_pit = prof.get("k_pct") or LEAGUE_K_RATE
    p_opp = team_k_rate(opp_team_id, season) or LEAGUE_K_RATE
    lam = max(0.05, min(0.50, (p_pit * p_opp) / LEAGUE_K_RATE)) * bf
    return round(lam, 1)


def clean_avg(a):
    """'.198' style, dropping a leading zero."""
    try:
        return f"{float(a):.3f}".lstrip("0") or ".000"
    except (ValueError, TypeError):
        return str(a)


def pitcher_notes(name, starts, prof, vshand, season):
    """Return a list of short scouting sentences from the thresholds above."""
    notes = []
    last10 = starts[-10:]
    n10 = len(last10)

    # Durability / usage
    if n10 >= 5:
        p90 = sum(1 for s in last10 if s["pitches"] >= 90)
        p80 = sum(1 for s in last10 if s["pitches"] >= 80)
        deep5 = sum(1 for s in last10 if s["ip"] >= 5.0)
        if p90 >= WORKHORSE_P90:
            notes.append(f"Workhorse — 90+ pitches in {p90} of his last {n10} starts.")
        elif p80 >= DURABLE_P80:
            notes.append(f"Durable — 80+ pitches in {p80} of his last {n10} starts.")
        if deep5 >= DEEP_5IP:
            notes.append(f"Works into the 6th regularly — 5+ IP in {deep5} of his last {n10}.")

    # Misses bats
    try:
        if prof.get("k9") and float(prof["k9"]) >= K9_MISSES_BATS:
            notes.append(f"Misses bats — {prof['k9']} K/9, about a strikeout an inning.")
    except (ValueError, TypeError):
        pass

    # Hot form: run prevention over last 4
    last4 = starts[-4:]
    ip4 = sum(s["ip"] for s in last4)
    er4 = sum(s["er"] for s in last4)
    if len(last4) == 4 and ip4 > 0:
        era4 = 9 * er4 / ip4
        if era4 <= HOT_ERA_LAST4:
            notes.append(f"Rolling — {era4:.2f} ERA over his last 4 ({er4} ER in {ip4:.1f} IP).")

    # Going deep lately
    last6 = starts[-6:]
    if len(last6) == 6:
        deep6 = sum(1 for s in last6 if s["ip"] >= 6.0)
        if deep6 >= DEEP6_LAST6:
            notes.append(f"Going deep — 6+ innings in {deep6} of his last 6 starts.")

    # Team record in his starts
    if last10 and last10[0]["team_id"]:
        results = team_results_map(last10[0]["team_id"], season)
        decided = [results[s["gamePk"]] for s in last10 if s["gamePk"] in results]
        wins = sum(1 for w in decided if w)
        if len(decided) >= 8 and wins >= TEAM_WINS_LAST10:
            notes.append(f"Team has won {wins} of his last {len(decided)} starts.")

    # Strikeout streak
    streak = 0
    for s in reversed(starts):
        if s["k"] >= 5:
            streak += 1
        else:
            break
    if streak >= KSTREAK_MIN:
        notes.append(f"Strikeout run — 5+ K in {streak} straight starts.")

    # Handedness
    for side, label in (("L", "left"), ("R", "right")):
        v = vshand.get(side)
        if v and v.get("avg") and v.get("ab", 0) >= TOUGH_SIDE_MIN_AB:
            try:
                if float(v["avg"]) <= TOUGH_SIDE_BA:
                    notes.append(f"Tough on {label}-handed bats — {clean_avg(v['avg'])} against ({v['ab']} AB).")
            except (ValueError, TypeError):
                pass

    return notes[:4]


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
def todays_games(date):
    r = requests.get(f"{API}/schedule",
                     params={"sportId": 1, "date": date, "hydrate": "probablePitcher"},
                     timeout=TIMEOUT)
    r.raise_for_status()
    dates = r.json().get("dates", [])
    games = dates[0]["games"] if dates else []
    out = []
    for g in games:
        a, h = g["teams"]["away"], g["teams"]["home"]
        out.append({
            "time": game_time_et(g.get("gameDate", "")),
            "away_name": a["team"]["name"], "home_name": h["team"]["name"],
            "away_id": a["team"].get("id"), "home_id": h["team"].get("id"),
            "away_code": code_for(a["team"].get("id"), a["team"]["name"]),
            "home_code": code_for(h["team"].get("id"), h["team"]["name"]),
            "away_pitcher": a.get("probablePitcher"), "home_pitcher": h.get("probablePitcher"),
        })
    return out


def build_arm(pitcher, code, opp_id, season):
    if not pitcher:
        return {"tbd": True, "code": code}
    pid = pitcher["id"]
    starts = gamelog_starts(pid, season)
    prof = pitcher_season(pid, season)
    vshand = pitcher_vshand(pid, season)
    last4 = list(reversed(starts[-4:]))  # newest first
    return {
        "tbd": False, "code": code,
        "name": pitcher.get("fullName", "Unknown"),
        "hand": {"L": "LHP", "R": "RHP"}.get(prof["hand"], ""),
        "era": prof["era"], "k9": prof["k9"], "ip": prof["ip"], "k": prof["k"],
        "avg_pitches": avg_pitches_per_start(starts),
        "vshand": vshand, "last4": last4,
        "notes": pitcher_notes(pitcher.get("fullName", ""), starts, prof, vshand, season),
        "exp_k": project_k(prof, opp_id, season),
    }


def build(date):
    season = int(date[:4])
    games = []
    for g in todays_games(date):
        games.append({
            "time": g["time"], "away_code": g["away_code"], "home_code": g["home_code"],
            "away_name": g["away_name"], "home_name": g["home_name"],
            "arms": [
                build_arm(g["away_pitcher"], g["away_code"], g["home_id"], season),
                build_arm(g["home_pitcher"], g["home_code"], g["away_id"], season),
            ],
        })
    return games


# --------------------------------------------------------------------------- #
# Text + CSV
# --------------------------------------------------------------------------- #
def text_report(date, games):
    out = [f"Probable starters — {date}", "=" * 56]
    if not games:
        out.append("No games scheduled today.")
        return "\n".join(out)
    for g in games:
        out.append(f"\n{g['away_name']} @ {g['home_name']} — {g['time']}")
        for a in g["arms"]:
            if a["tbd"]:
                out.append(f"  {a['code']}: starter TBD")
                continue
            hand = f", {a['hand']}" if a["hand"] else ""
            out.append(f"  {a['name']} ({a['code']}{hand})")
            out.append(f"    {a['era']} ERA · {a['k9']} K/9 · {a['ip']} IP · {a['k']} K · "
                       f"{a['avg_pitches']} P/GS · proj ~{a['exp_k']} K")
            for note in a["notes"]:
                out.append(f"    • {note}")
            if a["last4"]:
                out.append("    Last 4 (IP/K/BB/P-S/H):")
                for s in a["last4"]:
                    out.append(f"      {s['date'][5:]} {s['opp']:<7} "
                               f"{s['ip_str']}/{s['k']}/{s['bb']}/{s['pitches']}-{s['strikes']}/{s['hits']}")
    return "\n".join(out)


def csv_rows(date, games):
    rows = []
    for g in games:
        m = f"{g['away_code']} @ {g['home_code']}"
        for a in g["arms"]:
            if a["tbd"]:
                continue
            rows.append({"date": date, "matchup": m, "code": a["code"], "pitcher": a["name"],
                         "hand": a["hand"], "era": a["era"], "k9": a["k9"], "ip": a["ip"],
                         "k": a["k"], "avg_pitches": a["avg_pitches"], "exp_k": a["exp_k"],
                         "notes": " | ".join(a["notes"])})
    return rows


def write_csv(rows):
    cols = ["date", "matchup", "code", "pitcher", "hand", "era", "k9", "ip", "k",
            "avg_pitches", "exp_k", "notes"]
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# --------------------------------------------------------------------------- #
# HTML  (see mlb_board_preview.html for the matching design)
# --------------------------------------------------------------------------- #
def esc(x):
    return html.escape(str(x if x is not None else ""))


HTML_HEAD = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Probable Starters</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--ink:#0e1513;--card:#14201c;--raise:#1b2a25;--chalk:#f1eee4;--dim:#9aa79c;
--line:rgba(241,238,228,.10);--clay:#cf7a3c;--turf:#5aa06a;--signal:#f0b429;--called:#e0563b;
--disp:"Saira Condensed",sans-serif;--body:"IBM Plex Sans",sans-serif;--mono:"IBM Plex Mono",monospace;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ink);color:var(--chalk);font-family:var(--body);line-height:1.5;
background-image:radial-gradient(circle at 50% -10%,rgba(90,160,106,.10),transparent 55%)}
.wrap{max-width:1140px;margin:0 auto;padding:32px 20px 72px}
.mast{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;padding-bottom:16px;border-bottom:2px solid var(--line);flex-wrap:wrap}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--turf)}
h1{font-family:var(--disp);font-weight:700;font-size:clamp(38px,7vw,68px);line-height:.92;text-transform:uppercase;margin-top:6px}
h1 .k{color:var(--signal)}
.mast-right{text-align:right;font-family:var(--mono);font-size:13px;color:var(--dim)}
.mast-right .date{color:var(--chalk);font-size:15px}
.games{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px;margin-top:28px}
@media(max-width:560px){.games{grid-template-columns:1fr}}
.game{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;opacity:0;transform:translateY(10px);animation:rise .5s ease forwards}
@keyframes rise{to{opacity:1;transform:none}}
.game-head{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;border-bottom:1px solid var(--line);background:var(--raise)}
.matchup{font-family:var(--disp);font-weight:600;font-size:22px}
.matchup .at{color:var(--dim);margin:0 6px;font-size:16px}
.gtime{font-family:var(--mono);font-size:12px;color:var(--clay)}
.arms{display:grid;grid-template-columns:1fr 1fr}
@media(max-width:560px){.arms{grid-template-columns:1fr}}
.arm{padding:18px}
.arm+.arm{border-left:1px solid var(--line)}
@media(max-width:560px){.arm+.arm{border-left:none;border-top:1px solid var(--line)}}
.arm-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.pname{font-family:var(--disp);font-weight:600;font-size:26px;line-height:1}
.pcode{font-family:var(--mono);font-size:11px;color:var(--dim);letter-spacing:.14em;text-transform:uppercase}
.hand{font-family:var(--mono);font-size:11px;color:var(--ink);background:var(--turf);padding:2px 7px;border-radius:4px;letter-spacing:.06em;white-space:nowrap}
.hand.l{background:var(--clay)}
.chips{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;font-family:var(--mono);font-size:12px}
.chip .n{color:var(--chalk);font-size:15px;font-weight:600}
.chip .n.proj{color:var(--signal)}
.chip .l{color:var(--dim);font-size:10px;letter-spacing:.08em;text-transform:uppercase}
.notes{margin-top:14px;padding-top:12px;border-top:1px dashed var(--line);list-style:none;display:flex;flex-direction:column;gap:6px}
.note{position:relative;padding-left:16px;font-size:13px;color:var(--chalk)}
.note::before{content:"K";position:absolute;left:0;font-family:var(--disp);font-weight:700;color:var(--turf);font-size:12px}
.split{margin-top:12px;font-family:var(--mono);font-size:11px;color:var(--dim)}
.split b{color:var(--chalk);font-weight:600}
.last4{margin-top:14px}
.last4 .cap{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);margin-bottom:6px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px}
th{color:var(--dim);font-weight:500;text-align:right;padding:3px 0;font-size:10px;text-transform:uppercase}
th:first-child,td:first-child{text-align:left}
td{padding:3px 0;border-top:1px solid var(--line);color:var(--chalk)}
td .k{color:var(--signal);font-weight:600}
.tbd{padding:30px 18px;color:var(--dim);font-family:var(--mono);font-size:13px;text-align:center}
.foot{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11px;color:var(--dim);line-height:1.7}
.foot b{color:var(--clay);font-weight:500}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body><div class="wrap">
"""

HTML_TAIL = """  <footer class="foot">
  <div><b>Notes</b> are auto-generated from recent game logs and season splits — descriptive flags, not bet advice.</div>
  <div>Sample sizes shown so you can weigh them. Rebuilt every morning. Hitter streaks & Statcast coming next.</div>
  </footer></div></body></html>
"""


def html_arm(a):
    if a["tbd"]:
        return (f'<div class="arm"><div class="arm-top"><div><div class="pcode">{esc(a["code"])}</div>'
                f'<div class="pname">Starter TBD</div></div></div>'
                f'<div class="tbd">Not yet announced —<br>fills in once the team posts its starter.</div></div>')
    hand = ""
    if a["hand"]:
        hand = f'<span class="{"hand l" if a["hand"]=="LHP" else "hand"}">{esc(a["hand"])}</span>'
    chips = [("era", a["era"], "ERA"), ("k9", a["k9"], "K/9"), ("ip", a["ip"], "IP"),
             ("k", a["k"], "K"), ("ap", a["avg_pitches"], "P/GS")]
    chips_html = "".join(
        f'<div class="chip"><div class="n">{esc(v if v is not None else "—")}</div><div class="l">{l}</div></div>'
        for _, v, l in chips
    )
    chips_html += f'<div class="chip"><div class="n proj">{esc(a["exp_k"])}</div><div class="l">proj K</div></div>'

    notes = "".join(f'<li class="note">{esc(n)}</li>' for n in a["notes"])
    notes_html = f'<ul class="notes">{notes}</ul>' if notes else ""

    parts = []
    vl, vr = a["vshand"].get("L"), a["vshand"].get("R")
    if vl and vl.get("avg"):
        parts.append(f'vs LHB <b>{esc(clean_avg(vl["avg"]))}</b> ({vl["ab"]} AB)')
    if vr and vr.get("avg"):
        parts.append(f'vs RHB <b>{esc(clean_avg(vr["avg"]))}</b> ({vr["ab"]} AB)')
    split_html = f'<div class="split">{" · ".join(parts)}</div>' if parts else ""

    if a["last4"]:
        body = "".join(
            f'<tr><td>{esc(s["date"][5:])}</td><td>{esc(s["opp"])}</td><td>{esc(s["ip_str"])}</td>'
            f'<td class="k">{esc(s["k"])}</td><td>{esc(s["bb"])}</td>'
            f'<td>{esc(s["pitches"])}-{esc(s["strikes"])}</td><td>{esc(s["hits"])}</td></tr>'
            for s in a["last4"]
        )
        last4 = ('<div class="last4"><div class="cap">Last 4 starts</div><table>'
                 '<tr><th>Date</th><th>Opp</th><th>IP</th><th>K</th><th>BB</th><th>P-S</th><th>H</th></tr>'
                 f'{body}</table></div>')
    else:
        last4 = '<div class="last4"><div class="cap">Last 4 starts</div><div class="pcode">no game-log data</div></div>'

    return (f'<div class="arm"><div class="arm-top"><div><div class="pcode">{esc(a["code"])}</div>'
            f'<div class="pname">{esc(a["name"])}</div></div>{hand}</div>'
            f'<div class="chips">{chips_html}</div>{notes_html}{split_html}{last4}</div>')


def html_report(date, games):
    try:
        pretty = datetime.fromisoformat(date).strftime("%a · %b %-d, %Y")
    except ValueError:
        pretty = date
    p = [HTML_HEAD,
         '<header class="mast"><div><div class="eyebrow">MLB · Probable Pitchers</div>'
         '<h1>The Bump <span class="k">ꓘ</span></h1></div>'
         f'<div class="mast-right"><div class="date">{esc(pretty)}</div>'
         f'<div>{len(games)} game{"s" if len(games)!=1 else ""} on the card</div></div></header>']
    if not games:
        p.append('<section class="games"><div class="tbd">No games scheduled today.</div></section>')
    else:
        cards = []
        for i, g in enumerate(games):
            arms = "".join(html_arm(a) for a in g["arms"])
            cards.append(f'<article class="game" style="animation-delay:{i*0.05:.2f}s">'
                         f'<div class="game-head"><div class="matchup">{esc(g["away_code"])}'
                         f'<span class="at">@</span>{esc(g["home_code"])}</div>'
                         f'<div class="gtime">{esc(g["time"])}</div></div>'
                         f'<div class="arms">{arms}</div></article>')
        p.append(f'<section class="games">{"".join(cards)}</section>')
    p.append(HTML_TAIL)
    return "".join(p)


def main():
    date = target_date()
    try:
        games = build(date)
    except requests.RequestException as e:
        print(f"Error reaching MLB Stats API: {e}", file=sys.stderr)
        sys.exit(1)
    print(text_report(date, games))
    HTML_PATH.write_text(html_report(date, games), encoding="utf-8")
    print(f"Wrote {HTML_PATH.name}", file=sys.stderr)
    rows = csv_rows(date, games)
    if rows:
        write_csv(rows)
        print(f"Logged {len(rows)} starters to {CSV_PATH.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
