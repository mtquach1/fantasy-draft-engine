"""Look-ahead for "who should I pick?": will this player still be there at my NEXT turn?

The plain recommendation ranks players by what they add to your team right now. That ignores timing. If two players
are about equally good for you but one will almost surely last until your next pick and the other will not, take the
one who won't last. This module adds that.

How it works (a model, not a prophecy):
  1. The other teams are assumed to draft off a shared, noisy "ADP" board. The board is the pre-draft ranking
     (`projected_rank`) plus random noise that grows with the rank (noise = 2 + 0.12 * rank, the same heuristic the
     pace benchmark uses). Each simulated run draws new noise and sorts the players; the first picks of that order
     are the players the others take.
  2. We run `RUNS` such drafts. For each candidate we count how often he is still there when you pick
     (`p_at_my_pick`) and how often he would still be there at your pick after that (`p_next_turn`, if you passed).
  3. Two-pick plan: for each candidate c we add (a) what c is worth to you now and (b) what the best player still
     around at your following pick would then add, on top of c. Candidates are ranked by that total. The player most
     often available as that follow-up is shown as "next".

Honest limits: other teams also draft by need and by their own site's ADP, which this doesn't know. It does not know
about keepers or auction nominations. It's the same noise model as the pace benchmark, so it shares that model's
weaknesses. When real ADP is available, swap it in for `projected_rank`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RUNS = 300            # simulated drafts per call
N_CAND = 12           # candidates examined in depth (the rest keep their plain ranking)
NOISE_BASE, NOISE_SLOPE = 2.0, 0.12
MIN_P_AT_PICK = 0.02  # a player who is almost surely gone before your turn isn't a real option
N_FOLLOW = 12         # follow-up candidates examined for your NEXT pick
MIN_SURVIVE = 0.15    # a follow-up must have at least this chance of lasting until your next pick
MIN_FOLLOW_SHARE = 0.20   # name a follow-up only if he is the best choice in at least this share of simulated drafts


def rank_key(pool: pd.DataFrame) -> np.ndarray:
    """What the other teams are assumed to rank players by: real ADP when the `adp` column is there (players
    without one fall back to the pre-draft rank), otherwise the pre-draft rank alone."""
    base = pool["projected_rank"].fillna(pool["projected_rank"].max() + 1).astype(float)
    if "adp" in pool.columns and pool["adp"].notna().sum() >= max(10, 0.2 * len(pool)):
        base = pool["adp"].astype(float).where(pool["adp"].notna() & (pool["adp"] > 0), base)
    return base.to_numpy(float)


def pick_gaps(setup, picks_made: int) -> dict | None:
    """How many picks the OTHER teams make before your next pick (`k1`), and between that pick and the one after
    it (`k2`). None when look-ahead doesn't apply (auction, no setup, draft over, you have no picks left)."""
    if setup is None or setup.draft_type == "auction" or not setup.my_slot:
        return None
    from .draft import slot_for_pick
    total, cur = setup.teams * setup.rounds, picks_made + 1
    mine = [n for n in range(cur, total + 1) if slot_for_pick(setup.draft_type, setup.teams, n) == setup.my_slot]
    if not mine:
        return None
    t = mine[0]
    t2 = mine[1] if len(mine) > 1 else None
    return {"current_pick": cur, "my_pick": t, "next_pick": t2, "k1": t - cur,
            "k2": (t2 - t - 1) if t2 else 0, "on_the_clock": t == cur}


def simulate_order(ranks: np.ndarray, runs: int = RUNS, seed: int = 11) -> np.ndarray:
    """pos[r, i] = the place (0 = taken first) of player i when run r's noisy board is sorted.
    The fixed seed keeps the numbers steady from one page refresh to the next."""
    rng = np.random.default_rng(seed)
    keys = ranks[None, :] + rng.normal(0.0, 1.0, (runs, len(ranks))) * (NOISE_BASE + NOISE_SLOPE * ranks[None, :])
    order = np.argsort(keys, axis=1, kind="stable")
    pos = np.empty_like(order)
    rows = np.arange(runs)[:, None]
    pos[rows, order] = np.arange(ranks.size)[None, :]
    return pos


def survival_by_id(df: pd.DataFrame, gaps: dict, runs: int = RUNS) -> dict:
    """player_id -> chance he's still available at your next pick (every available player, same model as rerank)."""
    pool = df[df["available"]].reset_index(drop=True)
    if pool.empty:
        return {}
    rk = rank_key(pool)
    p = (simulate_order(rk, runs) >= gaps["k1"]).mean(0)
    return dict(zip(pool["player_id"], p))


def _wording(c: dict, gaps: dict) -> str:
    p1, pn = c.get("p_at_my_pick"), c.get("p_next_turn")
    if pn is None:
        return ""
    t2 = gaps["next_pick"]
    if gaps["on_the_clock"]:
        if pn < 0.05:
            return f"Almost surely gone by your pick at #{t2}: take him now if you want him."
        if pn < 0.35:
            return f"Unlikely to last to your pick at #{t2} ({round(pn * 100)}%): take him now."
        if pn > 0.75:
            return f"Likely still there at #{t2} ({round(pn * 100)}%): you could wait on him."
        return f"Toss-up to last to #{t2} ({round(pn * 100)}%)."
    if p1 is not None and p1 < 0.5:
        return f"Only {round(p1 * 100)}% likely to be there at your pick (#{gaps['my_pick']})."
    return f"{round((p1 or 0) * 100)}% likely to be there at your pick (#{gaps['my_pick']})."


def rerank(df: pd.DataFrame, rows: list[dict], mine: pd.DataFrame, gaps: dict, alpha: float,
           starter_value, runs: int = RUNS, n_cand: int = N_CAND) -> tuple[list[dict], dict]:
    """Re-order `rows` (each has player_id, vorp, team_gain) using the look-ahead and add timing fields.
    `starter_value(team_df)` returns the lineup value of a set of players (supplied by recommend.py).
    Returns (all rows, plan). Rows outside the top `n_cand` are appended unchanged, after the examined ones."""
    if not rows:
        return rows, {"status": "off", "reason": "No players left."}
    pool = df[df["available"]].reset_index(drop=True)
    if len(pool) < 2:
        return rows, {"status": "off", "reason": "Not enough players left."}
    rk = rank_key(pool)
    where = {pid: i for i, pid in enumerate(pool["player_id"])}
    pos = simulate_order(rk, runs)
    k1, k2 = gaps["k1"], gaps["k2"]

    def blend(r):  # one number in "points": early picks lean on VORP, later ones on team need
        return (1 - alpha) * r["vorp"] + alpha * r["team_gain"]

    cand = sorted(rows, key=blend, reverse=True)
    cand = [r for r in cand if r["player_id"] in where][:n_cand]
    ci = np.array([where[r["player_id"]] for r in cand])
    p_at = (pos[:, ci] >= k1).mean(0)                                    # still there at my pick
    keep = [i for i in range(len(cand)) if p_at[i] >= MIN_P_AT_PICK]
    if not keep:
        return rows, {"status": "off", "reason": "Every top candidate is almost certain to be gone by your pick."}
    cand, ci, p_at = [cand[i] for i in keep], ci[keep], p_at[keep]
    n = len(cand)
    cand_df = [df[df["player_id"] == r["player_id"]].iloc[[0]] for r in cand]
    # One table holds my roster, every first-pick candidate and (below) every follow-up candidate, so each hypothetical roster
    # is a cheap row selection instead of a new concatenation.
    n_mine = len(mine)
    base = starter_value(mine)
    g_now = np.array([blend(r) for r in cand])
    have_next = gaps["next_pick"] is not None

    value = g_now.copy()
    follow: list[str | None] = [None] * n
    follow_p: list[float] = [0.0] * n
    options: list[list] = [[] for _ in range(n)]
    wait_ok = (pos[:, ci] >= k1 + k2).mean(0)                           # would still be there at my pick after next
    if have_next:
        # The follow-up pool: players with a real chance (>= MIN_SURVIVE) of lasting until the NEXT pick, best for me first.
        # (The first-pick candidates are mostly top players who will be long gone by then.)
        surv2 = (pos >= k1 + k2).mean(0)
        fpool = [r for r in sorted(rows, key=blend, reverse=True) if r["player_id"] in where and surv2[where[r["player_id"]]] >= MIN_SURVIVE][:N_FOLLOW]
        m = len(fpool)
    else:
        m = 0
    if have_next and m > 0:
        fi = np.array([where[r["player_id"]] for r in fpool])
        fdf = [df[df["player_id"] == r["player_id"]].iloc[[0]] for r in fpool]
        everyone = pd.concat([mine] + cand_df + fdf)
        mine_rows = list(range(n_mine))
        c_row = [n_mine + a for a in range(n)]
        f_row = [n_mine + n + b for b in range(m)]
        # g_next[c, d] = what d adds on top of (mine + c), same blend
        g_next = np.full((n, m), -np.inf)
        for a in range(n):
            with_c = everyone.iloc[mine_rows + [c_row[a]]]
            v_c = starter_value(with_c)
            for b in range(m):
                if fpool[b]["player_id"] == cand[a]["player_id"]:
                    continue
                gain = starter_value(everyone.iloc[mine_rows + [c_row[a], f_row[b]]]) - v_c
                g_next[a, b] = (1 - alpha) * fpool[b]["vorp"] + alpha * gain
        floor = float(np.min(g_next[np.isfinite(g_next)])) if np.isfinite(g_next).any() else 0.0
        for a in range(n):
            ok_a = pos[:, ci[a]] >= k1                                  # runs where I really get c
            if not ok_a.any():
                continue
            pc = pos[ok_a][:, ci[a]][:, None]                           # c's place in each run where I get him
            pd_ = pos[ok_a][:, fi]                                      # (runs, m) places of every follow-up candidate d
            shifted = pd_ - (pc < pd_)                                  # my taking c moves everyone after him up one
            avail = shifted >= (k1 + k2)                                # d survives the other teams' k1+k2 picks
            scores = np.where(avail, g_next[a][None, :], -np.inf)
            best = scores.max(1)
            best = np.where(np.isfinite(best), best, floor)
            value[a] = g_now[a] + float(best.mean())
            hit = np.isfinite(scores.max(1))
            if hit.any():
                counts = np.bincount(scores.argmax(1)[hit], minlength=m)
                share = counts / float(ok_a.sum())                      # of ALL simulated drafts where I get c, how often is d my best follow-up
                order_f = np.argsort(-counts, kind="stable")
                options[a] = [{"name": fpool[j]["name"], "p": round(float(share[j]), 3)} for j in order_f[:3] if counts[j] > 0]
                if share[order_f[0]] >= MIN_FOLLOW_SHARE:
                    follow[a], follow_p[a] = fpool[order_f[0]]["name"], float(share[order_f[0]])

    # Not on the clock: a player who may be gone is worth less, so shade by the chance he's there.
    fallback = float(np.median(value))
    eff = p_at * value + (1 - p_at) * fallback
    order = np.argsort(-eff, kind="stable")
    examined = {cand[i]["player_id"] for i in range(n)}
    out = []
    for rank_i, i in enumerate(order):
        r = dict(cand[i])
        r["p_at_my_pick"] = round(float(p_at[i]), 3)
        r["p_next_turn"] = round(float(wait_ok[i]), 3) if have_next else None
        r["lookahead_value"] = round(float(value[i]), 3)
        r["likely_next"] = follow[i]
        r["likely_next_p"] = round(follow_p[i], 3) if follow[i] else None
        r["next_options"] = options[i]
        r["note"] = _wording(r, gaps) if have_next else (
            "" if gaps["on_the_clock"] else f"{round(float(p_at[i]) * 100)}% likely to be there at your pick (#{gaps['my_pick']}).")
        out.append(r)
    out += [r for r in rows if r["player_id"] not in examined]
    top = out[0]
    using_adp = "adp" in pool.columns and pool["adp"].notna().sum() >= max(10, 0.2 * len(pool))
    plan = {"status": "ok", "using_adp": bool(using_adp), "on_the_clock": gaps["on_the_clock"], "my_pick": gaps["my_pick"],
            "next_pick": gaps["next_pick"], "picks_before_my_turn": k1, "picks_before_next_turn": k2,
            "runs": runs, "take": top["name"], "then": top.get("likely_next"), "then_p": top.get("likely_next_p")}
    plan["headline"] = _headline(top, plan)
    return out, plan


def _headline(top: dict, plan: dict) -> str:
    if plan["on_the_clock"]:
        s = f"Take {top['name']}."
        if top.get("note"):
            s += " " + top["note"]
        s += _follow_text(top, plan, " ")
        return s
    s = f"Your pick is #{plan['my_pick']}, {plan['picks_before_my_turn']} pick(s) away. Best bet: {top['name']}"
    if top.get("p_at_my_pick") is not None:
        s += f" ({round(top['p_at_my_pick'] * 100)}% likely to still be there)"
    s += "."
    s += _follow_text(top, plan, " Then, ")
    return s


def _follow_text(top: dict, plan: dict, lead: str) -> str:
    """The follow-up sentence, with its odds. Honest when nobody stands out: 'it depends on who falls'."""
    if plan.get("next_pick") is None:
        return ""
    t, opts = plan["next_pick"], top.get("next_options") or []
    if top.get("likely_next"):
        pct = round((top.get("likely_next_p") or 0) * 100)
        return f"{lead}Likely follow-up at #{t}: {top['likely_next']} (best option in about {pct}% of simulated drafts)."
    if opts:
        o = opts[0]
        return f"{lead}No clear follow-up at #{t}: it depends on who falls (most often {o['name']}, {round(o['p'] * 100)}% of simulated drafts)."
    return ""
