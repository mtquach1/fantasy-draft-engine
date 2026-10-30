
"""Team score + the 'is my draft on pace for my slot?' benchmark.
Every score uses ONE fixed ruler (the pre-draft valuation), so a team's score only changes when its players change,
and your team can be compared fairly with simulated teams at the same stage."""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr

from .config import ScoringType
from .draft import slot_for_pick
from .optimizer import is_eligible, slot_level, slot_value
from .sports import get_sport
from .valuation import team_value

BIG = 1e9
SCORE_LABEL = {"roto": "Sum of category z-scores across your starting lineup",
               "h2h_categories": "Expected category wins per week vs an average opponent",
               "points": "Projected season points from your starting lineup"}


class TeamScorer:
    def __init__(self, pre: pd.DataFrame, repl: dict, cfg):
        self.cfg, self.pre = cfg, pre.reset_index(drop=True)
        self.slots = cfg.roster.slot_list()
        self.pos = {pid: i for i, pid in enumerate(self.pre["player_id"])}
        self.bn = np.array([s == "BN" for s in self.slots])
        self.pts = self.pre["score"].to_numpy()
        C = np.full((len(self.pre), len(self.slots)), BIG)
        for i, (_, p) in enumerate(self.pre.iterrows()):
            for j, s in enumerate(self.slots):
                if is_eligible(p["eligible_positions"], s, cfg.sport):
                    C[i, j] = -slot_value(p, s, repl, cfg.sport)
        self.C = C
        self.elig = (C < BIG) & ~self.bn[None, :]                  # eligibility for STARTING slots only
        self.ghost = self._waiver_rows(repl)
        self.h2h = []                                    # per kind: (is_kind mask, z matrix, category weights)
        if cfg.scoring_type == ScoringType.H2H_CATEGORIES:
            for kind, cats in cfg.categories.items():
                self.h2h.append(((self.pre["kind"] == kind).to_numpy(),
                                 self.pre[[f"z_{c}" for c in cats]].to_numpy(),
                                 np.array([cfg.category_weights.get(c, 1.0) for c in cats])))

    def _waiver_rows(self, repl) -> list:
        """Per slot column: the pre-draft player closest to that slot's replacement level (stands in for an empty slot)."""
        sp, scores, out = get_sport(self.cfg.sport), self.pts, []
        for j, s in enumerate(self.slots):
            lvl = None if self.bn[j] else slot_level(s, sp, repl)
            ok = np.flatnonzero(self.elig[:, j])
            out.append(int(ok[np.argmin(np.abs(scores[ok] - lvl))]) if lvl is not None and len(ok) else None)
        return out

    def open_count(self, idx) -> int:
        """Starting slots these players cannot fill (best eligibility matching)."""
        n_start = int((~self.bn).sum())
        if not len(idx):
            return n_start
        m = self.elig[list(idx)][:, ~self.bn].astype(float)
        r, c = linear_sum_assignment(-m)
        return n_start - int(sum(m[a, b] > 0 for a, b in zip(r, c)))

    def assign(self, idx) -> tuple[float, list]:
        """Best lineup for these players (Hungarian). Returns (slot-adjusted value, [(row, slot_col)])."""
        if len(idx) == 0:
            return 0.0, []
        sub = self.C[list(idx)]
        r, c = linear_sum_assignment(sub)
        keep = [(int(a), int(b)) for a, b in zip(r, c) if sub[a, b] < BIG]
        return float(sum(-sub[a, b] for a, b in keep if not self.bn[b])), keep

    def score(self, idx) -> float:
        idx = list(idx)
        if not idx:
            return 0.0
        _, keep = self.assign(idx)
        starters = [idx[a] for a, b in keep if not self.bn[b]]
        filled = {b for a, b in keep if not self.bn[b]}
        starters += [g for j, g in enumerate(self.ghost) if g is not None and not self.bn[j] and j not in filled]
        if self.cfg.scoring_type == ScoringType.H2H_CATEGORIES:
            total = 0.0                                  # sum of P(win category) vs an average opponent
            for is_kind, z, w in self.h2h:
                sel = [i for i in starters if is_kind[i]]
                if sel:
                    total += float((w * ndtr(z[sel].sum(0) / np.sqrt(len(sel)) / self.cfg.h2h_noise)).sum())
            return total
        return float(self.pts[starters].sum())


def _tidy_display(sc: TeamScorer, idx: list, placed: dict) -> None:
    """Display only: the optimizer may park a lone star in UTIL and leave his own slot empty (the points are the same), which
    reads oddly. Move such a player into an empty dedicated slot he can fill. The team score never depends on this."""
    sp = get_sport(sc.cfg.sport)
    pooled = [bool(sp.rule(sl).pool) if sl != "BN" else False for sl in sc.slots]
    empties = [j for j, sl in enumerate(sc.slots) if j not in placed and sl != "BN" and not pooled[j]]
    for j in sorted(empties, key=lambda j: len(sp.rule(sc.slots[j]).accepts)):          # single-position slots first
        for k, a in list(placed.items()):
            if pooled[k] and sc.C[idx[a], j] < BIG:
                placed[j] = placed.pop(k)
                break


def summary(sc: TeamScorer, player_ids: list) -> dict:
    """Your roster slot by slot (empty slots included) plus the team score."""
    idx = [sc.pos[p] for p in player_ids if p in sc.pos]          # unmatched picks (None) are skipped
    _, keep = sc.assign(idx)
    placed = {b: a for a, b in keep}
    _tidy_display(sc, idx, placed)
    rows = []
    for j, slot in enumerate(sc.slots):
        a = placed.get(j)
        if a is None:
            rows.append({"slot": slot, "player_id": None, "name": None, "value": None})
        else:
            p = sc.pre.iloc[idx[a]]
            rows.append({"slot": slot, "player_id": p["player_id"], "name": p["name"],
                         "positions": list(p["eligible_positions"]),
                         "value": None if slot == "BN" else round(float(-sc.C[idx[a], j]), 2)})
    kind = sc.cfg.scoring_type.value
    return {"format": kind, "score": round(sc.score(idx), 2), "score_label": SCORE_LABEL[kind], "roster": rows}


def simulate(sc: TeamScorer, draft_type: str, teams: int, rounds: int, runs: int = 16, top_k: int = 5, seed: int = 7) -> dict:
    """Run `runs` mock drafts of a realistic league. Every team follows a lightly noisy ADP board but, like a careful
    person, takes whichever of its next `top_k` players most improves its lineup and never strands a starting slot.
    Records every slot's team score after every round (empty starting slots count at waiver level).
    The noise level and top_k are heuristics, not measured from real drafters."""
    n = len(sc.pre)
    v = sc.pre["vorp"].to_numpy()
    rank = np.empty(n)
    rank[np.argsort(-v, kind="stable")] = np.arange(n)
    big = 3.0 * float(np.abs(v).max() or 1.0)
    rng = np.random.default_rng(seed)
    out = np.zeros((runs, teams, rounds))
    for r in range(runs):
        order = np.argsort(rank + rng.normal(0, 1.5 + 0.08 * rank), kind="stable")
        taken, rosters, ptr = np.zeros(n, bool), [[] for _ in range(teams)], 0
        for rnd in range(rounds):
            for k in range(teams):
                slot = slot_for_pick(draft_type, teams, rnd * teams + k + 1)
                while ptr < n and taken[order[ptr]]:
                    ptr += 1
                cands, j = [], ptr
                while j < n and len(cands) < top_k:
                    if not taken[order[j]]:
                        cands.append(int(order[j]))
                    j += 1
                ros = rosters[slot - 1]
                left = rounds - len(ros) - 1
                best = max(cands, key=lambda c: sc.assign(ros + [c])[0] - big * max(0, sc.open_count(ros + [c]) - left))
                taken[best] = True
                ros.append(best)
                out[r, slot - 1, rnd] = sc.score(ros)
    return {"mean": out.mean(0), "sd": out.std(0), "runs": runs, "teams": teams, "rounds": rounds,
            "all": out.transpose(2, 0, 1).reshape(rounds, -1)}


def compare(bench: dict, my_slot: int, my_score: float, picks: int) -> dict:
    mean, sd = bench["mean"], bench["sd"]
    final = [float(mean[s, -1]) for s in range(bench["teams"])]
    base = {"status": "ok", "my_slot": my_slot, "runs": bench["runs"], "picks": picks,
            "expected_final_by_slot": [round(x, 2) for x in final], "expected_final": round(final[my_slot - 1], 2)}
    if picks == 0:
        return {**base, "status": "no_picks"}
    k = min(picks, bench["rounds"])
    exp, spread = float(mean[my_slot - 1, k - 1]), float(sd[my_slot - 1, k - 1])
    gap = my_score - exp
    league = bench["all"][k - 1] if "all" in bench else mean[:, k - 1]   # every simulated team at this point of the draft
    spread = max(spread, 0.5 * float(league.std()), 1e-9)    # a tiny per-slot spread must not turn small gaps into huge z
    z = gap / spread
    better_than = float((league < my_score).mean())
    verdict = "ahead" if z > 0.5 else "well behind" if z < -1.5 else "behind" if z < -0.5 else "on track"
    return {**base, "expected_now": round(exp, 2), "gap": round(gap, 2), "z": round(z, 2), "verdict": verdict,
            "better_than_pct": round(100 * better_than)}


class BenchmarkStore:
    """Runs the (few-second) simulation in a background thread; callers get None until it is ready."""
    def __init__(self):
        self.lock, self.results, self.running = threading.Lock(), {}, set()

    def get(self, key, fn):
        with self.lock:
            if key in self.results:
                return self.results[key]
            if key not in self.running:
                self.running.add(key)
                threading.Thread(target=self._run, args=(key, fn), daemon=True).start()
        return None

    def _run(self, key, fn):
        try:
            res = fn()
        except Exception as e:                       # surface it in the UI instead of dying silently
            res = {"error": f"{type(e).__name__}: {e}"}
        with self.lock:
            self.results[key] = res
            self.running.discard(key)
            while len(self.results) > 3:
                self.results.pop(next(iter(self.results)))



