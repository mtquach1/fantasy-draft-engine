"""Build data/nba/players.csv from YOUR OWN projection model (no third-party projections).

Needs per-game tables from Basketball-Reference in data/nba/raw/, named per_game_2024.csv, per_game_2025.csv, per_game_2026.csv
(the year is the season's END year; three or more seasons). Run:  python build_own_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app import own_model as om                      # noqa: E402
from app.ingestion import data_dir                   # noqa: E402


def main(raw_dir=None, out_dir=None, top_n=300, blend=0.0, reference=None):
    base = Path(data_dir()) / "nba"
    raw_dir, out_dir = Path(raw_dir or base / "raw"), Path(out_dir or base)
    files = om.find_seasons(raw_dir)
    if len(files) < 3:
        print(f"Need at least 3 per-game seasons in {raw_dir} (found {sorted(files)}). Name them per_game_2024.csv, per_game_2025.csv, ...")
        return 1
    seasons = {y: om.load_season(p) for y, p in files.items()}
    years = sorted(seasons)
    p = om.Params()
    beta, n = om.fit_age_from_transition(seasons, years[:2], p)
    proj = om.project_next(seasons, years, p, beta)
    ov_path = Path(raw_dir).parent / "overrides.csv"
    if ov_path.exists():
        proj, report = om.apply_overrides(proj, om.load_overrides(ov_path))
        print(f"Applied overrides from {ov_path.name}: " + f"{sum(r.startswith('UPDATED') for r in report)} updated, "
              f"{sum(r.startswith('ADDED') for r in report)} added, {sum(r.startswith('SKIPPED') for r in report)} skipped.")
        for line in report:
            if line.startswith("SKIPPED"):
                print("  " + line)
    proj_own = proj                                        # the pure own model (plus your overrides): what the live blend slider starts from
    if blend:
        ref_path = Path(reference or Path(raw_dir).parent / "players_third_party.csv")
        if not ref_path.exists():
            print(f"--blend needs a reference projection file. Not found: {ref_path}")
            return 1
        import pandas as pd
        proj, n_b = om.blend_with_reference(proj_own, pd.read_csv(ref_path), blend)
        print(f"Blended {n_b} players {blend:.0%} toward {ref_path.name} (own model {1 - blend:.0%}).")
    keep = proj.head(top_n).index                          # one player set for both files, so the slider can mix them
    app = om.to_app_frame(proj.loc[keep], len(keep))
    own_app = om.to_app_frame(proj_own.loc[keep], len(keep))
    out_dir.mkdir(parents=True, exist_ok=True)
    app.to_csv(out_dir / "players.csv", index=False)
    own_app.to_csv(out_dir / "players_own.csv", index=False)           # the live slider reads this and the reference file
    import json
    (out_dir / "blend.json").write_text(json.dumps({"weight": float(blend)}), encoding="utf-8")
    print(f"Seasons used: {years}. Age curve fitted on the {years[0]}->{years[1]} transition ({n} players).")
    print(f"Wrote {len(app)} players to {out_dir / 'players.csv'}. Top 8 by projected ESPN-default fantasy points:")
    for r in proj.head(8).itertuples():
        print(f"  {r.Name:26s} {r.GP:5.1f} games  {r.MPG:4.1f} mpg  {r.fpts:7.0f} pts")
    if not ov_path.exists():
        print("Not in this model: rookies, trades, role changes, current injuries. Add them in data/nba/overrides.csv (see README).")
    print("Next: check /health/data, then restart the server.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build data/nba/players.csv from your own projection model.")
    ap.add_argument("--blend", type=float, default=0.0, help="0 to 1: how far to pull toward a reference projection file (default 0 = pure own model)")
    ap.add_argument("--reference", default=None, help="the reference players.csv (default data/nba/players_third_party.csv)")
    a = ap.parse_args()
    if not 0.0 <= a.blend <= 1.0:
        sys.exit("--blend must be between 0 and 1")
    sys.exit(main(blend=a.blend, reference=a.reference))
