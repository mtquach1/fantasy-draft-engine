"""Check this project folder before uploading it to GitHub.   Run from the project folder:   python check_before_github.py

It compares every file with the exact latest package (so a stale or edited file shows up), lists anything that would be uploaded
but isn't part of the package, and checks that your private and third-party files are being ignored. It changes nothing except
running `git init` if the folder isn't a Git repository yet (you need that anyway)."""
import hashlib
import subprocess
import sys
from pathlib import Path

MANIFEST = {
    ".dockerignore": "5222362399569e473c5181d1b03395c9f1a9695fa55dd918c165d6239df28388",
    ".env.example": "e11b91d574f371b9fb4bcf815ca0b2f4f3d645f36f4f5748fcb95fe9c6a3cc9d",
    ".github/workflows/tests.yml": "f1043942a1e782885aa823610e335fc79c5048cc09c77f314acad6ddc0ef9ba5",
    ".gitignore": "83eb1611689f0628fd17b21081c0beba86dc9b6f558980170b2be2031705fab2",
    "CHEATSHEET.md": "f180a26e04040486859e98be3d4307886a55d9ae9d5479f44e754a3571f34849",
    "Dockerfile": "95ce8f75468533ac6106757b310302587fb0c4a8e3f72f8ac25c988c96553e41",
    "LICENSE": "56763bd37562a281f1b14a1f96343db8735c1e24614aee79f2057ea9478877d4",
    "README.md": "79cef1aff6b5117abf6dd57eff4be04bed9e8bf01caa30ff657d8e6f3c984b23",
    "app/__init__.py": "6a3cf5192354f71615ac51034b3e97c20eda99643fcaf5bbe6d41ad59bd12167",
    "app/auction.py": "88ca56fd05d6e4423101fdd9b4a33754dc6b9cc955b0056aee7f5bb85f9263fb",
    "app/calibration.py": "943aac53f9e7f47d99372a0aa293b145de34b4e959343a15667eae6f175d7f27",
    "app/config.py": "536b61df6755cb830af4c7278ceaaff061acc0594464cf13e3af74c2a7f854c1",
    "app/connectors/__init__.py": "1c8320aae40dab6ca3d3cc33b5ea193682ddc5cef9fdb698eb3633111918cc98",
    "app/connectors/base.py": "7898a4266d2e4ddecb9cba7d85bcad97afc7c4964aa64e7dca018511089c80a7",
    "app/connectors/espn.py": "da3fe1c014b91a44a6d60e2b9074f25dd562296e4ac3a6fcb92f4fbb62d1d63a",
    "app/connectors/manual.py": "66fae33968db9f73f676de165a79b63761ee0866dffdd0c426a0f0dd9df338fd",
    "app/connectors/mock.py": "da7ef23c12c2fdc7ee8065ba5a9a75d4c09f2b34ebe391fe5b45ab23b850df3e",
    "app/connectors/yahoo.py": "ddd62b2b940932d251c98b66a4c5f62c5600db44d91412c2207b5a08ca0ae4ae",
    "app/draft.py": "6b51aa27463c740a7ccaf1cc5fcdf265a3998ee88305b03d222608ff6c33d511",
    "app/draft_export.py": "25d29dfa7d3d8dbfbdba5e1858d32faf5509dfb045942f4f679d2aab79f979d9",
    "app/ingestion.py": "8e62dac41cbcffe5327fe6b395656a0e35c5ea5a7c92189f4b577b5152608105",
    "app/live_blend.py": "abf24abe5f31c92b8b1aabad17d2aba329159163d2f2e1038568e59975422460",
    "app/lookahead.py": "9e5f5039d9a5e5ed8628b14603f6cf6045d643d7256920c0313106439c38ce48",
    "app/main.py": "894cb015cb56ee4f4521fb50f1478bcce220b9c4e0976257d6a9fd75b871676a",
    "app/nba_build.py": "5ea2137a9703b496cb6bbdc19ae7e95a512eeb6813d1e2026728101b6f352958",
    "app/observability.py": "a291a86ff040b63abd74d819959696deec68fc136650d0e4f4927209bd544e99",
    "app/optimizer.py": "c33e5cbaf5f359c7169a8ff1bf64511710c7a83b04ea92b8e2c788dc68b177c2",
    "app/own_model.py": "b5fa647e034e15e57a8f938b911a746fa51401603d761a337f324176c3dba272",
    "app/persist.py": "520f429793eeb325a1fe89d527ea30e367e3cdd7ac3380b845a704ea0cfa859b",
    "app/recommend.py": "a06334ad234c0535d33b3cfae26d05f3d934e7e3d100b5fb8ec78375d9002dd7",
    "app/scoring.py": "d3ac5bb2044429a5855c96b0925d2f377edfd589dfdb59ba022b2e2136210639",
    "app/season_ruler.py": "ab848787f08799b7f8da94c14ea1f9dfce1627fd3fc11e78cf0c46975acd2acb",
    "app/sports.py": "d8a00efca0aaeb0bc98a387cfe84f1471ea266d8514c9e419f54ad1454f4f5af",
    "app/state.py": "e66706b05fd5ef9ce910a1447efae4ce69dc97facd2f085f3168f46cb1994896",
    "app/static/index.html": "d8fbeaf2cd8f11354643c3cb5b6bc80a933f68a4f3fccdd37a021407d48b1fae",
    "app/stats.py": "90668a7ea87f94f82a2054c70d3c84d3ba2572c4799515e705392e39e6e86879",
    "app/sync.py": "681e8394c9804ba578c60597fcdf9f246dc6c4a19951dff2ceaa312e842a2795",
    "app/validate.py": "45596586c305a3feb8ade6162cf55353200c3cfcb63c865bcc8f67a99c53591a",
    "app/valuation.py": "27d53c9311bf1d9ff438082b0d5d0399dbd350046445fb3ea38f3af48d7e8f48",
    "app/views.py": "2e0bbda81d1cc657381967f952bc381082b7a89fabff0851695cd7df67145928",
    "build_nba_projections.py": "6e8e66b1a178bc3992127fbb9d49173c0a362ce6fd701cbdc97d7011bb30619b",
    "build_own_model.py": "4aba0e4df7bc40638d2d843b1252494955856f4d60943f9192882ca6e8366ecd",
    "build_sps_projections.py": "210bcd43a93b9f9bd724e0a149c69802a3a817202fef12124df884133d23471b",
    "data/.gitkeep": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "data/README.md": "4a241b4ce17700b48424dbd28c03107c635973bf277d2977f6a71d06acf00b08",
    "docker-compose.yml": "9e14b3c21b1964afb86b3654658d1cb8693f4a6d7ef3a8f504c56494090fa0c8",
    "docker-entrypoint.sh": "7feba3dc5fbde9768c02ba631944593680a972f7f2dc42f836620dd356728051",
    "docs/DESIGN.md": "57806476ac319771f762e5f83676936cdfba34067d42ad9b6d07a05bf90eeb86",
    "docs/PIPELINE.md": "e6a563dd7fbd1e0ba6b4ea77a4a9a5849c7a027c56f16aa65fdfea17bb0951f4",
    "espn_check.py": "b95a6fad2c337d301820c3cafa9e1f9016dac59529d8b5e5f4e66bc79eb69c35",
    "make_sample_data.py": "1faf8a4569a8717a28fbaa5cab7b11b0c56516ad4052eed9ed7acd12f242fcf9",
    "make_sample_sport.py": "f16c9df43ba593a2eabf80ae62b0b4def89ca793b4d9bcbea2ae605e63d176ad",
    "overrides_suggested.csv": "36c1f3e83a184c797b24aae262aa9c4bab829b8eee2f3e1c367c61b8ca4c4bb3",
    "pipeline/__init__.py": "82ad3420e0264bd02df7a6531dfb88183c9cf0574302d15bdabfdbcb08f87870",
    "pipeline/__main__.py": "290bb1699a8bdd3014d59341c58b3d350c0089b2eed2d5402fa041837df7c1c1",
    "pipeline/events.py": "cd627969c12ab865660c663162f31799c11950c527634ed2c6f33a2be756db5c",
    "pipeline/examples/injury_news.jsonl": "1a1d5cc11e1be6d8d4319f8e062fccfd52ce4ef581221dfc9d9a254b8626a337",
    "pipeline/ingest.py": "5ce69d03634c46bafb3b3f6eeeace29ee03470d06efea2bcbf3996d04eb0afbd",
    "pipeline/quality.py": "8a4eeb097c22c8f0a912c3971f2be2604be48376c283a010233bff62e43a9d10",
    "pipeline/replay.py": "692651064ed7595d4db12f46c8c8f641e31b1a418cfb83b653f6b2d864b2469f",
    "pipeline/sources.py": "6913cc268061261d21caaa9219c6247f8cf4f0f1d73fcd35fd6991acfafa3c6d",
    "pipeline/store.py": "0e66c5794788697ff0015c22f1ae845f692e96c6ac244dd8386191ef1f967172",
    "players_third_party.csv": "b1233714f73a5f78734509db5ce79ba8d8084be777609233b7d3de47da5b4d82",
    "render.yaml": "9e7e078061e1a52c739b7a671e4e5fc5bb7ea776153d0c3a431c352433a1d221",
    "requirements.txt": "d44a0b6827c1fa7e0f0e305278584848c46f09041c21c489e668e167f2678000",
    "research/calibrate_lookahead.py": "bac261dfa2f64f0af211941d7781085885b85a4f3061d0a89d843ef1a3855b09",
    "research/draft_backtest.py": "700157bacc66f98e47183d19136d48adbde57e9a0015143dc155fcf04e473cb4",
    "research/draft_simulation.py": "52d2d3140633a741656e528ba8407e8304aa82a64c5e9fafb63ae9f155da5e65",
    "research/draft_simulation_results.csv": "39760564e3816dcd8a4b11e24539d3adb9970ef0c6d5a4ff5dcc572e535c9c05",
    "research/evaluate_backtest.py": "f224a6f4b3c9459766f6385e5ca348a03d55b2001a7c795a29d7da9de9d393b2",
    "research/growth_check.py": "690d90aea7b80004e7cc52ec07383fbc449043a92fd9366b60175d74cae4da3b",
    "research/load_test.py": "d64dda47a6cb3298cb9f7f4278977c0b7662cc149cd460b8aae5f78f33818c2d",
    "research/mock_rank.py": "b549f21593b00772eaa0a68de86ed59bd797cf9ba78b42b07c6ad24a5f3d4e36",
    "research/why_report.py": "c2f32840ca7b512b5e6ed898fc07b8675ce46a3568049aea288e100568e16992",
    "start_draft.bat": "318c07fda99d5230985ccfa13c158d9ae37f8873eb7ddbe12c58c81920444bd5",
    "start_draft.sh": "017675c952b07f71eeeac41cffc101d0d36422b9cb516bc16a8a6ba23e3816d0",
    "tests/__init__.py": "6a3cf5192354f71615ac51034b3e97c20eda99643fcaf5bbe6d41ad59bd12167",
    "tests/fixtures/nba_api_leaguedash_sample.json": "b45c99097bdd20d92ab21c5fce6dff01fb2ee61d66123b1b44b36d11cc387d43",
    "tests/pipeline_helpers.py": "d932f4d71102e58213405528f03581cbde18e51930d5de14e89460dfc0617347",
    "tests/test_auction.py": "d755df507e973a0473a6959859e797be13bfd6e82044918efa1443e9992fdc43",
    "tests/test_calibration.py": "8ebb9407c23aae873c9669cfef29584f77c86f8ddda675a632c4c658875b32bc",
    "tests/test_draft_export.py": "99b6df272d0840f10204d8e7a28e08dc17ce2fd18333a58f627dd067548f2864",
    "tests/test_espn.py": "1dc379610975aaa553f99170c395154917979a964dfd172772b9f379215f841d",
    "tests/test_espn_default.py": "2c850683d150ce202757a89ad4f978cb5152b7b30ae046c94cee89d6230c23d8",
    "tests/test_live_blend.py": "e7a9f804844b18b0ae3c93d4a57aef7a1265d027b575cd179c2cd779d4293595",
    "tests/test_load_test.py": "c89e40c9332cc0e5f8918b9b9736bf821ebd4eeb2beb5a977d56ee37f19bdd17",
    "tests/test_lookahead.py": "581eca794ef5992f81b9044e3689337ab44394e0dba4718414d8c598b7c4f79d",
    "tests/test_math.py": "a90e49cc2ba783511e935b3dff39b2d64a0ef7021103b1bcfa7543340d7c2504",
    "tests/test_merged_pool.py": "6e1be3ad5c69a721b1b82a865d0061a8fe76964f97462b9e1ca266ae4ba312b6",
    "tests/test_mock_rank.py": "a2caf0bd3c834d760e3e1b6a48f974ceb1b88a244f9e1a975dae6fd251f1cfdb",
    "tests/test_nba_build.py": "10f813e7e4cc71d37a7e5645d3d2e7aad0fda5f25b42f35c91a53d1e51fa0f61",
    "tests/test_needs.py": "5a0b367c716b98daccd3e3640136ca797de1a1f980f210019212b234e8b58684",
    "tests/test_observability.py": "7b3818ef7b48a8e90896604da339103a8d898196a3b98c43bb493a143c0626ca",
    "tests/test_own_model.py": "74f286be6380eac794346e604bcc178781c28d4887978b3ec6f8e1ecc0de57b3",
    "tests/test_page_and_notice.py": "9f20cf1e8d3069b850714bfef38b0071bdf6867905fcbef6e1a5623f922f6a7b",
    "tests/test_persist.py": "1cc41da7f7d62cb21f87ddff0b1e387783a042856b5287ddba79b57add6625ec",
    "tests/test_pipeline_api.py": "c7608126fd70494fb6f20a97cea7f26ee69b6e66d31babcac00203f471721d65",
    "tests/test_pipeline_events.py": "4ec33361cccabc595e3444649ebaec000a32f911617ca6062f3ec6f7f44add88",
    "tests/test_pipeline_ingest.py": "d163c5b91ff23cf43d5ae7a92fa03d3b3f47d4e9cd9ff327557e51d8b23f7be9",
    "tests/test_pipeline_quality.py": "796c98ba8d92cf27d96bd40f55afea633ea2026c50c89f63b56cd2c732df1e22",
    "tests/test_pipeline_replay_cli.py": "d6e061eb366d6d5a10644a1a850e4ca2e6ebb2683fea663e9cd2a99555c450cc",
    "tests/test_pipeline_store.py": "5522bdd05377ce3a3b5edfd35a264b517e48bb8fa878beba86e3abf91c960eab",
    "tests/test_platforms.py": "cf09721bc7ad1f53c18f171859cb4ab2e84efefa02d9b85bc39a1c62b996d86b",
    "tests/test_race_safety.py": "cf6c78fcc02fe29d0e84bbd2343900e19b3f768a34acc90fec4905791b0dcc71",
    "tests/test_ranking_sort_and_speed_tools.py": "da84460b0f8367d87d1cdbc5d34fab156538c391773f07b33fed63b34258b9e5",
    "tests/test_research_reports.py": "d8f103b6ce53e4fe3c9d29c0cddb96c8eedb9315a261f5a00e2ffb8146ed162e",
    "tests/test_runtime_overrides.py": "8e15390b9c4ab1fea645324b70ce5bd3ee6b3495d9d23bcf960654e79dcba188",
    "tests/test_sessions.py": "0450c9b1423f61ca70034cc8e7774756e70aa8b1648308f8941ad1ab8db82f0d",
    "tests/test_sps_build.py": "9fd1d174789e305421351f03e32ffe602a09c0557f6347ea43e35d9b5a501bff",
    "tests/test_yahoo_check.py": "063b253644eb37cec9150ec178b3faf282da798fc256fd4749e67cbf15241573",
    "yahoo_check.py": "1ba2e13a7c688461f5d6668c608c0aeb8e65124f6f8c0657c979cd0bdd2cc8f0"
}
OPTIONAL = {"players_third_party.csv", "overrides_suggested.csv"}                      # you may have moved these into data/nba/
# files your own runs generate (backtest results, reports): they may exist, may differ, and are fine to publish (aggregate numbers only)
GENERATED = ("research/draft_backtest_results", "research/WHY_REPORT", "research/draft_simulation_results")
MAY_DIFFER = {"README.md", ".gitignore", ".env.example", "render.yaml", "CHEATSHEET.md"}  # files you are allowed to edit
RISKY = ("oauth2", "per_game_20", "per_game+20", "players_third_party", "overrides_suggested", "overrides.csv", "draft_state_",
         "players_own", "players_previous", "growth_check_players", "players.previous", "blend.json", "blend_nba", "overrides_nba", "data/nba", "data/oauth", ".env")
PRIVATE_SHOULD_BE_IGNORED = [".env", "data/oauth2.json", "data/nba/players.csv", "data/nba/overrides.csv", "data/nba/players_third_party.csv",
                             "data/nba/players_own.csv", "data/nba/raw/per_game_2024.csv", "data/nba/raw/per_game_2026.csv", "research/growth_check_players.csv"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()    # line endings don't count as a difference


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def main() -> int:
    root = Path.cwd()
    problems, notes = [], []
    print(f"Checking {root}\n")
    if not (root / "app" / "main.py").exists():
        print("This doesn't look like the project folder (no app/main.py). Open PowerShell in the folder that contains 'app' and 'tests'.")
        return 2

    # 1. every package file is present and identical
    missing, different, ok = [], [], 0
    for rel, digest in MANIFEST.items():
        p = root / rel
        if not p.exists():
            if rel not in OPTIONAL:
                missing.append(rel)
            continue
        if sha(p) == digest:
            ok += 1
        elif rel in MAY_DIFFER:
            notes.append(f"{rel} differs from the package (fine, you may edit it)")
        else:
            different.append(rel)
    print(f"[1] Package files: {ok} identical, {len(missing)} missing, {len(different)} different")
    for r in missing:
        print(f"      MISSING   {r}")
    for r in different:
        print(f"      DIFFERENT {r}   (an older or edited copy: re-extract the latest zip over the folder)")
    if missing:
        problems.append("some package files are missing")
    if different:
        problems.append("some files are not the latest version")

    # 2. what would be uploaded
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        print("\n[2] Git isn't installed (or not on PATH). Install Git for Windows, reopen PowerShell, and run this again.")
        problems.append("git not available")
    else:
        if not (root / ".git").exists():
            git("init")
            notes.append("ran 'git init' for you")
        listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
        upload = sorted({f.replace("\\", "/") for f in listed if f})
        extra = [f for f in upload if f not in MANIFEST and not f.startswith(GENERATED) and f != "check_before_github.py"]
        risky = [f for f in upload if any(k in f for k in RISKY) and f != ".env.example"]
        print(f"\n[2] Files that 'git add .' would upload: {len(upload)}   (the package has {len(MANIFEST) - len(OPTIONAL)} uploadable files)")
        if extra:
            print("      Not part of the package (delete them, or add them to .gitignore, unless you meant to publish them):")
            for f in extra:
                print(f"      EXTRA     {f}")
            problems.append("extra files would be uploaded")
        if risky:
            print("      PRIVATE OR THIRD-PARTY DATA WOULD BE PUBLISHED:")
            for f in risky:
                print(f"      RISKY     {f}")
            problems.append("private or third-party data would be uploaded")
        print("\n[3] Private and third-party files that exist here and must stay out of GitHub:")
        shown = 0
        for rel in PRIVATE_SHOULD_BE_IGNORED + sorted(OPTIONAL):
            if (root / rel).exists():
                ignored = rel not in upload
                shown += 1
                print(f"      {'ignored (good)  ' if ignored else 'NOT IGNORED    '} {rel}")
                if not ignored and not any(rel in r for r in risky):
                    problems.append(f"{rel} is not ignored")
        if not shown:
            print("      (none present)")

    # 4. verdict
    print()
    for n in notes:
        print(f"note: {n}")
    if problems:
        print("\nNOT READY. Fix these first:")
        for pr in dict.fromkeys(problems):
            print(f"  - {pr}")
        return 1
    print("\nREADY TO UPLOAD. Next: git add .   then   git commit -m \"Fantasy draft engine\"   then follow the GitHub steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
