"""Mission data startup utilities.

Provides mission status checking, interactive refresh menu, and
CLI flag resolution — shared by main.py and gradio_app.py.
"""

import json
from datetime import datetime
from pathlib import Path


def get_mission_status() -> dict:
    """Scan mission JSONs and return a status summary.

    Returns:
        Dict with keys: mission_count, mission_names, total_datasets, oldest_date.
    """
    missions_dir = Path(__file__).parent / "missions"
    mission_files = sorted(missions_dir.glob("*.json"))

    mission_names = [f.stem for f in mission_files]
    total_datasets = 0
    oldest_date = None

    for f in mission_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for inst in data.get("instruments", {}).values():
                total_datasets += len(inst.get("datasets", {}))
            gen_at = data.get("_meta", {}).get("generated_at", "")
            if gen_at:
                try:
                    dt = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
                    if oldest_date is None or dt < oldest_date:
                        oldest_date = dt
                except ValueError:
                    pass
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "mission_count": len(mission_files),
        "mission_names": mission_names,
        "total_datasets": total_datasets,
        "oldest_date": oldest_date,
    }


def show_mission_menu() -> str:
    """Print mission data status and show an interactive menu.

    Returns:
        Action string: "continue", "refresh", or "all".
    """
    status = get_mission_status()

    print("-" * 60)
    print("  Mission Data Status")
    print("-" * 60)

    if status["mission_count"] == 0:
        print("  No mission data found. Will download on first use.")
        print()
        return "continue"

    names = status["mission_names"]
    print(f"  Missions loaded: {status['mission_count']} ({', '.join(names)})")
    print(f"  Total datasets:  {status['total_datasets']}")
    if status["oldest_date"]:
        age_str = status["oldest_date"].strftime("%Y-%m-%d %H:%M UTC")
        print(f"  Last refreshed:  {age_str}")
    print()
    print("  [Enter] Continue with current data")
    print("  [r]     Refresh primary missions (re-download)")
    print("  [a]     Download ALL missions (~50 from CDAWeb)")
    print()

    try:
        choice = input("  Choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "continue"

    if choice in ("r", "refresh"):
        return "refresh"
    elif choice in ("a", "all"):
        return "all"
    return "continue"


def run_mission_refresh(action: str):
    """Execute mission data refresh based on chosen action.

    Args:
        action: "refresh" for primary missions, "all" for every CDAWeb mission.
    """
    from knowledge.bootstrap import populate_missions, clean_all_missions
    from knowledge.mission_prefixes import PRIMARY_MISSIONS
    import knowledge.bootstrap as bootstrap_mod

    if action == "refresh":
        print("\nRefreshing primary missions...")
        stems = set(PRIMARY_MISSIONS)
        clean_all_missions(only_stems=stems)
        bootstrap_mod._bootstrap_checked = False
        populate_missions(only_stems=stems)
    elif action == "all":
        print("\nDownloading ALL missions from CDAWeb...")
        clean_all_missions()
        bootstrap_mod._bootstrap_checked = False
        populate_missions()

    from knowledge.mission_loader import clear_cache
    clear_cache()
    print()


def resolve_refresh_flags(refresh: bool = False, refresh_all: bool = False):
    """Map CLI --refresh / --refresh-all flags to an action, or show interactive menu.

    Args:
        refresh: True if --refresh was passed.
        refresh_all: True if --refresh-all was passed.
    """
    if refresh:
        run_mission_refresh("refresh")
    elif refresh_all:
        run_mission_refresh("all")
    else:
        action = show_mission_menu()
        if action != "continue":
            run_mission_refresh(action)
