#!/usr/bin/env python3
"""
Gadfly: Independent Challenger for Coeus

Runs as a separate daemon process. Monitors Coeus's cycle logs and
writes challenges using adaptive techniques.

Usage:
    python gadfly_main.py                  # Normal operation
    python gadfly_main.py --status         # Show current Gadfly status
    python gadfly_main.py --once           # Run single cycle and exit
"""

import argparse
import atexit
import importlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_file) as f:
        return yaml.safe_load(f)


def check_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)


def show_status(base_path: str = "."):
    """Show current Gadfly status."""
    base = Path(base_path)

    print(f"\n{'='*60}")
    print("GADFLY STATUS")
    print(f"{'='*60}")

    state_file = base / "state" / "gadfly_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        print(f"\nGadfly cycle: {state.get('cycle_number', 0)}")
        print(f"Last Coeus cycle processed: {state.get('last_coeus_cycle', 0)}")
        print(f"Current technique: {state.get('current_technique', '?')}")
        print(f"Technique streak: {state.get('technique_streak', 0)}")
        print(f"Stated preferences tracked: {len(state.get('stated_preferences', []))}")
        print(f"Observed behaviors tracked: {len(state.get('observed_behaviors', []))}")

        history = state.get("technique_history", [])
        if history:
            print(f"\nTechnique history ({len(history)} switches):")
            for h in history[-5:]:
                print(f"  {h.get('technique', '?')} ({h.get('cycles_used', '?')} cycles) → switched at gadfly cycle {h.get('switched_at_gadfly_cycle', '?')}")
    else:
        print("\nNo previous state found. Gadfly has not run yet.")

    # Check channel file
    channel = base / "human_interaction" / "gadfly_challenges.md"
    if channel.exists():
        content = channel.read_text()
        challenge_count = content.count("## Challenge [")
        print(f"\nTotal challenges written: {challenge_count}")

    print(f"\n{'='*60}\n")


def _write_pid_file(base_path: str) -> Path:
    pid_file = Path(base_path) / "gadfly.pid"
    pid_file.write_text(str(os.getpid()))
    atexit.register(lambda: pid_file.unlink(missing_ok=True))
    return pid_file


def _reload_modules():
    """Reload Gadfly-relevant modules."""
    module_names = ["llm", "resources", "gadfly"]
    reloaded = []
    for name in module_names:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
            reloaded.append(name)
    return reloaded


def _create_gadfly(config: dict, base_path: str):
    from gadfly import GadflyAgent
    return GadflyAgent(config, base_path)


def run_gadfly(config_path: str, config: dict, base_path: str = ".", once: bool = False):
    """Run the Gadfly loop."""
    gadfly = _create_gadfly(config, base_path)

    shutdown_requested = False
    reload_requested = False

    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        print("\nGadfly shutdown requested...")
        shutdown_requested = True

    def reload_handler(signum, frame):
        nonlocal reload_requested
        print("\nGadfly reload requested (SIGHUP)...")
        reload_requested = True

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGHUP, reload_handler)

    pid_file = _write_pid_file(base_path)
    poll_interval = gadfly.gadfly_config.get("min_interval_seconds", 300)

    print(f"\n{'='*60}")
    print("GADFLY STARTING")
    print(f"{'='*60}")
    print(f"Gadfly cycle: {gadfly.cycle_number}")
    print(f"Current technique: {gadfly.state.get('current_technique', '?')}")
    print(f"Polling every {poll_interval}s for new Coeus cycles")
    print(f"PID file: {pid_file}")
    print(f"Reload with: kill -HUP $(cat {pid_file})")
    print(f"{'='*60}\n")

    try:
        while not shutdown_requested:
            if reload_requested:
                reload_requested = False
                print(f"\n[{datetime.now(timezone.utc).isoformat()}] Reloading Gadfly...")
                gadfly.shutdown()
                config = load_config(config_path)
                reloaded = _reload_modules()
                gadfly = _create_gadfly(config, base_path)
                print(f"  Reloaded modules: {', '.join(reloaded)}")

            if gadfly.should_run():
                print(f"\n[{datetime.now(timezone.utc).isoformat()}] Running Gadfly cycle {gadfly.cycle_number + 1}")
                result = gadfly.run_cycle()
                print(f"  Status: {result.get('status')}")
                print(f"  Technique: {result.get('technique', '?')}")
                print(f"  Coeus cycle: {result.get('coeus_cycle', '?')}")
                if result.get("input_tokens"):
                    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")

                if once:
                    print("\nSingle cycle completed. Exiting.")
                    break
            elif once:
                # In --once mode, run even if not enough cycles have passed
                print(f"\n[{datetime.now(timezone.utc).isoformat()}] Running Gadfly cycle (forced --once)")
                result = gadfly.run_cycle()
                print(f"  Status: {result.get('status')}")
                print(f"  Technique: {result.get('technique', '?')}")
                break

            if not once:
                # Poll for new Coeus cycles
                sleep_until = time.time() + poll_interval
                while time.time() < sleep_until and not shutdown_requested and not reload_requested:
                    time.sleep(min(10, sleep_until - time.time()))

    finally:
        gadfly.shutdown()
        print("\nGadfly shut down cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Gadfly: Independent Challenger for Coeus"
    )
    parser.add_argument("--config", "-c", default="config/settings.yaml",
                        help="Path to configuration file")
    parser.add_argument("--status", "-s", action="store_true",
                        help="Show current Gadfly status")
    parser.add_argument("--once", "-1", action="store_true",
                        help="Run a single cycle and exit")
    parser.add_argument("--base-path", default=".",
                        help="Base path for agent files")

    args = parser.parse_args()

    if args.status:
        show_status(args.base_path)
        return

    check_api_key()
    config = load_config(args.config)

    if not config.get("gadfly", {}).get("enabled", True):
        print("Gadfly is disabled in configuration.")
        return

    run_gadfly(args.config, config, args.base_path, once=args.once)


if __name__ == "__main__":
    main()
