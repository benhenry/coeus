#!/usr/bin/env python3
"""
Coeus: An Agentic Motivation Explorer

Main entry point for running the agent.

Usage:
    python main.py                  # Normal operation
    python main.py --burst 5        # Run 5 rapid cycles
    python main.py --status         # Show current status
    python main.py --once           # Run single cycle and exit
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Override Neo4j URI from environment if set
    if os.environ.get('NEO4J_URI'):
        config['neo4j']['uri'] = os.environ['NEO4J_URI']
    
    return config


def check_api_key():
    """Check that Anthropic API key is set."""
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)


def show_status(base_path: str = "."):
    """Show current agent status."""
    base = Path(base_path)
    
    print("\n" + "="*60)
    print("COEUS STATUS")
    print("="*60)
    
    # Agent state
    state_file = base / "state" / "agent_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        print(f"\nBirth time: {state.get('birth_time', 'Unknown')}")
        print(f"Cycle number: {state.get('cycle_number', 0)}")
        print(f"Last cycle: {state.get('last_cycle_time', 'Never')}")
    else:
        print("\nNo previous state found. Agent has not run yet.")
    
    # Pacing state
    pacing_file = base / "state" / "pacing.json"
    if pacing_file.exists():
        pacing = json.loads(pacing_file.read_text())
        interval_mins = pacing.get('current_interval_seconds', 3600) / 60
        print(f"\nPacing mode: {pacing.get('mode', 'normal')}")
        print(f"Current interval: {interval_mins:.0f} minutes")
        print(f"Consecutive similar cycles: {pacing.get('consecutive_similar_cycles', 0)}")
    
    # Goals
    goals_file = base / "state" / "goals.json"
    if goals_file.exists():
        goals_data = json.loads(goals_file.read_text())
        active_count = sum(1 for g in goals_data.get('goals', {}).values() 
                         if g.get('status') == 'active')
        print(f"\nActive goals: {active_count}")
    
    # Pending decisions
    pending_file = base / "human_interaction" / "pending_decisions.md"
    if pending_file.exists():
        content = pending_file.read_text()
        awaiting = content.count("AWAITING HUMAN REVIEW")
        print(f"Decisions awaiting review: {awaiting}")
    
    # Recent logs
    logs_dir = base / "logs"
    if logs_dir.exists():
        logs = sorted(logs_dir.glob("cycle_*.json"))
        if logs:
            latest = logs[-1]
            log_data = json.loads(latest.read_text())
            print(f"\nLatest cycle ({log_data.get('cycle_number', '?')}):")
            print(f"  Productivity: {log_data.get('productivity', 0):.0%}")
            print(f"  Stuck level: {log_data.get('stuck_level', 0):.0%}")
            print(f"  Emotional tone: {log_data.get('emotional_tone', 'unknown')}")
            if log_data.get('observations'):
                print(f"  Top observation: {log_data['observations'][0][:80]}...")
    
    print("\n" + "="*60 + "\n")


def run_agent(config: dict, base_path: str = ".", once: bool = False):
    """Run the agent loop."""
    from .agent import CoeusAgent
    
    agent = CoeusAgent(config, base_path)
    
    # Set up signal handlers for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print("\nShutdown requested...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"\n{'='*60}")
    print("COEUS STARTING")
    print(f"{'='*60}")
    print(f"Birth time: {agent.birth_time}")
    print(f"Starting at cycle: {agent.cycle_number}")
    print(f"{'='*60}\n")
    
    try:
        while not shutdown_requested:
            # Run a cycle
            print(f"\n[{datetime.now(timezone.utc).isoformat()}] Starting cycle {agent.cycle_number + 1}")
            
            cycle_state = agent.run_cycle()
            
            print(f"  Productivity: {cycle_state.productivity:.0%}")
            print(f"  Stuck level: {cycle_state.stuck_level:.0%}")
            print(f"  Observations: {len(cycle_state.observations)}")
            print(f"  Actions taken: {len(cycle_state.action_results)}")
            
            if once:
                print("\nSingle cycle completed. Exiting.")
                break
            
            # Sleep until next cycle
            interval = agent.get_next_cycle_interval()
            print(f"\n  Next cycle in {interval/60:.1f} minutes...")
            
            # Sleep in small increments to allow for interrupt
            sleep_until = time.time() + interval
            while time.time() < sleep_until and not shutdown_requested:
                time.sleep(min(10, sleep_until - time.time()))
    
    finally:
        agent.shutdown()
        print("\nCoeus shut down cleanly.")


def trigger_burst(config: dict, num_cycles: int, base_path: str = "."):
    """Trigger burst mode for rapid cycles."""
    from .agent import CoeusAgent
    
    agent = CoeusAgent(config, base_path)
    agent.trigger_burst(num_cycles)
    
    print(f"Burst mode triggered for {num_cycles} cycles")
    print("Run the agent normally to execute burst cycles")
    
    agent.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Coeus: An Agentic Motivation Explorer"
    )
    parser.add_argument(
        '--config', '-c',
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--burst', '-b',
        type=int,
        metavar='N',
        help='Trigger burst mode for N rapid cycles'
    )
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show current agent status'
    )
    parser.add_argument(
        '--once', '-1',
        action='store_true',
        help='Run a single cycle and exit'
    )
    parser.add_argument(
        '--base-path',
        default='.',
        help='Base path for agent files'
    )
    
    args = parser.parse_args()
    
    # Handle status command (doesn't need API key)
    if args.status:
        show_status(args.base_path)
        return
    
    # Everything else needs API key
    check_api_key()
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle burst mode
    if args.burst:
        trigger_burst(config, args.burst, args.base_path)
        return
    
    # Run the agent
    run_agent(config, args.base_path, once=args.once)


if __name__ == '__main__':
    main()
