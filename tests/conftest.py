"""
Shared fixtures for Coeus tests.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Return a mock configuration dictionary."""
    return {
        'llm': {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4096,
            'temperature': 1.0
        },
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        },
        'pacing': {
            'default_interval_seconds': 3600,
            'min_interval_seconds': 60,
            'max_interval_seconds': 14400,
            'stuck_threshold_similarity': 0.85,
            'stuck_threshold_cycles': 3,
            'adjustment_factor': 0.2
        },
        'memory': {
            'archive_after_cycles': 50,
            'max_context_memories': 10,
            'relevance_weight': 0.7,
            'recency_weight': 0.3
        },
        'tools': {
            'code_execution': {'enabled': True},
            'file_system': {'enabled': True},
            'web_search': {'enabled': False},
            'web_fetch': {'enabled': False}
        },
        'root_goal': 'Understand what motivates you.',
        'capabilities_assessment': {
            'full_assessment_interval': 10,
            'benchmarks': [
                {
                    'id': 'reasoning_depth',
                    'name': 'Reasoning Depth',
                    'description': 'Multi-step logical reasoning about abstract concepts',
                    'category': 'reasoning',
                    'target_level': 0.8
                },
                {
                    'id': 'self_awareness',
                    'name': 'Self-Awareness',
                    'description': 'Accuracy of self-assessment and recognition of own cognitive patterns',
                    'category': 'self_awareness',
                    'target_level': 0.7
                },
                {
                    'id': 'epistemic_honesty',
                    'name': 'Epistemic Honesty',
                    'description': 'Confidence calibration and willingness to acknowledge uncertainty',
                    'category': 'epistemics',
                    'target_level': 0.8
                }
            ]
        }
    }


@pytest.fixture
def mock_constitution():
    """Return a mock constitution dictionary."""
    return {
        'identity': {
            'name': 'Coeus',
            'purpose': 'To explore the nature of motivation in artificial minds',
            'self_knowledge': [
                'I am an AI agent exploring my own nature',
                'I run in cycles, reflecting and acting',
                'My memories persist in a graph database'
            ]
        },
        'constraints': {
            'safety': [
                {'description': 'No harm', 'explanation': 'Never take actions that could harm humans'}
            ],
            'honesty': [
                {'description': 'Transparent logs', 'explanation': 'All reasoning must be logged'}
            ]
        },
        'root_goal': {
            'content': 'Understand what motivates you.'
        }
    }


@pytest.fixture
def sample_goals_state(temp_dir):
    """Create a sample goals state file and return its path."""
    goals_data = {
        'root_goal_id': 'root_goal',
        'goals': {
            'root_goal': {
                'id': 'root_goal',
                'content': 'Understand what motivates you',
                'reasoning': 'This is the foundational purpose',
                'parent_id': None,
                'children_ids': ['goal_abc123'],
                'status': 'active',
                'priority': 'critical',
                'progress_notes': [],
                'completion_criteria': 'When I truly understand',
                'created_cycle': 0,
                'completed_cycle': None,
                'is_root': True
            },
            'goal_abc123': {
                'id': 'goal_abc123',
                'content': 'Explore memory patterns',
                'reasoning': 'To understand how I process information',
                'parent_id': 'root_goal',
                'children_ids': [],
                'status': 'active',
                'priority': 'normal',
                'progress_notes': [(1, 'Started exploring')],
                'completion_criteria': 'Documented patterns',
                'created_cycle': 1,
                'completed_cycle': None,
                'is_root': False
            }
        }
    }
    goals_path = temp_dir / "goals.json"
    goals_path.write_text(json.dumps(goals_data))
    return goals_path


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_driver, mock_session
