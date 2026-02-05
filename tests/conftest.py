"""
Pytest configuration and fixtures for consciousnessX tests
"""

import pytest
import numpy as np
import torch
from pathlib import Path


@pytest.fixture
def sample_microtubule_data():
    """Generate sample microtubule data for testing"""
    return {
        "positions": np.random.rand(100, 3),
        "quantum_states": np.random.rand(100, 2) + 1j * np.random.rand(100, 2),
        "tubulin_states": np.random.randint(0, 2, 100),
        "coherence_length": 1.5,
        "temperature": 300.0,
    }


@pytest.fixture
def sample_ion_channel_data():
    """Generate sample ion channel data for testing"""
    return {
        "voltage": np.random.randn(100),
        "current": np.random.randn(100),
        "gating_variables": np.random.rand(100, 3),
        "channel_states": np.random.randint(0, 3, 100),
    }


@pytest.fixture
def sample_quantum_state():
    """Generate sample quantum state for testing"""
    state = np.random.rand(8) + 1j * np.random.rand(8)
    state = state / np.linalg.norm(state)
    return state


@pytest.fixture
def sample_consciousness_metrics():
    """Sample consciousness metrics for testing"""
    return {
        "phi": 0.75,
        "quantum_coherence": 0.82,
        "integrated_information": 0.68,
        "neural_complexity": 0.71,
    }


@pytest.fixture
def mock_model():
    """Mock model for testing"""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    return MockModel()


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory fixture"""
    return tmp_path


@pytest.fixture
def config_path(tmp_path):
    """Create a temporary config file for testing"""
    config_file = tmp_path / "test_config.yml"
    config_file.write_text("""
environment: testing
debug: true
log_level: DEBUG

hardware:
  device: cpu
  num_workers: 1

training:
  batch_size: 8
  learning_rate: 0.01
  epochs: 2
""")
    return str(config_file)
