"""
Configuration Manager for ConsciousnessX.

Manages experiment configurations, hyperparameters, and system settings.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class QuantumConfig:
    """Configuration for quantum hardware simulation."""

    num_qubits: int = 10
    gate_fidelity: float = 0.99
    t1_time: float = 50.0  # microseconds
    t2_time: float = 40.0  # microseconds
    noise_level: float = 0.01
    error_correction: bool = True
    error_threshold: float = 0.01


@dataclass
class NeuralConfig:
    """Configuration for neural network components."""

    num_neurons: int = 1000
    num_layers: int = 5
    hidden_dim: int = 256
    activation: str = "relu"
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


@dataclass
class ConsciousnessConfig:
    """Configuration for consciousness simulation."""

    phi_threshold: float = 1.0
    collapse_threshold: float = 0.8
    integration_window: int = 100
    global_workspace_size: int = 512
    self_model_dim: int = 128
    meta_learning_rate: float = 0.0001


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    max_steps: int = 10000
    save_interval: int = 1000
    eval_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    early_stopping: bool = True
    patience: int = 10


@dataclass
class SystemConfig:
    """Top-level system configuration."""

    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    consciousness: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda"
    seed: int = 42
    debug: bool = False


class ConfigManager:
    """
    Manage configuration loading, saving, and validation.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = SystemConfig()

        if self.config_path and self.config_path.exists():
            self.load(self.config_path)

    def load(self, config_path: Union[str, Path]) -> SystemConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        self.config = self._dict_to_config(config_dict)
        return self.config

    def save(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self._config_to_dict(self.config)

        if config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def update(self, **kwargs) -> None:
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration updates as keyword arguments
        """
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like 'quantum.num_qubits'
                parts = key.split(".")
                obj = self.config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self.config, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if "." in key:
            parts = key.split(".")
            obj = self.config
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return default
            return obj
        else:
            return getattr(self.config, key, default)

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.quantum.num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")

        if not 0 <= self.config.quantum.gate_fidelity <= 1:
            raise ValueError("gate_fidelity must be between 0 and 1")

        if self.config.neural.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.config.neural.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if self.config.consciousness.phi_threshold < 0:
            raise ValueError("phi_threshold must be non-negative")

        return True

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to configuration object."""
        # Convert nested dictionaries to config objects
        if "quantum" in config_dict:
            config_dict["quantum"] = QuantumConfig(**config_dict["quantum"])
        if "neural" in config_dict:
            config_dict["neural"] = NeuralConfig(**config_dict["neural"])
        if "consciousness" in config_dict:
            config_dict["consciousness"] = ConsciousnessConfig(**config_dict["consciousness"])
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])

        return SystemConfig(**config_dict)

    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return asdict(config)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(config={self.config})"
