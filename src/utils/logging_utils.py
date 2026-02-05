"""
Logging utilities for ConsciousnessX framework.

Provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'consciousnessx',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training progress with formatted output.
    """
    
    def __init__(
        self,
        name: str = 'training',
        log_dir: Optional[str] = None,
        log_to_file: bool = True
    ):
        """
        Initialize training logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_to_file: Whether to log to file
        """
        self.logger = get_logger(name)
        self.log_dir = log_dir
        self.log_to_file = log_to_file
        
        if log_dir and log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_path / f'training_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_step(
        self,
        step: int,
        loss: float,
        metric: Optional[float] = None,
        lr: Optional[float] = None,
        extra: Optional[dict] = None
    ) -> None:
        """
        Log training step information.
        
        Args:
            step: Training step number
            loss: Loss value
            metric: Optional metric value
            lr: Optional learning rate
            extra: Optional extra information
        """
        msg = f"Step {step:6d} | Loss: {loss:8.4f}"
        
        if metric is not None:
            msg += f" | Metric: {metric:8.4f}"
        
        if lr is not None:
            msg += f" | LR: {lr:.6f}"
        
        if extra:
            for key, value in extra.items():
                msg += f" | {key}: {value}"
        
        self.logger.info(msg)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        time_elapsed: Optional[float] = None
    ) -> None:
        """
        Log epoch summary.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Optional validation loss
            time_elapsed: Optional time elapsed in seconds
        """
        msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:8.4f}"
        
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:8.4f}"
        
        if time_elapsed is not None:
            msg += f" | Time: {time_elapsed:.2f}s"
        
        self.logger.info(msg)
    
    def log_metrics(self, metrics: dict, prefix: str = '') -> None:
        """
        Log a dictionary of metrics.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Optional prefix for metric names
        """
        for key, value in metrics.items():
            name = f"{prefix}_{key}" if prefix else key
            self.logger.info(f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}")


class ProgressLogger:
    """
    Logger for progress tracking with percentage and ETA.
    """
    
    def __init__(self, total: int, name: str = 'Progress'):
        """
        Initialize progress logger.
        
        Args:
            total: Total number of items
            name: Name of the progress tracker
        """
        self.total = total
        self.name = name
        self.current = 0
        self.logger = get_logger('progress')
        self.start_time = None
    
    def start(self) -> None:
        """Start progress tracking."""
        import time
        self.start_time = time.time()
        self.logger.info(f"{self.name}: Started (0/{self.total})")
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress.
        
        Args:
            increment: Number of items completed
        """
        import time
        self.current += increment
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            eta = (elapsed / self.current) * (self.total - self.current)
            
            msg = (f"{self.name}: {self.current}/{self.total} "
                   f"({self.current/self.total*100:.1f}%) | "
                   f"ETA: {eta:.1f}s")
            
            self.logger.info(msg)
    
    def finish(self) -> None:
        """Finish progress tracking."""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.info(f"{self.name}: Completed in {elapsed:.2f}s")


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} returned {result}")
        return result
    return wrapper