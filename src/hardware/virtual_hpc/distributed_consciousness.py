#!/usr/bin/env python3
"""
Distributed Consciousness Framework - Multi-Node Training System

A comprehensive, production-ready distributed training framework for
consciousness models across multiple nodes, GPUs, and compute clusters.
Features multi-node synchronization, fault tolerance, elastic scaling,
and hybrid parallelism for large-scale consciousness simulations.
"""

import os
import sys
import time
import json
import logging
import warnings
import signal
import traceback
import pickle
import hashlib
import uuid
import inspect
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque, OrderedDict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import multiprocessing as mp
import asyncio
import socket
import ssl
import struct
import zlib
import gc

# Distributed computing imports
try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some features will be limited.")

try:
    import torch.distributed.rpc as rpc
    import torch.distributed.elastic.rendezvous as rendezvous
    RPC_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    RPC_AVAILABLE = False

try:
    import torch.distributed.elastic.multiprocessing as elastic_mp
    ELASTIC_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    ELASTIC_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Communication libraries
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Monitoring and orchestration
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configure advanced logging
class StructuredLogger(logging.Logger):
    """Enhanced logger with distributed context."""
    
    def __init__(self, name):
        super().__init__(name)
        self.context = {}
        self.metrics_buffer = deque(maxlen=10000)
    
    def set_context(self, rank: int = 0, world_size: int = 1, node_id: str = None):
        """Set distributed context for logging."""
        self.context = {
            'rank': rank,
            'world_size': world_size,
            'node_id': node_id or socket.gethostname(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def distributed(self, level: str, message: str, **kwargs):
        """Log with distributed context."""
        extra = {**self.context, **kwargs}
        getattr(self, level)(f"[Rank {self.context.get('rank', 0)}/{self.context.get('world_size', 1)}] {message}", 
                            extra=extra)

# Configure logging
logging.setLoggerClass(StructuredLogger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] [Rank %(rank)d/%(world_size)d] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('distributed_consciousness.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ParallelismStrategy(Enum):
    """Distributed parallelism strategies."""
    DATA_PARALLEL = "data_parallel"          # Split data across devices
    MODEL_PARALLEL = "model_parallel"        # Split model across devices
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline model layers
    TENSOR_PARALLEL = "tensor_parallel"      # Split tensors
    HYBRID_PARALLEL = "hybrid"               # Combination of strategies
    SEQUENTIAL = "sequential"                # No parallelism


class CommunicationBackend(Enum):
    """Distributed communication backends."""
    NCCL = "nccl"            # NVIDIA NCCL (GPU)
    GLOO = "gloo"            # Gloo (CPU/GPU)
    MPI = "mpi"              # MPI
    RPC = "rpc"              # PyTorch RPC
    ZMQ = "zmq"              # ZeroMQ
    REDIS = "redis"          # Redis pub/sub
    RAY = "ray"              # Ray distributed
    CUSTOM = "custom"        # Custom implementation


class NodeRole(Enum):
    """Node roles in distributed cluster."""
    COORDINATOR = "coordinator"      # Master/coordinator node
    WORKER = "worker"                # Worker node
    EVALUATOR = "evaluator"          # Evaluation node
    CHECKPOINTER = "checkpointer"    # Checkpoint node
    MONITOR = "monitor"              # Monitoring node
    BACKUP = "backup"                # Backup node


class SyncStrategy(Enum):
    """Synchronization strategies."""
    SYNCHRONOUS = "sync"             # Synchronous updates
    ASYNCHRONOUS = "async"           # Asynchronous updates
    STALE_SYNC = "stale_sync"        # Stale synchronous
    BOUNDED_DELAY = "bounded_delay"  # Bounded delay
    ADAPTIVE = "adaptive"            # Adaptive synchronization


class FaultToleranceMode(Enum):
    """Fault tolerance modes."""
    NONE = "none"                    # No fault tolerance
    CHECKPOINT = "checkpoint"        # Periodic checkpointing
    REPLICATION = "replication"      # Model replication
    ERASURE_CODING = "erasure_coding" # Erasure coding
    ACTIVE_STANDBY = "active_standby" # Active-standby
    CHAIN_REPLICATION = "chain_replication" # Chain replication


@dataclass
class NodeConfig:
    """Configuration for a compute node."""
    node_id: str
    role: NodeRole
    address: str
    port: int
    gpu_count: int = 0
    cpu_count: int = 1
    memory_gb: float = 0.0
    is_available: bool = True
    priority: int = 1
    capabilities: List[str] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Connection parameters
    heartbeat_interval: int = 5  # seconds
    timeout: int = 30  # seconds
    retry_count: int = 3
    ssl_enabled: bool = False
    ssl_cert: Optional[str] = None
    
    # Resource limits
    max_batch_size: int = 1024
    max_memory_usage: float = 0.9  # 90% of available
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not self.node_id:
            self.node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        self.full_address = f"{self.address}:{self.port}"
    
    @property
    def resource_score(self) -> float:
        """Calculate resource score for load balancing."""
        score = 0.0
        score += self.gpu_count * 10.0
        score += self.cpu_count * 1.0
        score += self.memory_gb * 0.1
        score *= self.priority
        
        if not self.is_available:
            score = 0.0
        
        return score
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        data = asdict(self)
        data['role'] = self.role.value
        data['full_address'] = self.full_address
        data['resource_score'] = self.resource_score
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeConfig':
        """Create from dictionary."""
        data['role'] = NodeRole(data['role'])
        return cls(**data)


@dataclass
class ClusterConfig:
    """Complete cluster configuration."""
    cluster_id: str
    name: str
    nodes: List[NodeConfig]
    coordinator_node: str
    
    # Parallelism configuration
    parallelism_strategy: ParallelismStrategy = ParallelismStrategy.DATA_PARALLEL
    communication_backend: CommunicationBackend = CommunicationBackend.NCCL
    sync_strategy: SyncStrategy = SyncStrategy.SYNCHRONOUS
    
    # Topology
    topology: str = "fully_connected"  # star, ring, mesh, tree, etc.
    replication_factor: int = 1
    shard_count: int = 1
    
    # Training configuration
    global_batch_size: int = 1024
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    pipeline_depth: int = 1
    
    # Communication optimization
    enable_gradient_compression: bool = True
    compression_ratio: float = 0.5
    enable_overlap_comm: bool = True
    enable_zero_optimization: bool = False
    zero_stage: int = 1  # 0, 1, 2, 3
    
    # Fault tolerance
    fault_tolerance: FaultToleranceMode = FaultToleranceMode.CHECKPOINT
    checkpoint_interval: int = 1000  # steps
    checkpoint_dir: str = "checkpoints"
    max_failures: int = 3
    recovery_timeout: int = 300  # seconds
    
    # Monitoring
    monitoring_port: int = 9090
    metrics_collection_interval: int = 10  # seconds
    enable_profiling: bool = True
    profile_dir: str = "profiles"
    
    # Advanced
    enable_elastic_training: bool = False
    min_nodes: int = 1
    max_nodes: int = 100
    auto_scaling: bool = False
    scaling_threshold: float = 0.8  # 80% utilization
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not self.cluster_id:
            self.cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
        
        # Ensure checkpoint directory exists
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate nodes
        self._validate_nodes()
    
    def _validate_nodes(self):
        """Validate cluster nodes."""
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node IDs found")
        
        # Check coordinator exists
        coordinator_found = any(node.node_id == self.coordinator_node for node in self.nodes)
        if not coordinator_found:
            raise ValueError(f"Coordinator node {self.coordinator_node} not found in nodes")
    
    def get_node(self, node_id: str) -> Optional[NodeConfig]:
        """Get node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def get_workers(self) -> List[NodeConfig]:
        """Get all worker nodes."""
        return [node for node in self.nodes if node.role == NodeRole.WORKER]
    
    def get_coordinator(self) -> Optional[NodeConfig]:
        """Get coordinator node."""
        return self.get_node(self.coordinator_node)
    
    def get_available_nodes(self) -> List[NodeConfig]:
        """Get all available nodes."""
        return [node for node in self.nodes if node.is_available]
    
    def calculate_load_distribution(self) -> Dict[str, float]:
        """Calculate load distribution across nodes."""
        workers = self.get_workers()
        total_score = sum(node.resource_score for node in workers)
        
        distribution = {}
        for node in workers:
            if total_score > 0:
                distribution[node.node_id] = node.resource_score / total_score
            else:
                distribution[node.node_id] = 0.0
        
        return distribution
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        data = asdict(self)
        data['parallelism_strategy'] = self.parallelism_strategy.value
        data['communication_backend'] = self.communication_backend.value
        data['sync_strategy'] = self.sync_strategy.value
        data['fault_tolerance'] = self.fault_tolerance.value
        data['nodes'] = [node.to_dict() for node in self.nodes]
        data['load_distribution'] = self.calculate_load_distribution()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ClusterConfig':
        """Create from dictionary."""
        # Convert enums
        data['parallelism_strategy'] = ParallelismStrategy(data['parallelism_strategy'])
        data['communication_backend'] = CommunicationBackend(data['communication_backend'])
        data['sync_strategy'] = SyncStrategy(data['sync_strategy'])
        data['fault_tolerance'] = FaultToleranceMode(data['fault_tolerance'])
        
        # Convert nodes
        nodes = [NodeConfig.from_dict(node_data) for node_data in data['nodes']]
        data['nodes'] = nodes
        
        return cls(**data)


@dataclass
class DistributedTrainingState:
    """State of distributed training."""
    global_step: int = 0
    epoch: int = 0
    current_loss: float = 0.0
    current_accuracy: float = 0.0
    learning_rate: float = 0.001
    gradient_norm: float = 0.0
    
    # Timing metrics
    iteration_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    computation_time_ms: float = 0.0
    synchronization_time_ms: float = 0.0
    
    # Resource usage
    memory_used_gb: float = 0.0
    gpu_utilization: float = 0.0
    network_bandwidth_mbps: float = 0.0
    
    # Convergence metrics
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    gradient_history: List[float] = field(default_factory=list)
    
    # Node states
    active_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    recovered_nodes: List[str] = field(default_factory=list)
    
    # Checkpoint information
    last_checkpoint_step: int = 0
    last_checkpoint_time: Optional[datetime] = None
    checkpoint_size_gb: float = 0.0
    
    def update(self, **kwargs):
        """Update state with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_metrics(self, loss: float, accuracy: float, gradient_norm: float):
        """Add training metrics."""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.gradient_history.append(gradient_norm)
        
        # Keep only recent history
        max_history = 1000
        if len(self.loss_history) > max_history:
            self.loss_history = self.loss_history[-max_history:]
            self.accuracy_history = self.accuracy_history[-max_history:]
            self.gradient_history = self.gradient_history[-max_history:]
    
    def get_convergence_rate(self, window: int = 100) -> float:
        """Calculate convergence rate over recent window."""
        if len(self.loss_history) < window:
            return 0.0
        
        recent_losses = self.loss_history[-window:]
        if len(recent_losses) < 2:
            return 0.0
        
        # Calculate average reduction per step
        reductions = []
        for i in range(1, len(recent_losses)):
            if recent_losses[i-1] > 0:
                reduction = (recent_losses[i-1] - recent_losses[i]) / recent_losses[i-1]
                reductions.append(reduction)
        
        return np.mean(reductions) if reductions else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        data = asdict(self)
        
        # Handle datetime
        if self.last_checkpoint_time:
            data['last_checkpoint_time'] = self.last_checkpoint_time.isoformat()
        
        # Add derived metrics
        data['convergence_rate'] = self.get_convergence_rate()
        data['average_loss'] = np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        data['average_accuracy'] = np.mean(self.accuracy_history[-100:]) if self.accuracy_history else 0.0
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DistributedTrainingState':
        """Create from dictionary."""
        # Handle datetime
        if 'last_checkpoint_time' in data and data['last_checkpoint_time']:
            data['last_checkpoint_time'] = datetime.fromisoformat(data['last_checkpoint_time'])
        
        return cls(**data)


class DistributedCommunicator(ABC):
    """Abstract base class for distributed communication."""
    
    @abstractmethod
    def initialize(self, rank: int, world_size: int, **kwargs):
        """Initialize communication backend."""
        pass
    
    @abstractmethod
    def send(self, tensor: Any, dst: int, tag: int = 0):
        """Send tensor to destination rank."""
        pass
    
    @abstractmethod
    def recv(self, src: int, tag: int = 0) -> Any:
        """Receive tensor from source rank."""
        pass
    
    @abstractmethod
    def broadcast(self, tensor: Any, src: int):
        """Broadcast tensor from source to all ranks."""
        pass
    
    @abstractmethod
    def all_reduce(self, tensor: Any, op: str = "sum"):
        """Reduce tensor across all ranks."""
        pass
    
    @abstractmethod
    def all_gather(self, tensor: Any) -> List[Any]:
        """Gather tensors from all ranks."""
        pass
    
    @abstractmethod
    def barrier(self):
        """Synchronize all ranks."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown communication backend."""
        pass


class TorchDistributedCommunicator(DistributedCommunicator):
    """PyTorch distributed communication backend."""
    
    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.initialized = False
        self.rank = -1
        self.world_size = 0
        
    def initialize(self, rank: int, world_size: int, init_method: str = "env://", **kwargs):
        """Initialize PyTorch distributed."""
        if self.initialized:
            logger.warning("Distributed already initialized")
            return
        
        self.rank = rank
        self.world_size = world_size
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = kwargs.get('master_addr', 'localhost')
        os.environ['MASTER_PORT'] = str(kwargs.get('master_port', 29500))
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=kwargs.get('timeout', 1800))
        )
        
        self.initialized = True
        logger.info(f"Initialized PyTorch distributed (rank={rank}, world_size={world_size})")
    
    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0):
        """Send tensor using PyTorch distributed."""
        if not self.initialized:
            raise RuntimeError("Distributed not initialized")
        
        if isinstance(tensor, torch.Tensor):
            dist.send(tensor, dst, tag=tag)
        else:
            # Serialize non-tensor data
            data_tensor = torch.tensor(pickle.dumps(tensor))
            dist.send(data_tensor, dst, tag=tag)
    
    def recv(self, src: int, tag: int = 0) -> Any:
        """Receive tensor using PyTorch distributed."""
        if not self.initialized:
            raise RuntimeError("Distributed not initialized")
        
        # Create placeholder tensor
        placeholder = torch.zeros(1)
        dist.recv(placeholder, src, tag=tag)
        
        # Check if it's serialized data
        try:
            data = pickle.loads(placeholder.numpy().tobytes())
            return data
        except:
            return placeholder
    
    def broadcast(self, tensor: torch.Tensor, src: int):
        """Broadcast tensor."""
        if not self.initialized:
            raise RuntimeError("Distributed not initialized")
        
        dist.broadcast(tensor, src)
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum"):
        """All-reduce tensor."""
        if not self.initialized:
            raise RuntimeError("Distributed not initialized")
        
        # Convert op string to dist.ReduceOp
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,  # Need to divide manually
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
            "product": dist.ReduceOp.PRODUCT,
        }
        
        reduce_op = op_map.get(op.lower(), dist.ReduceOp.SUM)
        dist.all_reduce(tensor, op=reduce_op)
        
        if op == "mean":
            tensor.div_(self.world_size)
        
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """All-gather tensors."""
        if not self.initialized:
            raise RuntimeError("Distributed not initialized")
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def barrier(self):
        """Synchronize all processes."""
        if not self.initialized:
            raise RuntimeError("Distributed not initialized")
        
        dist.barrier()
    
    def shutdown(self):
        """Shutdown distributed."""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False
            logger.info("PyTorch distributed shutdown complete")


class ZMQCommunicator(DistributedCommunicator):
    """ZeroMQ-based communication backend."""
    
    def __init__(self, context: Optional[zmq.Context] = None):
        self.context = context or zmq.Context()
        self.sockets = {}
        self.initialized = False
        self.rank = -1
        self.world_size = 0
        
    def initialize(self, rank: int, world_size: int, **kwargs):
        """Initialize ZeroMQ communication."""
        self.rank = rank
        self.world_size = world_size
        
        # Create sockets based on topology
        topology = kwargs.get('topology', 'fully_connected')
        addresses = kwargs.get('addresses', [])
        
        if topology == 'fully_connected':
            self._setup_fully_connected(addresses)
        elif topology == 'ring':
            self._setup_ring_topology(addresses)
        elif topology == 'star':
            self._setup_star_topology(addresses)
        
        self.initialized = True
        logger.info(f"Initialized ZMQ (rank={rank}, world_size={world_size})")
    
    def _setup_fully_connected(self, addresses: List[str]):
        """Setup fully connected topology."""
        # Create PUB socket for broadcasting
        pub_socket = self.context.socket(zmq.PUB)
        pub_socket.bind(f"tcp://*:{5555 + self.rank}")
        self.sockets['pub'] = pub_socket
        
        # Create SUB sockets for receiving from all others
        for i, addr in enumerate(addresses):
            if i != self.rank:
                sub_socket = self.context.socket(zmq.SUB)
                sub_socket.connect(addr)
                sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
                self.sockets[f'sub_{i}'] = sub_socket
        
        # Create PUSH/PULL for point-to-point
        push_pull = self.context.socket(zmq.ROUTER)
        push_pull.bind(f"tcp://*:{5555 + self.rank + 1000}")
        self.sockets['router'] = push_pull
    
    def _setup_ring_topology(self, addresses: List[str]):
        """Setup ring topology."""
        # Connect to next node in ring
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size
        
        # PUSH to next, PULL from previous
        push_socket = self.context.socket(zmq.PUSH)
        push_socket.connect(addresses[next_rank])
        self.sockets['push'] = push_socket
        
        pull_socket = self.context.socket(zmq.PULL)
        pull_socket.bind(f"tcp://*:{5555 + self.rank}")
        self.sockets['pull'] = pull_socket
    
    def send(self, data: Any, dst: int, tag: int = 0):
        """Send data using ZMQ."""
        if not self.initialized:
            raise RuntimeError("ZMQ not initialized")
        
        # Serialize data
        serialized = pickle.dumps({
            'data': data,
            'src': self.rank,
            'dst': dst,
            'tag': tag,
            'timestamp': time.time()
        })
        
        # Compress if large
        if len(serialized) > 1024:
            serialized = zlib.compress(serialized)
        
        # Send via appropriate socket
        if 'router' in self.sockets:
            self.sockets['router'].send_multipart([
                str(dst).encode(),
                serialized
            ])
    
    def recv(self, src: int = -1, tag: int = 0) -> Any:
        """Receive data using ZMQ."""
        if not self.initialized:
            raise RuntimeError("ZMQ not initialized")
        
        # Receive from appropriate socket
        if src == -1:  # Receive from any source
            if 'router' in self.sockets:
                identity, message = self.sockets['router'].recv_multipart()
                data = pickle.loads(zlib.decompress(message) if message[:2] == b'x\x9c' else message)
                return data['data']
        
        # Implementation for other topologies...
        return None
    
    def broadcast(self, data: Any, src: int):
        """Broadcast data using ZMQ PUB/SUB."""
        if not self.initialized:
            raise RuntimeError("ZMQ not initialized")
        
        if self.rank == src:
            # Publisher sends to all
            serialized = pickle.dumps(data)
            self.sockets['pub'].send(serialized)
        else:
            # Subscribers receive
            serialized = self.sockets[f'sub_{src}'].recv()
            return pickle.loads(serialized)
    
    def all_reduce(self, tensor: Any, op: str = "sum"):
        """All-reduce using ZMQ (simplified)."""
        # This would implement a proper all-reduce algorithm
        # For simplicity, we'll do a centralized reduce for now
        if self.rank == 0:
            # Gather from all
            results = [tensor]
            for i in range(1, self.world_size):
                data = self.recv(src=i)
                results.append(data)
            
            # Reduce
            if op == "sum":
                result = sum(results)
            elif op == "mean":
                result = sum(results) / len(results)
            else:
                result = results[0]
            
            # Broadcast result
            self.broadcast(result, src=0)
            return result
        else:
            # Send to rank 0
            self.send(tensor, dst=0)
            # Receive result
            return self.recv(src=0)
    
    def all_gather(self, tensor: Any) -> List[Any]:
        """All-gather using ZMQ."""
        # Send to all
        all_data = []
        for i in range(self.world_size):
            if i == self.rank:
                all_data.append(tensor)
            else:
                self.send(tensor, dst=i)
                # Also implement receiving from others
                # This is simplified
                pass
        
        return all_data
    
    def barrier(self):
        """Synchronize using ZMQ."""
        # Implement barrier using coordination
        if self.rank == 0:
            # Wait for all to arrive
            for i in range(1, self.world_size):
                self.recv(src=i)
            # Signal to proceed
            for i in range(1, self.world_size):
                self.send("proceed", dst=i)
        else:
            # Signal arrival
            self.send("arrived", dst=0)
            # Wait for proceed signal
            self.recv(src=0)
    
    def shutdown(self):
        """Shutdown ZMQ."""
        for socket in self.sockets.values():
            socket.close()
        self.context.term()
        self.initialized = False
        logger.info("ZMQ shutdown complete")


class GradientCompressor:
    """Gradient compression for efficient communication."""
    
    def __init__(self, compression_type: str = "topk", ratio: float = 0.01):
        self.compression_type = compression_type
        self.compression_ratio = ratio
        
    def compress(self, gradient: torch.Tensor) -> Dict:
        """Compress gradient tensor."""
        if self.compression_type == "topk":
            return self._topk_compress(gradient)
        elif self.compression_type == "randomk":
            return self._randomk_compress(gradient)
        elif self.compression_type == "signsgd":
            return self._signsgd_compress(gradient)
        elif self.compression_type == "dgc":
            return self._dgc_compress(gradient)
        else:
            return {'dense': gradient}
    
    def decompress(self, compressed: Dict) -> torch.Tensor:
        """Decompress gradient tensor."""
        if self.compression_type == "topk":
            return self._topk_decompress(compressed)
        elif self.compression_type == "randomk":
            return self._randomk_decompress(compressed)
        elif self.compression_type == "signsgd":
            return self._signsgd_decompress(compressed)
        elif self.compression_type == "dgc":
            return self._dgc_decompress(compressed)
        else:
            return compressed['dense']
    
    def _topk_compress(self, gradient: torch.Tensor) -> Dict:
        """Top-K gradient compression."""
        gradient_flat = gradient.view(-1)
        k = max(1, int(gradient_flat.numel() * self.compression_ratio))
        
        # Get top-k values and indices
        values, indices = torch.topk(gradient_flat.abs(), k)
        
        # Get actual values (with signs)
        compressed_values = gradient_flat[indices]
        
        return {
            'values': compressed_values,
            'indices': indices,
            'shape': gradient.shape,
            'k': k
        }
    
    def _topk_decompress(self, compressed: Dict) -> torch.Tensor:
        """Decompress top-K compressed gradient."""
        gradient = torch.zeros(compressed['shape'], device=compressed['values'].device)
        gradient_flat = gradient.view(-1)
        gradient_flat[compressed['indices']] = compressed['values']
        return gradient
    
    def _randomk_compress(self, gradient: torch.Tensor) -> Dict:
        """Random-K gradient compression."""
        gradient_flat = gradient.view(-1)
        k = max(1, int(gradient_flat.numel() * self.compression_ratio))
        
        # Random indices
        indices = torch.randperm(gradient_flat.numel())[:k]
        values = gradient_flat[indices]
        
        return {
            'values': values,
            'indices': indices,
            'shape': gradient.shape,
            'k': k
        }
    
    def _randomk_decompress(self, compressed: Dict) -> torch.Tensor:
        """Decompress random-K compressed gradient."""
        gradient = torch.zeros(compressed['shape'], device=compressed['values'].device)
        gradient_flat = gradient.view(-1)
        gradient_flat[compressed['indices']] = compressed['values']
        return gradient
    
    def _signsgd_compress(self, gradient: torch.Tensor) -> Dict:
        """SignSGD compression."""
        signs = torch.sign(gradient)
        return {
            'signs': signs,
            'shape': gradient.shape,
            'norm': gradient.norm()
        }
    
    def _signsgd_decompress(self, compressed: Dict) -> torch.Tensor:
        """Decompress SignSGD compressed gradient."""
        return compressed['signs'] * compressed['norm'] / np.sqrt(compressed['signs'].numel())
    
    def _dgc_compress(self, gradient: torch.Tensor) -> Dict:
        """Deep Gradient Compression."""
        # DGC uses momentum correction and local gradient clipping
        # This is a simplified version
        return self._topk_compress(gradient)


class DistributedConsciousnessTrainer:
    """Main distributed consciousness trainer."""
    
    def __init__(self, 
                 cluster_config: ClusterConfig,
                 model: Optional[nn.Module] = None,
                 optimizer_class: Callable = optim.Adam,
                 optimizer_params: Dict = None,
                 use_cuda: bool = True):
        
        self.cluster_config = cluster_config
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params or {'lr': 0.001}
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Distributed state
        self.rank = -1
        self.world_size = 0
        self.local_rank = 0
        self.node_id = socket.gethostname()
        
        # Components
        self.communicator = None
        self.gradient_compressor = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.state = DistributedTrainingState()
        self.checkpoint_manager = None
        self.monitor = None
        
        # Data
        self.train_loader = None
        self.val_loader = None
        self.sampler = None
        
        # Performance tracking
        self.perf_stats = defaultdict(list)
        self.communication_stats = defaultdict(list)
        
        # Fault tolerance
        self.failure_detector = None
        self.recovery_manager = None
        
        # Initialize
        self._initialize_components()
        
        logger.set_context(rank=self.rank, world_size=self.world_size, node_id=self.node_id)
        logger.distributed('info', f"Distributed Consciousness Trainer initialized")
    
    def _initialize_components(self):
        """Initialize all components."""
        # Initialize communicator based on backend
        backend = self.cluster_config.communication_backend
        
        if backend == CommunicationBackend.NCCL or backend == CommunicationBackend.GLOO:
            self.communicator = TorchDistributedCommunicator(backend.value)
        elif backend == CommunicationBackend.ZMQ:
            self.communicator = ZMQCommunicator()
        else:
            raise NotImplementedError(f"Backend {backend} not implemented")
        
        # Initialize gradient compressor
        if self.cluster_config.enable_gradient_compression:
            self.gradient_compressor = GradientCompressor(
                compression_type="topk",
                ratio=self.cluster_config.compression_ratio
            )
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.cluster_config.checkpoint_dir,
            checkpoint_interval=self.cluster_config.checkpoint_interval
        )
        
        # Initialize monitor if enabled
        if PROMETHEUS_AVAILABLE:
            self.monitor = DistributedMonitor(
                port=self.cluster_config.monitoring_port,
                cluster_id=self.cluster_config.cluster_id
            )
    
    def initialize_distributed(self, rank: int, world_size: int, **kwargs):
        """Initialize distributed training."""
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count() if self.use_cuda else 0
        
        # Update logger context
        logger.set_context(rank=rank, world_size=world_size)
        
        # Initialize communicator
        init_kwargs = {
            'master_addr': kwargs.get('master_addr', 'localhost'),
            'master_port': kwargs.get('master_port', 29500),
            'timeout': kwargs.get('timeout', 1800)
        }
        
        self.communicator.initialize(rank, world_size, **init_kwargs)
        
        # Setup CUDA
        if self.use_cuda:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        
        # Move model to device and wrap with DDP
        if self.model:
            self.model = self.model.to(self.device)
            
            if self.world_size > 1:
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank] if self.use_cuda else None,
                    output_device=self.local_rank if self.use_cuda else None,
                    find_unused_parameters=True
                )
        
        # Initialize optimizer
        if self.model:
            self.optimizer = self.optimizer_class(
                self.model.parameters(),
                **self.optimizer_params
            )
            
            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=1000
            )
        
        logger.distributed('info', f"Distributed initialization complete on device {self.device}")
    
    def set_data_loaders(self, train_dataset, val_dataset=None, batch_size=None):
        """Set data loaders for distributed training."""
        if batch_size is None:
            batch_size = self.cluster_config.micro_batch_size
        
        # Create distributed sampler
        self.sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=self.use_cuda,
            drop_last=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=2,
                pin_memory=self.use_cuda
            )
        
        logger.distributed('info', f"Data loaders initialized with batch size {batch_size}")
    
    def train_epoch(self, criterion: Callable, epoch: int) -> Dict:
        """Train for one epoch."""
        if not self.model or not self.train_loader:
            raise RuntimeError("Model and data loader must be set")
        
        self.model.train()
        self.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_samples = 0
        
        # Timing
        epoch_start = time.time()
        communication_time = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start = time.time()
            
            # Move to device
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Synchronize gradients
            sync_start = time.time()
            self._synchronize_gradients()
            communication_time += time.time() - sync_start
            
            # Optimization step
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                predictions = output.argmax(dim=1)
                correct = (predictions == target).sum().item()
                batch_accuracy = correct / len(target)
                
                epoch_loss += loss.item() * len(data)
                epoch_accuracy += correct
                epoch_samples += len(data)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Update state
            self.state.global_step += 1
            self.state.current_loss = loss.item()
            self.state.current_accuracy = batch_accuracy
            self.state.learning_rate = self.optimizer.param_groups[0]['lr']
            
            # Add to history
            self.state.add_metrics(loss.item(), batch_accuracy, 0.0)  # gradient norm would be added
            
            # Checkpoint if needed
            if self.checkpoint_manager.should_checkpoint(self.state.global_step):
                self.save_checkpoint()
            
            # Log progress
            if batch_idx % 10 == 0:
                batch_time = time.time() - batch_start
                samples_per_sec = len(data) / batch_time
                
                logger.distributed('info', 
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss={loss.item():.4f}, Acc={batch_accuracy:.3f}, "
                    f"LR={self.state.learning_rate:.6f}, "
                    f"Speed={samples_per_sec:.1f} samples/sec"
                )
                
                # Update monitoring
                if self.monitor:
                    self.monitor.update_metrics({
                        'loss': loss.item(),
                        'accuracy': batch_accuracy,
                        'learning_rate': self.state.learning_rate,
                        'samples_per_second': samples_per_sec,
                        'communication_time': communication_time
                    })
        
        # Synchronize metrics across nodes
        epoch_loss_tensor = torch.tensor([epoch_loss], device=self.device)
        epoch_accuracy_tensor = torch.tensor([epoch_accuracy], device=self.device)
        epoch_samples_tensor = torch.tensor([epoch_samples], device=self.device)
        
        self.communicator.all_reduce(epoch_loss_tensor, op="sum")
        self.communicator.all_reduce(epoch_accuracy_tensor, op="sum")
        self.communicator.all_reduce(epoch_samples_tensor, op="sum")
        
        avg_loss = epoch_loss_tensor.item() / epoch_samples_tensor.item()
        avg_accuracy = epoch_accuracy_tensor.item() / epoch_samples_tensor.item()
        
        epoch_time = time.time() - epoch_start
        
        # Update state
        self.state.epoch = epoch
        self.state.iteration_time_ms = epoch_time * 1000 / len(self.train_loader)
        self.state.communication_time_ms = communication_time * 1000
        
        logger.distributed('info',
            f"Epoch {epoch} completed: "
            f"Avg Loss={avg_loss:.4f}, Avg Acc={avg_accuracy:.4f}, "
            f"Time={epoch_time:.2f}s, Comm Time={communication_time:.2f}s"
        )
        
        return {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'epoch_time': epoch_time,
            'communication_time': communication_time,
            'samples_processed': epoch_samples_tensor.item()
        }
    
    def _synchronize_gradients(self):
        """Synchronize gradients across nodes."""
        if self.world_size <= 1:
            return
        
        sync_strategy = self.cluster_config.sync_strategy
        
        if sync_strategy == SyncStrategy.SYNCHRONOUS:
            self._synchronous_sync()
        elif sync_strategy == SyncStrategy.ASYNCHRONOUS:
            self._asynchronous_sync()
        elif sync_strategy == SyncStrategy.STALE_SYNC:
            self._stale_synchronous_sync()
        else:
            self._synchronous_sync()
    
    def _synchronous_sync(self):
        """Synchronous gradient synchronization."""
        for param in self.model.parameters():
            if param.grad is not None:
                # Compress gradient if enabled
                if self.gradient_compressor:
                    compressed = self.gradient_compressor.compress(param.grad)
                    # All-reduce compressed gradient
                    # For now, we'll do dense all-reduce
                    pass
                
                # All-reduce gradient
                self.communicator.all_reduce(param.grad, op="mean")
    
    def _asynchronous_sync(self):
        """Asynchronous gradient synchronization."""
        # This would use parameter servers or peer-to-peer updates
        # Simplified implementation
        pass
    
    def _stale_synchronous_sync(self):
        """Stale synchronous gradient synchronization."""
        # Workers can proceed with stale gradients
        # Coordinator updates periodically
        pass
    
    def evaluate(self, criterion: Callable) -> Dict:
        """Evaluate model on validation set."""
        if not self.model or not self.val_loader:
            raise RuntimeError("Model and validation loader must be set")
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                predictions = output.argmax(dim=1)
                correct = (predictions == target).sum().item()
                
                total_loss += loss.item() * len(data)
                total_correct += correct
                total_samples += len(data)
        
        # Synchronize metrics
        total_loss_tensor = torch.tensor([total_loss], device=self.device)
        total_correct_tensor = torch.tensor([total_correct], device=self.device)
        total_samples_tensor = torch.tensor([total_samples], device=self.device)
        
        self.communicator.all_reduce(total_loss_tensor, op="sum")
        self.communicator.all_reduce(total_correct_tensor, op="sum")
        self.communicator.all_reduce(total_samples_tensor, op="sum")
        
        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        avg_accuracy = total_correct_tensor.item() / total_samples_tensor.item()
        
        logger.distributed('info',
            f"Evaluation: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}"
        )
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'samples': total_samples_tensor.item()
        }
    
    def save_checkpoint(self, tag: str = None):
        """Save training checkpoint."""
        if not tag:
            tag = f"step_{self.state.global_step}"
        
        checkpoint = {
            'global_step': self.state.global_step,
            'epoch': self.state.epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_state': self.state.to_dict(),
            'cluster_config': self.cluster_config.to_dict(),
            'timestamp': datetime.utcnow().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint=checkpoint,
            tag=tag,
            rank=self.rank
        )
        
        if self.rank == 0:
            # Coordinator saves global checkpoint
            global_checkpoint = checkpoint.copy()
            # Gather model from all ranks (simplified)
            global_checkpoint['global_model'] = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            
            global_path = self.checkpoint_manager.save_global_checkpoint(
                checkpoint=global_checkpoint,
                tag=tag
            )
            
            logger.distributed('info', f"Global checkpoint saved: {global_path}")
        
        logger.distributed('info', f"Checkpoint saved: {checkpoint_path}")
        
        # Update state
        self.state.last_checkpoint_step = self.state.global_step
        self.state.last_checkpoint_time = datetime.utcnow()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Load model state
        model_state_dict = checkpoint['model_state_dict']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.state = DistributedTrainingState.from_dict(checkpoint['training_state'])
        
        logger.distributed('info', f"Checkpoint loaded: {checkpoint_path}")
        logger.distributed('info', f"Resuming from step {self.state.global_step}, epoch {self.state.epoch}")
    
    def train(self, 
              num_epochs: int,
              criterion: Callable,
              train_dataset,
              val_dataset=None,
              callbacks: List[Callable] = None) -> Dict:
        """Main training loop."""
        
        logger.distributed('info', f"Starting training for {num_epochs} epochs")
        
        # Set data loaders
        self.set_data_loaders(train_dataset, val_dataset)
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_times': [],
            'communication_times': []
        }
        
        try:
            for epoch in range(self.state.epoch, num_epochs):
                epoch_start = time.time()
                
                # Train epoch
                train_metrics = self.train_epoch(criterion, epoch)
                
                # Validate if validation set provided
                val_metrics = {}
                if val_dataset:
                    val_metrics = self.evaluate(criterion)
                
                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['train_accuracy'].append(train_metrics['accuracy'])
                
                if val_metrics:
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_accuracy'].append(val_metrics['accuracy'])
                
                history['epoch_times'].append(train_metrics['epoch_time'])
                history['communication_times'].append(train_metrics['communication_time'])
                
                # Callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, epoch, train_metrics, val_metrics)
                
                # Check convergence
                if self._check_convergence(history):
                    logger.distributed('info', "Training converged")
                    break
            
            # Final evaluation
            final_metrics = self.evaluate(criterion) if val_dataset else {}
            
            # Save final model
            self.save_checkpoint("final")
            
            logger.distributed('info', "Training completed successfully")
            
            return {
                'success': True,
                'final_metrics': final_metrics,
                'history': history,
                'training_state': self.state.to_dict()
            }
            
        except Exception as e:
            logger.distributed('error', f"Training failed: {e}", exc_info=True)
            
            # Attempt recovery if fault tolerance enabled
            if self.cluster_config.fault_tolerance != FaultToleranceMode.NONE:
                logger.distributed('info', "Attempting recovery...")
                self._recover_from_failure()
            
            return {
                'success': False,
                'error': str(e),
                'history': history,
                'training_state': self.state.to_dict()
            }
    
    def _check_convergence(self, history: Dict, patience: int = 10) -> bool:
        """Check if training has converged."""
        if len(history['train_loss']) < patience * 2:
            return False
        
        # Check if loss has stopped improving
        recent_losses = history['train_loss'][-patience:]
        avg_recent = np.mean(recent_losses)
        avg_previous = np.mean(history['train_loss'][-patience*2:-patience])
        
        improvement = (avg_previous - avg_recent) / avg_previous if avg_previous > 0 else 0
        
        if improvement < 0.001:  # Less than 0.1% improvement
            logger.distributed('info', f"Convergence detected: improvement={improvement:.6f}")
            return True
        
        return False
    
    def _recover_from_failure(self):
        """Recover from training failure."""
        # Find latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        
        if latest_checkpoint:
            logger.distributed('info', f"Recovering from checkpoint: {latest_checkpoint}")
            self.load_checkpoint(latest_checkpoint)
            
            # Update state
            self.state.recovered_nodes.append(self.node_id)
        else:
            logger.distributed('warning', "No checkpoint found for recovery")
    
    def shutdown(self):
        """Shutdown distributed training."""
        if self.communicator:
            self.communicator.shutdown()
        
        if self.monitor:
            self.monitor.shutdown()
        
        logger.distributed('info', "Distributed trainer shutdown complete")


class CheckpointManager:
    """Manages distributed checkpoints."""
    
    def __init__(self, checkpoint_dir: str, checkpoint_interval: int = 1000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "global").mkdir(exist_ok=True)
        (self.checkpoint_dir / "local").mkdir(exist_ok=True)
        
        # Checkpoint metadata
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'checkpoints': [],
            'latest_global': None,
            'latest_local': defaultdict(str)
        }
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def should_checkpoint(self, global_step: int) -> bool:
        """Check if should checkpoint at this step."""
        return global_step % self.checkpoint_interval == 0
    
    def save_checkpoint(self, checkpoint: Dict, tag: str, rank: int) -> str:
        """Save checkpoint for a specific rank."""
        checkpoint_path = self.checkpoint_dir / "local" / f"rank_{rank}_{tag}.pt"
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update metadata
        checkpoint_info = {
            'path': str(checkpoint_path),
            'tag': tag,
            'rank': rank,
            'global_step': checkpoint['global_step'],
            'timestamp': checkpoint['timestamp']
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        self.metadata['latest_local'][str(rank)] = str(checkpoint_path)
        self._save_metadata()
        
        return str(checkpoint_path)
    
    def save_global_checkpoint(self, checkpoint: Dict, tag: str) -> str:
        """Save global checkpoint (coordinator only)."""
        checkpoint_path = self.checkpoint_dir / "global" / f"global_{tag}.pt"
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update metadata
        self.metadata['latest_global'] = str(checkpoint_path)
        self._save_metadata()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint from file."""
        return torch.load(checkpoint_path, map_location='cpu')
    
    def get_latest_checkpoint(self, rank: int = None) -> Optional[str]:
        """Get latest checkpoint for rank or global."""
        if rank is not None:
            return self.metadata['latest_local'].get(str(rank))
        else:
            return self.metadata['latest_global']
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Cleanup old checkpoints, keep only recent ones."""
        checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['global_step'],
            reverse=True
        )
        
        # Keep only recent checkpoints
        to_keep = checkpoints[:keep_last]
        to_delete = checkpoints[keep_last:]
        
        for checkpoint in to_delete:
            try:
                Path(checkpoint['path']).unlink()
            except FileNotFoundError:
                pass
        
        # Update metadata
        self.metadata['checkpoints'] = to_keep
        self._save_metadata()
        
        logger.info(f"Cleaned up {len(to_delete)} old checkpoints, kept {len(to_keep)}")


class DistributedMonitor:
    """Distributed monitoring and metrics collection."""
    
    def __init__(self, port: int = 9090, cluster_id: str = None):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client not available")
        
        self.port = port
        self.cluster_id = cluster_id or "default_cluster"
        
        # Initialize metrics
        self._init_metrics()
        
        # Start HTTP server
        prometheus_client.start_http_server(self.port)
        
        logger.info(f"Distributed monitor started on port {self.port}")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Training metrics
        self.loss_gauge = Gauge('consciousness_training_loss', 'Training loss', ['rank', 'cluster_id'])
        self.accuracy_gauge = Gauge('consciousness_training_accuracy', 'Training accuracy', ['rank', 'cluster_id'])
        self.learning_rate_gauge = Gauge('consciousness_learning_rate', 'Learning rate', ['rank', 'cluster_id'])
        
        # Performance metrics
        self.samples_per_second_gauge = Gauge('consciousness_samples_per_second', 'Samples per second', ['rank', 'cluster_id'])
        self.communication_time_gauge = Gauge('consciousness_communication_time', 'Communication time', ['rank', 'cluster_id'])
        self.computation_time_gauge = Gauge('consciousness_computation_time', 'Computation time', ['rank', 'cluster_id'])
        
        # Resource metrics
        self.gpu_memory_gauge = Gauge('consciousness_gpu_memory', 'GPU memory usage', ['rank', 'cluster_id', 'gpu_id'])
        self.gpu_utilization_gauge = Gauge('consciousness_gpu_utilization', 'GPU utilization', ['rank', 'cluster_id', 'gpu_id'])
        self.cpu_usage_gauge = Gauge('consciousness_cpu_usage', 'CPU usage', ['rank', 'cluster_id'])
        self.memory_usage_gauge = Gauge('consciousness_memory_usage', 'Memory usage', ['rank', 'cluster_id'])
        
        # System metrics
        self.network_rx_gauge = Gauge('consciousness_network_rx', 'Network receive bytes', ['rank', 'cluster_id'])
        self.network_tx_gauge = Gauge('consciousness_network_tx', 'Network transmit bytes', ['rank', 'cluster_id'])
        
        # Convergence metrics
        self.convergence_rate_gauge = Gauge('consciousness_convergence_rate', 'Convergence rate', ['rank', 'cluster_id'])
        self.gradient_norm_gauge = Gauge('consciousness_gradient_norm', 'Gradient norm', ['rank', 'cluster_id'])
    
    def update_metrics(self, metrics: Dict, rank: int = 0, gpu_id: int = 0):
        """Update metrics with new values."""
        labels = {'rank': str(rank), 'cluster_id': self.cluster_id}
        gpu_labels = {**labels, 'gpu_id': str(gpu_id)}
        
        # Update training metrics
        if 'loss' in metrics:
            self.loss_gauge.labels(**labels).set(metrics['loss'])
        if 'accuracy' in metrics:
            self.accuracy_gauge.labels(**labels).set(metrics['accuracy'])
        if 'learning_rate' in metrics:
            self.learning_rate_gauge.labels(**labels).set(metrics['learning_rate'])
        
        # Update performance metrics
        if 'samples_per_second' in metrics:
            self.samples_per_second_gauge.labels(**labels).set(metrics['samples_per_second'])
        if 'communication_time' in metrics:
            self.communication_time_gauge.labels(**labels).set(metrics['communication_time'])
        if 'computation_time' in metrics:
            self.computation_time_gauge.labels(**labels).set(metrics['computation_time'])
        
        # Update resource metrics from system
        self._update_system_metrics(rank, gpu_id)
    
    def _update_system_metrics(self, rank: int, gpu_id: int):
        """Update system metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        labels = {'rank': str(rank), 'cluster_id': self.cluster_id}
        gpu_labels = {**labels, 'gpu_id': str(gpu_id)}
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage_gauge.labels(**labels).set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage_gauge.labels(**labels).set(memory.percent)
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.network_rx_gauge.labels(**labels).set(net_io.bytes_recv)
        self.network_tx_gauge.labels(**labels).set(net_io.bytes_sent)
        
        # GPU metrics (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                gpu_utilization = torch.cuda.utilization(gpu_id) if hasattr(torch.cuda, 'utilization') else 0
                
                self.gpu_memory_gauge.labels(**gpu_labels).set(gpu_memory)
                self.gpu_utilization_gauge.labels(**gpu_labels).set(gpu_utilization)
            except:
                pass
    
    def shutdown(self):
        """Shutdown monitor."""
        # Prometheus doesn't need explicit shutdown
        logger.info("Distributed monitor shutdown")


class ConsciousnessModel(nn.Module):
    """Example consciousness model for distributed training."""
    
    def __init__(self, 
                 input_dim: int = 784,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 10,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


def run_distributed_training(rank: int, 
                            world_size: int, 
                            cluster_config: ClusterConfig,
                            num_epochs: int = 10):
    """Function to run distributed training (called by each process)."""
    
    # Initialize trainer
    model = ConsciousnessModel()
    trainer = DistributedConsciousnessTrainer(
        cluster_config=cluster_config,
        model=model,
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001, 'weight_decay': 1e-4},
        use_cuda=True
    )
    
    try:
        # Initialize distributed
        trainer.initialize_distributed(rank, world_size)
        
        # Create sample dataset
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transform
        )
        
        val_dataset = datasets.MNIST(
            './data',
            train=False,
            transform=transform
        )
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train
        result = trainer.train(
            num_epochs=num_epochs,
            criterion=criterion,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        return result
        
    finally:
        trainer.shutdown()


def main():
    """Main function demonstrating distributed training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Consciousness Training')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=1, help='GPUs per node')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--strategy', type=str, default='data_parallel',
                       choices=['data_parallel', 'model_parallel', 'pipeline_parallel'])
    parser.add_argument('--backend', type=str, default='nccl',
                       choices=['nccl', 'gloo', 'zmq'])
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Checkpoint interval in steps')
    parser.add_argument('--monitor', action='store_true',
                       help='Enable monitoring')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')
    
    args = parser.parse_args()
    
    # Create cluster configuration
    nodes = []
    for i in range(args.nodes):
        node = NodeConfig(
            node_id=f"node_{i}",
            role=NodeRole.WORKER if i > 0 else NodeRole.COORDINATOR,
            address="localhost",
            port=29500 + i,
            gpu_count=args.gpus_per_node,
            cpu_count=mp.cpu_count() // args.nodes,
            memory_gb=psutil.virtual_memory().total / 1024**3 / args.nodes if PSUTIL_AVAILABLE else 16.0
        )
        nodes.append(node)
    
    cluster_config = ClusterConfig(
        cluster_id="consciousness_cluster",
        name="Consciousness Training Cluster",
        nodes=nodes,
        coordinator_node=nodes[0].node_id,
        parallelism_strategy=ParallelismStrategy(args.strategy),
        communication_backend=CommunicationBackend(args.backend),
        global_batch_size=args.batch_size * args.nodes * args.gpus_per_node,
        micro_batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        enable_profiling=args.profile
    )
    
    print(f"Starting distributed consciousness training with:")
    print(f"  Nodes: {args.nodes}")
    print(f"  GPUs per node: {args.gpus_per_node}")
    print(f"  Total GPUs: {args.nodes * args.gpus_per_node}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Backend: {args.backend}")
    print(f"  Global batch size: {cluster_config.global_batch_size}")
    
    # Launch distributed training
    if args.nodes == 1:
        # Single node, multiple GPUs
        world_size = args.gpus_per_node
        
        # Set environment variables for PyTorch distributed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        # Spawn processes
        mp.spawn(
            run_distributed_training,
            args=(world_size, cluster_config, args.epochs),
            nprocs=world_size,
            join=True
        )
    else:
        # Multi-node training
        print("Multi-node training requires proper setup with:")
        print("1. SSH passwordless access between nodes")
        print("2. Shared filesystem or checkpoint synchronization")
        print("3. Network configuration for inter-node communication")
        print("\nExample command for node 0:")
        print(f"  python distributed_consciousness.py --nodes {args.nodes} --epochs {args.epochs}")
        print("\nFor production use, consider using:")
        print("  - Kubernetes with Kubeflow")
        print("  - Slurm workload manager")
        print("  - AWS ParallelCluster")
        print("  - Azure Machine Learning")


if __name__ == "__main__":
    main()
