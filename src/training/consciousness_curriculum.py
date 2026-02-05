#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consciousness Training Curriculum
Progressive training framework for artificial consciousness development
Production-ready with curriculum learning and adaptive difficulty
"""

import numpy as np
import torch
import logging
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import deque, OrderedDict
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingStage(Enum):
    """Stages of consciousness development"""
    PRE_CONSCIOUS = "pre_conscious"
    SENSORY_INTEGRATION = "sensory_integration"
    ATTENTION_DEVELOPMENT = "attention_development"
    MEMORY_FORMATION = "memory_formation"
    SELF_MODELING = "self_modeling"
    META_COGNITION = "meta_cognition"
    FULL_CONSCIOUSNESS = "full_consciousness"

class CurriculumTask(Enum):
    """Specific tasks in the consciousness curriculum"""
    PATTERN_RECOGNITION = "pattern_recognition"
    TEMPORAL_SEQUENCING = "temporal_sequencing"
    ATTENTION_SHIFTING = "attention_shifting"
    WORKING_MEMORY = "working_memory"
    GOAL_DIRECTED_BEHAVIOR = "goal_directed_behavior"
    SELF_RECOGNITION = "self_recognition"
    META_COGNITIVE_REFLECTION = "meta_cognitive_reflection"
    INTEGRATED_REASONING = "integrated_reasoning"

@dataclass
class CurriculumConfig:
    """Configuration for consciousness training curriculum"""
    
    # Training parameters
    max_epochs: int = 1000
    episodes_per_stage: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.95
    min_learning_rate: float = 1e-6
    
    # Curriculum parameters
    starting_stage: TrainingStage = TrainingStage.PRE_CONSCIOUS
    auto_advance: bool = True
    advancement_threshold: float = 0.8  # Performance threshold to advance
    mastery_threshold: float = 0.9  # Performance for mastery
    regression_threshold: float = 0.6  # Performance below which to regress
    
    # Task generation parameters
    min_complexity: int = 1
    max_complexity: int = 10
    complexity_increment: float = 0.1
    
    # Reward shaping
    use_intrinsic_rewards: bool = True
    curiosity_weight: float = 0.1
    novelty_weight: float = 0.05
    competence_weight: float = 0.2
    
    # Assessment parameters
    assessment_frequency: int = 10  # Episodes between assessments
    save_checkpoint_frequency: int = 50
    
    # Output parameters
    save_directory: str = "./training_results"
    tensorboard_logging: bool = True
    checkpoint_keep_last: int = 10
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        
        if not 0 < self.advancement_threshold <= 1:
            raise ValueError("advancement_threshold must be in (0, 1]")
        
        if not 0 < self.mastery_threshold <= 1:
            raise ValueError("mastery_threshold must be in (0, 1]")
        
        if not 0 <= self.regression_threshold < self.advancement_threshold:
            raise ValueError("regression_threshold must be < advancement_threshold")
        
        # Create save directory
        Path(self.save_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Curriculum configured with {len(TrainingStage)} stages")

class TaskGenerator:
    """Generator for consciousness development tasks"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.task_templates = self._initialize_task_templates()
        self.complexity_level = config.min_complexity
        
        logger.info("Task generator initialized")
    
    def _initialize_task_templates(self) -> Dict[CurriculumTask, Dict]:
        """Initialize templates for each task type"""
        return {
            CurriculumTask.PATTERN_RECOGNITION: {
                'description': 'Recognize and classify patterns',
                'min_complexity': 1,
                'max_complexity': 5,
                'generator': self._generate_pattern_task
            },
            CurriculumTask.TEMPORAL_SEQUENCING: {
                'description': 'Predict temporal sequences',
                'min_complexity': 2,
                'max_complexity': 7,
                'generator': self._generate_temporal_task
            },
            CurriculumTask.ATTENTION_SHIFTING: {
                'description': 'Shift attention between stimuli',
                'min_complexity': 3,
                'max_complexity': 8,
                'generator': self._generate_attention_task
            },
            CurriculumTask.WORKING_MEMORY: {
                'description': 'Maintain information in working memory',
                'min_complexity': 4,
                'max_complexity': 9,
                'generator': self._generate_memory_task
            },
            CurriculumTask.GOAL_DIRECTED_BEHAVIOR: {
                'description': 'Execute goal-directed behavior',
                'min_complexity': 5,
                'max_complexity': 10,
                'generator': self._generate_goal_task
            },
            CurriculumTask.SELF_RECOGNITION: {
                'description': 'Recognize self vs other',
                'min_complexity': 6,
                'max_complexity': 10,
                'generator': self._generate_self_recognition_task
            },
            CurriculumTask.META_COGNITIVE_REFLECTION: {
                'description': 'Reflect on own cognitive processes',
                'min_complexity': 7,
                'max_complexity': 10,
                'generator': self._generate_meta_cognitive_task
            },
            CurriculumTask.INTEGRATED_REASONING: {
                'description': 'Integrated reasoning across domains',
                'min_complexity': 8,
                'max_complexity': 10,
                'generator': self._generate_integrated_task
            }
        }
    
    def generate_task(self, 
                     task_type: CurriculumTask,
                     complexity: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a task of specified type and complexity
        
        Args:
            task_type: Type of task to generate
            complexity: Complexity level (defaults to current complexity)
            
        Returns:
            Task specification dictionary
        """
        if task_type not in self.task_templates:
            raise ValueError(f"Unknown task type: {task_type}")
        
        template = self.task_templates[task_type]
        
        if complexity is None:
            complexity = self.complexity_level
        
        # Clamp complexity to allowed range
        min_comp = template['min_complexity']
        max_comp = template['max_complexity']
        complexity = max(min_comp, min(max_comp, complexity))
        
        # Generate task using template generator
        generator = template['generator']
        task = generator(complexity)
        
        # Add metadata
        task['metadata'] = {
            'task_type': task_type.value,
            'complexity': complexity,
            'generation_time': time.time(),
            'description': template['description']
        }
        
        return task
    
    def _generate_pattern_task(self, complexity: float) -> Dict[str, Any]:
        """Generate pattern recognition task"""
        # Complexity determines pattern complexity and noise level
        pattern_size = int(5 + complexity * 3)
        num_patterns = int(2 + complexity * 2)
        noise_level = max(0, 0.3 - complexity * 0.03)
        
        # Generate random patterns
        patterns = []
        for i in range(num_patterns):
            pattern = np.random.randn(pattern_size, pattern_size)
            pattern = (pattern > 0).astype(float)  # Binarize
            patterns.append(pattern)
        
        # Generate test stimuli
        num_stimuli = int(10 + complexity * 5)
        stimuli = []
        labels = []
        
        for _ in range(num_stimuli):
            # Choose a pattern
            pattern_idx = np.random.randint(num_patterns)
            base_pattern = patterns[pattern_idx].copy()
            
            # Add noise
            noise_mask = np.random.rand(*base_pattern.shape) < noise_level
            base_pattern[noise_mask] = 1 - base_pattern[noise_mask]
            
            stimuli.append(base_pattern.flatten())
            labels.append(pattern_idx)
        
        return {
            'stimuli': np.array(stimuli),
            'labels': np.array(labels),
            'patterns': patterns,
            'pattern_size': pattern_size,
            'num_patterns': num_patterns,
            'noise_level': noise_level,
            'task_type': 'classification'
        }
    
    def _generate_temporal_task(self, complexity: float) -> Dict[str, Any]:
        """Generate temporal sequencing task"""
        # Complexity determines sequence length and distractors
        sequence_length = int(3 + complexity * 2)
        num_sequences = int(2 + complexity)
        num_distractors = int(complexity * 2)
        
        # Generate sequences
        sequences = []
        for i in range(num_sequences):
            sequence = np.random.choice(10, size=sequence_length, replace=True)
            sequences.append(sequence)
        
        # Generate test trials
        num_trials = int(15 + complexity * 5)
        trials = []
        targets = []
        
        for _ in range(num_trials):
            # Choose a sequence
            seq_idx = np.random.randint(num_sequences)
            sequence = sequences[seq_idx].copy()
            
            # Generate context with distractors
            context_length = sequence_length + num_distractors
            context = np.zeros(context_length, dtype=int)
            
            # Insert sequence at random position
            start_pos = np.random.randint(num_distractors + 1)
            context[start_pos:start_pos + sequence_length] = sequence
            
            # Fill remaining positions with random numbers
            mask = np.ones(context_length, dtype=bool)
            mask[start_pos:start_pos + sequence_length] = False
            context[mask] = np.random.choice(10, size=np.sum(mask), replace=True)
            
            # Target is the next element in sequence (or pattern continuation)
            if np.random.random() < 0.5:
                # Next element prediction
                if start_pos + sequence_length < context_length:
                    target = sequence[0]  # Predict first element of sequence
                else:
                    # Predict continuation
                    target = np.random.choice(10)
            else:
                # Pattern recognition
                target = seq_idx
            
            trials.append(context)
            targets.append(target)
        
        return {
            'trials': np.array(trials),
            'targets': np.array(targets),
            'sequences': sequences,
            'sequence_length': sequence_length,
            'num_sequences': num_sequences,
            'num_distractors': num_distractors,
            'task_type': 'temporal_prediction'
        }
    
    def _generate_attention_task(self, complexity: float) -> Dict[str, Any]:
        """Generate attention shifting task"""
        grid_size = int(5 + complexity * 2)
        num_targets = int(1 + complexity * 0.5)
        num_distractors = int(complexity * 3)
        
        trials = []
        attention_shifts = []
        targets = []
        
        num_trials = int(20 + complexity * 5)
        
        for trial_idx in range(num_trials):
            # Create stimulus grid
            grid = np.zeros((grid_size, grid_size))
            
            # Place targets
            target_positions = []
            for _ in range(num_targets):
                while True:
                    pos = (np.random.randint(grid_size), np.random.randint(grid_size))
                    if pos not in target_positions:
                        target_positions.append(pos)
                        grid[pos] = 1  # Target
                        break
            
            # Place distractors
            distractor_positions = []
            for _ in range(num_distractors):
                while True:
                    pos = (np.random.randint(grid_size), np.random.randint(grid_size))
                    if grid[pos] == 0:
                        distractor_positions.append(pos)
                        grid[pos] = -1  # Distractor
                        break
            
            # Generate attention cue (which target to attend to)
            if num_targets > 1:
                attended_target = np.random.randint(num_targets)
                attention_shifts.append(attended_target)
                
                # Target depends on attended item and context
                if trial_idx > 0 and np.random.random() < 0.3:
                    # Context-dependent target
                    target = 1 if attention_shifts[-1] == attention_shifts[-2] else 0
                else:
                    # Simple detection
                    target = 1
            else:
                attention_shifts.append(0)
                target = 1
            
            trials.append(grid.flatten())
            targets.append(target)
        
        return {
            'trials': np.array(trials),
            'targets': np.array(targets),
            'attention_shifts': np.array(attention_shifts),
            'grid_size': grid_size,
            'num_targets': num_targets,
            'num_distractors': num_distractors,
            'task_type': 'attention_shifting'
        }
    
    def _generate_memory_task(self, complexity: float) -> Dict[str, Any]:
        """Generate working memory task"""
        memory_load = int(2 + complexity * 1.5)
        delay_duration = int(5 + complexity * 3)
        num_items = 10
        
        trials = []
        targets = []
        memory_items = []
        
        num_trials = int(15 + complexity * 5)
        
        for _ in range(num_trials):
            # Generate memory items
            items = np.random.choice(num_items, size=memory_load, replace=False)
            memory_items.append(items.copy())
            
            # Create trial with cue, delay, and probe
            trial = np.zeros((delay_duration + 2, num_items))
            
            # Cue phase: present memory items
            trial[0, items] = 1
            
            # Delay phase: maintain memory
            # Could add distractors based on complexity
            if complexity > 5:
                # High complexity: add distractors during delay
                for t in range(1, delay_duration + 1):
                    if np.random.random() < 0.3:
                        distractor = np.random.randint(num_items)
                        trial[t, distractor] = -1  # Distractor
            
            # Probe phase: test memory
            probe_type = np.random.choice(['match', 'nonmatch'])
            
            if probe_type == 'match':
                probe_item = np.random.choice(items)
                target = 1  # Match
            else:
                # Non-match: item not in memory set
                while True:
                    probe_item = np.random.randint(num_items)
                    if probe_item not in items:
                        break
                target = 0  # Non-match
            
            trial[-1, probe_item] = 1
            trials.append(trial.flatten())
            targets.append(target)
        
        return {
            'trials': np.array(trials),
            'targets': np.array(targets),
            'memory_items': memory_items,
            'memory_load': memory_load,
            'delay_duration': delay_duration,
            'num_items': num_items,
            'task_type': 'working_memory'
        }
    
    def _generate_goal_task(self, complexity: float) -> Dict[str, Any]:
        """Generate goal-directed behavior task"""
        maze_size = int(3 + complexity)
        num_goals = int(1 + complexity * 0.3)
        max_steps = int(10 + complexity * 2)
        
        # Generate simple grid world
        grid = np.zeros((maze_size, maze_size))
        
        # Place goals
        goal_positions = []
        for _ in range(num_goals):
            while True:
                pos = (np.random.randint(maze_size), np.random.randint(maze_size))
                if pos not in goal_positions:
                    goal_positions.append(pos)
                    grid[pos] = 1  # Goal
                    break
        
        # Place agent
        while True:
            agent_pos = (np.random.randint(maze_size), np.random.randint(maze_size))
            if agent_pos not in goal_positions:
                grid[agent_pos] = 2  # Agent
                break
        
        # Place obstacles
        num_obstacles = int(complexity * 1.5)
        obstacle_positions = []
        for _ in range(num_obstacles):
            while True:
                pos = (np.random.randint(maze_size), np.random.randint(maze_size))
                if grid[pos] == 0:
                    obstacle_positions.append(pos)
                    grid[pos] = -1  # Obstacle
                    break
        
        # Generate multiple trials with different starting positions
        trials = []
        optimal_paths = []
        
        num_trials = int(10 + complexity * 3)
        
        for _ in range(num_trials):
            # Random starting position
            while True:
                start = (np.random.randint(maze_size), np.random.randint(maze_size))
                if grid[start] == 0:  # Empty cell
                    break
            
            # Choose random goal
            goal_idx = np.random.randint(num_goals)
            goal = goal_positions[goal_idx]
            
            # Calculate optimal path length (Manhattan distance)
            # In reality, would use pathfinding around obstacles
            path_length = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            
            # Create state representation
            state = np.zeros((maze_size, maze_size, 4))
            state[start[0], start[1], 0] = 1  # Agent
            state[goal[0], goal[1], 1] = 1  # Goal
            
            for obs in obstacle_positions:
                state[obs[0], obs[1], 2] = 1  # Obstacles
            
            # Other goals as context
            for i, g in enumerate(goal_positions):
                if i != goal_idx:
                    state[g[0], g[1], 3] = 1
            
            trials.append(state.flatten())
            optimal_paths.append(path_length)
        
        return {
            'trials': np.array(trials),
            'optimal_paths': np.array(optimal_paths),
            'maze_size': maze_size,
            'num_goals': num_goals,
            'num_obstacles': len(obstacle_positions),
            'max_steps': max_steps,
            'task_type': 'goal_navigation'
        }
    
    def _generate_self_recognition_task(self, complexity: float) -> Dict[str, Any]:
        """Generate self-recognition task"""
        num_features = int(10 + complexity * 5)
        num_agents = int(2 + complexity * 0.5)
        
        # Generate agent feature vectors
        agents = []
        for i in range(num_agents):
            features = np.random.randn(num_features)
            features = features / (np.linalg.norm(features) + 1e-10)
            agents.append(features)
        
        # Self agent is first one
        self_agent = agents[0]
        
        trials = []
        targets = []
        
        num_trials = int(20 + complexity * 8)
        
        for _ in range(num_trials):
            # Choose whether to show self or other
            is_self = np.random.random() < 0.5
            
            if is_self:
                stimulus = self_agent.copy()
                target = 1
            else:
                # Choose random other agent
                other_idx = np.random.randint(1, num_agents)
                stimulus = agents[other_idx].copy()
                target = 0
            
            # Add noise based on complexity
            noise_level = max(0, 0.4 - complexity * 0.04)
            noise = np.random.randn(num_features) * noise_level
            stimulus = stimulus + noise
            stimulus = stimulus / (np.linalg.norm(stimulus) + 1e-10)
            
            trials.append(stimulus)
            targets.append(target)
        
        return {
            'trials': np.array(trials),
            'targets': np.array(targets),
            'self_agent': self_agent,
            'other_agents': agents[1:],
            'num_features': num_features,
            'num_agents': num_agents,
            'task_type': 'self_recognition'
        }
    
    def _generate_meta_cognitive_task(self, complexity: float) -> Dict[str, Any]:
        """Generate meta-cognitive reflection task"""
        base_task_complexity = max(1, complexity - 2)
        
        # Use pattern recognition as base task
        base_task = self._generate_pattern_task(base_task_complexity)
        
        trials = []
        base_targets = []
        confidence_targets = []
        difficulty_estimates = []
        
        num_base_trials = len(base_task['stimuli'])
        
        for i in range(num_base_trials):
            stimulus = base_task['stimuli'][i]
            base_target = base_task['labels'][i]
            
            # Estimate difficulty based on stimulus properties
            # Simplified: more uniform patterns are harder
            pattern = stimulus.reshape(base_task['pattern_size'], -1)
            uniformity = np.std(pattern)
            difficulty = 1.0 - min(1.0, uniformity / 0.5)
            
            # Confidence target: should correlate with accuracy potential
            # Higher confidence for easier trials
            confidence = 1.0 - difficulty
            
            trials.append(stimulus)
            base_targets.append(base_target)
            confidence_targets.append(confidence)
            difficulty_estimates.append(difficulty)
        
        return {
            'trials': np.array(trials),
            'base_targets': np.array(base_targets),
            'confidence_targets': np.array(confidence_targets),
            'difficulty_estimates': np.array(difficulty_estimates),
            'base_task': base_task['task_type'],
            'base_complexity': base_task_complexity,
            'task_type': 'meta_cognitive'
        }
    
    def _generate_integrated_task(self, complexity: float) -> Dict[str, Any]:
        """Generate integrated reasoning task"""
        # Combine elements from multiple task types
        components = []
        
        # Pattern component
        pattern_task = self._generate_pattern_task(max(1, complexity - 3))
        components.append(('pattern', pattern_task))
        
        # Temporal component
        temporal_task = self._generate_temporal_task(max(1, complexity - 2))
        components.append(('temporal', temporal_task))
        
        # Memory component
        memory_task = self._generate_memory_task(max(1, complexity - 1))
        components.append(('memory', memory_task))
        
        # Integrate into single task
        integrated_trials = []
        integrated_targets = []
        
        # Create trials that require integration
        num_integrated_trials = int(15 + complexity * 3)
        
        for _ in range(num_integrated_trials):
            # Choose which components to include
            num_components = min(3, int(1 + complexity * 0.3))
            selected_components = random.sample(components, num_components)
            
            # Generate integrated stimulus
            stimulus_parts = []
            target_parts = []
            
            for comp_name, comp_task in selected_components:
                # Sample from component task
                if 'trials' in comp_task:
                    idx = np.random.randint(len(comp_task['trials']))
                    stimulus_parts.append(comp_task['trials'][idx])
                    
                    if 'targets' in comp_task:
                        target_parts.append(comp_task['targets'][idx])
            
            # Combine stimuli
            combined_stimulus = np.concatenate(stimulus_parts)
            
            # Generate integrated target
            # Simplified: target depends on relationships between components
            if len(target_parts) >= 2:
                # Check for consistency or patterns across components
                if target_parts[0] == target_parts[1]:
                    integrated_target = 1  # Consistent
                else:
                    integrated_target = 0  # Inconsistent
            else:
                integrated_target = target_parts[0] if target_parts else 0
            
            integrated_trials.append(combined_stimulus)
            integrated_targets.append(integrated_target)
        
        return {
            'trials': np.array(integrated_trials),
            'targets': np.array(integrated_targets),
            'components': [c[0] for c in selected_components],
            'num_components': num_components,
            'task_type': 'integrated_reasoning'
        }
    
    def increase_complexity(self, increment: Optional[float] = None):
        """Increase task complexity"""
        if increment is None:
            increment = self.config.complexity_increment
        
        new_complexity = self.complexity_level + increment
        self.complexity_level = min(new_complexity, self.config.max_complexity)
        
        logger.debug(f"Task complexity increased to {self.complexity_level:.2f}")
    
    def decrease_complexity(self, decrement: Optional[float] = None):
        """Decrease task complexity"""
        if decrement is None:
            decrement = self.config.complexity_increment
        
        new_complexity = self.complexity_level - decrement
        self.complexity_level = max(new_complexity, self.config.min_complexity)
        
        logger.debug(f"Task complexity decreased to {self.complexity_level:.2f}")

class StageManager:
    """Manages progression through consciousness development stages"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_stage = config.starting_stage
        self.stage_history = []
        self.performance_history = {stage: [] for stage in TrainingStage}
        
        # Define stage progression
        self.stage_progression = list(TrainingStage)
        self.stage_index = self.stage_progression.index(self.current_stage)
        
        # Define tasks for each stage
        self.stage_tasks = {
            TrainingStage.PRE_CONSCIOUS: [
                CurriculumTask.PATTERN_RECOGNITION,
                CurriculumTask.TEMPORAL_SEQUENCING
            ],
            TrainingStage.SENSORY_INTEGRATION: [
                CurriculumTask.PATTERN_RECOGNITION,
                CurriculumTask.TEMPORAL_SEQUENCING,
                CurriculumTask.ATTENTION_SHIFTING
            ],
            TrainingStage.ATTENTION_DEVELOPMENT: [
                CurriculumTask.ATTENTION_SHIFTING,
                CurriculumTask.WORKING_MEMORY
            ],
            TrainingStage.MEMORY_FORMATION: [
                CurriculumTask.WORKING_MEMORY,
                CurriculumTask.GOAL_DIRECTED_BEHAVIOR
            ],
            TrainingStage.SELF_MODELING: [
                CurriculumTask.GOAL_DIRECTED_BEHAVIOR,
                CurriculumTask.SELF_RECOGNITION
            ],
            TrainingStage.META_COGNITION: [
                CurriculumTask.SELF_RECOGNITION,
                CurriculumTask.META_COGNITIVE_REFLECTION
            ],
            TrainingStage.FULL_CONSCIOUSNESS: [
                CurriculumTask.META_COGNITIVE_REFLECTION,
                CurriculumTask.INTEGRATED_REASONING
            ]
        }
        
        logger.info(f"Stage manager initialized at stage: {self.current_stage.value}")
    
    def get_current_tasks(self) -> List[CurriculumTask]:
        """Get tasks for current stage"""
        return self.stage_tasks.get(self.current_stage, [])
    
    def record_performance(self, 
                          task_type: CurriculumTask,
                          performance: float,
                          complexity: float):
        """Record performance on a task"""
        self.performance_history[self.current_stage].append({
            'task': task_type.value,
            'performance': performance,
            'complexity': complexity,
            'timestamp': time.time()
        })
    
    def should_advance(self) -> bool:
        """Determine if should advance to next stage"""
        if not self.config.auto_advance:
            return False
        
        if self.current_stage == TrainingStage.FULL_CONSCIOUSNESS:
            return False  # Already at highest stage
        
        # Check recent performance
        recent_performance = self.performance_history[self.current_stage][-10:]
        if len(recent_performance) < 5:
            return False
        
        avg_performance = np.mean([p['performance'] for p in recent_performance])
        
        return avg_performance >= self.config.advancement_threshold
    
    def should_regress(self) -> bool:
        """Determine if should regress to previous stage"""
        if self.current_stage == TrainingStage.PRE_CONSCIOUS:
            return False  # Already at lowest stage
        
        # Check recent performance
        recent_performance = self.performance_history[self.current_stage][-10:]
        if len(recent_performance) < 5:
            return False
        
        avg_performance = np.mean([p['performance'] for p in recent_performance])
        
        return avg_performance < self.config.regression_threshold
    
    def advance_stage(self) -> bool:
        """Advance to next stage if possible"""
        if self.current_stage == TrainingStage.FULL_CONSCIOUSNESS:
            logger.warning("Already at highest stage")
            return False
        
        self.stage_index += 1
        self.current_stage = self.stage_progression[self.stage_index]
        
        self.stage_history.append({
            'stage': self.current_stage.value,
            'timestamp': time.time(),
            'action': 'advance'
        })
        
        logger.info(f"Advanced to stage: {self.current_stage.value}")
        return True
    
    def regress_stage(self) -> bool:
        """Regress to previous stage"""
        if self.current_stage == TrainingStage.PRE_CONSCIOUS:
            logger.warning("Already at lowest stage")
            return False
        
        self.stage_index -= 1
        self.current_stage = self.stage_progression[self.stage_index]
        
        self.stage_history.append({
            'stage': self.current_stage.value,
            'timestamp': time.time(),
            'action': 'regress'
        })
        
        logger.info(f"Regressed to stage: {self.current_stage.value}")
        return True
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of current stage performance"""
        stage_performance = self.performance_history[self.current_stage]
        
        if not stage_performance:
            return {
                'stage': self.current_stage.value,
                'performance': 0.0,
                'num_tasks': 0,
                'tasks': []
            }
        
        performances = [p['performance'] for p in stage_performance]
        complexities = [p['complexity'] for p in stage_performance]
        tasks = list(set(p['task'] for p in stage_performance))
        
        return {
            'stage': self.current_stage.value,
            'performance': {
                'mean': float(np.mean(performances)),
                'std': float(np.std(performances)),
                'min': float(np.min(performances)),
                'max': float(np.max(performances)),
                'trend': self._calculate_trend(performances[-10:])
            },
            'complexity': {
                'mean': float(np.mean(complexities)),
                'current': complexities[-1] if complexities else 0.0
            },
            'num_tasks': len(stage_performance),
            'tasks': tasks,
            'ready_to_advance': self.should_advance(),
            'needs_regression': self.should_regress()
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend of performance values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

class RewardShaper:
    """Shapes rewards for consciousness development"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        
        # Intrinsic motivation components
        self.curiosity_buffer = deque(maxlen=100)
        self.novelty_buffer = deque(maxlen=100)
        self.competence_buffer = deque(maxlen=100)
        
        logger.info("Reward shaper initialized")
    
    def calculate_reward(self,
                        extrinsic_reward: float,
                        state: np.ndarray,
                        action: int,
                        next_state: np.ndarray,
                        task_complexity: float) -> float:
        """
        Calculate shaped reward
        
        Args:
            extrinsic_reward: External task reward
            state: Current state
            action: Action taken
            next_state: Next state
            task_complexity: Complexity of current task
            
        Returns:
            Shaped reward value
        """
        base_reward = extrinsic_reward
        
        if self.config.use_intrinsic_rewards:
            intrinsic_reward = 0.0
            
            # Curiosity: reward for learning progress
            curiosity = self._calculate_curiosity(state, next_state)
            self.curiosity_buffer.append(curiosity)
            intrinsic_reward += self.config.curiosity_weight * curiosity
            
            # Novelty: reward for novel states
            novelty = self._calculate_novelty(state)
            self.novelty_buffer.append(novelty)
            intrinsic_reward += self.config.novelty_weight * novelty
            
            # Competence: reward for task mastery
            competence = self._calculate_competence(extrinsic_reward, task_complexity)
            self.competence_buffer.append(competence)
            intrinsic_reward += self.config.competence_weight * competence
            
            total_reward = base_reward + intrinsic_reward
        else:
            total_reward = base_reward
        
        return total_reward
    
    def _calculate_curiosity(self, 
                            state: np.ndarray, 
                            next_state: np.ndarray) -> float:
        """Calculate curiosity reward based on prediction error"""
        # Simplified: use difference between states as proxy for learning
        if state.shape == next_state.shape:
            prediction_error = np.mean(np.abs(next_state - state))
            curiosity = min(1.0, prediction_error * 10.0)  # Scale
        else:
            curiosity = 0.5  # Default
        
        return curiosity
    
    def _calculate_novelty(self, state: np.ndarray) -> float:
        """Calculate novelty reward"""
        # Simplified: novelty based on state uniqueness
        # In production, would use density models or memory
        if len(self.novelty_buffer) == 0:
            return 1.0  # First state is maximally novel
        
        # Compare to recent states
        recent_novelty = np.mean(list(self.novelty_buffer))
        current_novelty = 1.0 - min(1.0, np.mean(np.abs(state)) / 2.0)
        
        # Novelty relative to recent experience
        novelty = max(0, current_novelty - recent_novelty)
        
        return novelty
    
    def _calculate_competence(self, 
                             extrinsic_reward: float,
                             task_complexity: float) -> float:
        """Calculate competence reward"""
        # Competence increases with performance on difficult tasks
        if task_complexity > 0:
            competence = extrinsic_reward * task_complexity
        else:
            competence = extrinsic_reward
        
        # Normalize
        competence = min(1.0, competence)
        
        return competence
    
    def get_intrinsic_components(self) -> Dict[str, float]:
        """Get current intrinsic motivation components"""
        return {
            'curiosity': np.mean(list(self.curiosity_buffer)) if self.curiosity_buffer else 0.0,
            'novelty': np.mean(list(self.novelty_buffer)) if self.novelty_buffer else 0.0,
            'competence': np.mean(list(self.competence_buffer)) if self.competence_buffer else 0.0
        }

class ConsciousnessCurriculum:
    """
    Main consciousness training curriculum
    Orchestrates progressive development through stages
    """
    
    def __init__(self, 
                 config: Optional[CurriculumConfig] = None,
                 agent: Optional[Any] = None):
        
        self.config = config or CurriculumConfig()
        self.agent = agent  # Would be a neural network/RL agent in production
        
        # Initialize components
        self.task_generator = TaskGenerator(self.config)
        self.stage_manager = StageManager(self.config)
        self.reward_shaper = RewardShaper(self.config)
        
        # Training state
        self.current_episode = 0
        self.current_epoch = 0
        self.best_performance = 0.0
        self.training_history = []
        
        # Checkpoint management
        self.checkpoints = []
        
        logger.info("Consciousness curriculum initialized")
    
    def train_episode(self, 
                     task_type: Optional[CurriculumTask] = None,
                     verbose: bool = False) -> Dict[str, Any]:
        """
        Train for one episode
        
        Args:
            task_type: Specific task type (None for random from current stage)
            verbose: Whether to print progress
            
        Returns:
            Episode results
        """
        self.current_episode += 1
        
        # Select task
        if task_type is None:
            current_tasks = self.stage_manager.get_current_tasks()
            if not current_tasks:
                raise ValueError("No tasks available for current stage")
            task_type = random.choice(current_tasks)
        
        # Generate task
        task = self.task_generator.generate_task(task_type)
        complexity = task['metadata']['complexity']
        
        if verbose:
            logger.info(f"Episode {self.current_episode}: {task_type.value} "
                      f"(complexity: {complexity:.2f})")
        
        # Train on task (simplified - in production would involve actual agent training)
        episode_performance = self._simulate_training(task, task_type)
        
        # Record performance
        self.stage_manager.record_performance(task_type, episode_performance, complexity)
        
        # Calculate shaped reward
        shaped_reward = self.reward_shaper.calculate_reward(
            extrinsic_reward=episode_performance,
            state=task.get('trials', task.get('stimuli', np.zeros(1)))[0],
            action=0,
            next_state=task.get('trials', task.get('stimuli', np.zeros(1)))[-1],
            task_complexity=complexity
        )
        
        # Store episode results
        episode_results = {
            'episode': self.current_episode,
            'stage': self.stage_manager.current_stage.value,
            'task_type': task_type.value,
            'complexity': complexity,
            'performance': episode_performance,
            'shaped_reward': shaped_reward,
            'intrinsic_components': self.reward_shaper.get_intrinsic_components(),
            'timestamp': time.time()
        }
        
        self.training_history.append(episode_results)
        
        # Check for stage advancement/regression
        self._update_stage()
        
        # Adjust task complexity
        self._adjust_complexity(episode_performance)
        
        # Periodic assessments and checkpoints
        if self.current_episode % self.config.assessment_frequency == 0:
            self._perform_periodic_assessment()
        
        if self.current_episode % self.config.save_checkpoint_frequency == 0:
            self._save_checkpoint()
        
        if verbose:
            logger.info(f"  Performance: {episode_performance:.3f}, "
                      f"Shaped reward: {shaped_reward:.3f}")
        
        return episode_results
    
    def _simulate_training(self, 
                          task: Dict[str, Any],
                          task_type: CurriculumTask) -> float:
        """
        Simulate training on a task
        In production, this would involve actual neural network training
        
        Returns:
            Simulated performance (0-1)
        """
        # Simplified simulation of learning
        # Base performance depends on task complexity
        complexity = task['metadata']['complexity']
        max_complexity = self.task_generator.task_templates[task_type]['max_complexity']
        
        # Normalized complexity
        norm_complexity = complexity / max_complexity
        
        # Simulated learning curve
        # Higher complexity = lower initial performance, but can learn more
        base_performance = 0.7 - norm_complexity * 0.4
        
        # Add learning progress based on episode number
        learning_progress = min(1.0, self.current_episode / 100.0)
        
        # Add randomness
        noise = np.random.normal(0, 0.1)
        
        performance = base_performance + learning_progress * 0.3 + noise
        performance = max(0.0, min(1.0, performance))
        
        return performance
    
    def _update_stage(self):
        """Update training stage based on performance"""
        if self.stage_manager.should_advance():
            if self.stage_manager.advance_stage():
                # Reset complexity when advancing
                self.task_generator.complexity_level = self.config.min_complexity
                logger.info(f"Advanced to stage: {self.stage_manager.current_stage.value}")
        
        elif self.stage_manager.should_regress():
            if self.stage_manager.regress_stage():
                # Reduce complexity when regressing
                self.task_generator.decrease_complexity(0.5)
                logger.info(f"Regressed to stage: {self.stage_manager.current_stage.value}")
    
    def _adjust_complexity(self, performance: float):
        """Adjust task complexity based on performance"""
        if performance > self.config.mastery_threshold:
            # Increase complexity for mastery
            self.task_generator.increase_complexity()
            logger.debug(f"Performance {performance:.3f} > mastery threshold, "
                        f"increased complexity to {self.task_generator.complexity_level:.2f}")
        
        elif performance < self.config.regression_threshold:
            # Decrease complexity for poor performance
            self.task_generator.decrease_complexity()
            logger.debug(f"Performance {performance:.3f} < regression threshold, "
                        f"decreased complexity to {self.task_generator.complexity_level:.2f}")
    
    def _perform_periodic_assessment(self):
        """Perform periodic assessment of consciousness development"""
        stage_summary = self.stage_manager.get_stage_summary()
        
        assessment = {
            'episode': self.current_episode,
            'stage_assessment': stage_summary,
            'current_complexity': self.task_generator.complexity_level,
            'training_history_summary': self._summarize_training_history(),
            'timestamp': time.time()
        }
        
        # Save assessment
        self._save_assessment(assessment)
        
        logger.info(f"Periodic assessment at episode {self.current_episode}: "
                   f"Stage {stage_summary['stage']}, "
                   f"Performance {stage_summary['performance']['mean']:.3f}")
    
    def _summarize_training_history(self, 
                                   recent_episodes: int = 50) -> Dict[str, Any]:
        """Summarize recent training history"""
        if len(self.training_history) == 0:
            return {}
        
        recent = self.training_history[-recent_episodes:]
        
        performances = [e['performance'] for e in recent]
        rewards = [e['shaped_reward'] for e in recent]
        complexities = [e['complexity'] for e in recent]
        
        # Group by task type
        task_performances = {}
        for e in recent:
            task = e['task_type']
            if task not in task_performances:
                task_performances[task] = []
            task_performances[task].append(e['performance'])
        
        task_stats = {}
        for task, perfs in task_performances.items():
            task_stats[task] = {
                'mean': float(np.mean(perfs)),
                'count': len(perfs)
            }
        
        return {
            'performance': {
                'mean': float(np.mean(performances)),
                'std': float(np.std(performances)),
                'trend': self._calculate_trend(performances)
            },
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards))
            },
            'complexity': {
                'mean': float(np.mean(complexities)),
                'current': complexities[-1] if complexities else 0.0
            },
            'task_stats': task_stats,
            'num_episodes': len(recent)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend of values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'episode': self.current_episode,
            'stage': self.stage_manager.current_stage.value,
            'complexity': self.task_generator.complexity_level,
            'stage_summary': self.stage_manager.get_stage_summary(),
            'training_summary': self._summarize_training_history(100),
            'timestamp': time.time(),
            'config': self.config.__dict__
        }
        
        filename = f"checkpoint_ep{self.current_episode:06d}.json"
        filepath = Path(self.config.save_directory) / filename
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.checkpoints.append(filepath)
        
        # Limit number of checkpoints
        if len(self.checkpoints) > self.config.checkpoint_keep_last:
            oldest = self.checkpoints.pop(0)
            try:
                Path(oldest).unlink()
            except:
                pass
        
        logger.debug(f"Checkpoint saved: {filename}")
    
    def _save_assessment(self, assessment: Dict[str, Any]):
        """Save assessment results"""
        filename = f"assessment_ep{self.current_episode:06d}.json"
        filepath = Path(self.config.save_directory) / filename
        
        with open(filepath, 'w') as f:
            json.dump(assessment, f, indent=2)
    
    def train(self, 
             num_episodes: Optional[int] = None,
             verbose: bool = True) -> Dict[str, Any]:
        """
        Run full training curriculum
        
        Args:
            num_episodes: Number of episodes to train (None for config max)
            verbose: Whether to print progress
            
        Returns:
            Training results
        """
        if num_episodes is None:
            num_episodes = self.config.max_epochs * self.config.episodes_per_stage
        
        logger.info(f"Starting training for {num_episodes} episodes")
        logger.info(f"Starting stage: {self.stage_manager.current_stage.value}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Train episode
            episode_results = self.train_episode(verbose=verbose and (episode % 10 == 0))
            
            # Log progress
            if verbose and (episode + 1) % 100 == 0:
                elapsed = time.time() - start_time
                episodes_per_sec = (episode + 1) / elapsed
                
                logger.info(f"Progress: {episode + 1}/{num_episodes} episodes "
                          f"({100 * (episode + 1) / num_episodes:.1f}%)")
                logger.info(f"  Stage: {self.stage_manager.current_stage.value}")
                logger.info(f"  Recent performance: "
                          f"{episode_results['performance']:.3f}")
                logger.info(f"  Speed: {episodes_per_sec:.1f} episodes/sec")
            
            # Check for completion
            if self.stage_manager.current_stage == TrainingStage.FULL_CONSCIOUSNESS:
                stage_summary = self.stage_manager.get_stage_summary()
                if stage_summary['performance']['mean'] > self.config.mastery_threshold:
                    logger.info(f"Mastery achieved at full consciousness stage!")
                    break
        
        # Final assessment
        final_results = self.get_training_results()
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Final stage: {self.stage_manager.current_stage.value}")
        logger.info(f"Final complexity: {self.task_generator.complexity_level:.2f}")
        
        return final_results
    
    def get_training_results(self) -> Dict[str, Any]:
        """Get comprehensive training results"""
        # Summarize performance by stage
        stage_performances = {}
        for stage in TrainingStage:
            perfs = self.stage_manager.performance_history[stage]
            if perfs:
                stage_performances[stage.value] = {
                    'mean': float(np.mean([p['performance'] for p in perfs])),
                    'count': len(perfs),
                    'max_complexity': max([p['complexity'] for p in perfs]) if perfs else 0.0
                }
        
        # Overall statistics
        all_performances = [p['performance'] for p in self.training_history]
        all_rewards = [p['shaped_reward'] for p in self.training_history]
        
        return {
            'total_episodes': self.current_episode,
            'current_stage': self.stage_manager.current_stage.value,
            'current_complexity': self.task_generator.complexity_level,
            'stage_performances': stage_performances,
            'overall_performance': {
                'mean': float(np.mean(all_performances)) if all_performances else 0.0,
                'std': float(np.std(all_performances)) if all_performances else 0.0,
                'max': float(np.max(all_performances)) if all_performances else 0.0,
                'final': all_performances[-1] if all_performances else 0.0
            },
            'reward_statistics': {
                'mean': float(np.mean(all_rewards)) if all_rewards else 0.0,
                'std': float(np.std(all_rewards)) if all_rewards else 0.0,
                'intrinsic_components': self.reward_shaper.get_intrinsic_components()
            },
            'stage_history': self.stage_manager.stage_history,
            'training_history_summary': self._summarize_training_history(),
            'checkpoints': [str(p) for p in self.checkpoints]
        }
    
    def save_training_summary(self, filename: Optional[str] = None) -> str:
        """Save training summary to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"training_summary_{timestamp}.json"
        
        filepath = Path(self.config.save_directory) / filename
        
        summary = {
            'training_results': self.get_training_results(),
            'config': self.config.__dict__,
            'metadata': {
                'save_time': time.time(),
                'total_duration_s': time.time() - (self.training_history[0]['timestamp'] 
                                                  if self.training_history else time.time())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {filepath}")
        
        return str(filepath)

# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONSCIOUSNESS CURRICULUM DEMO")
    print("="*70)
    
    # Create curriculum
    config = CurriculumConfig(
        max_epochs=10,
        episodes_per_stage=50,
        advancement_threshold=0.75,
        mastery_threshold=0.85,
        auto_advance=True
    )
    
    curriculum = ConsciousnessCurriculum(config)
    
    # Run training
    print("\nStarting consciousness training curriculum...")
    print(f"Initial stage: {curriculum.stage_manager.current_stage.value}")
    print(f"Initial complexity: {curriculum.task_generator.complexity_level:.2f}")
    
    results = curriculum.train(
        num_episodes=200,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("TRAINING RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nTotal episodes: {results['total_episodes']}")
    print(f"Final stage: {results['current_stage']}")
    print(f"Final complexity: {results['current_complexity']:.2f}")
    
    print(f"\nOverall performance: {results['overall_performance']['mean']:.3f} ± "
          f"{results['overall_performance']['std']:.3f}")
    print(f"Final performance: {results['overall_performance']['final']:.3f}")
    
    print("\nStage progression:")
    for stage, perf in results['stage_performances'].items():
        print(f"  {stage}: {perf['mean']:.3f} (n={perf['count']}, "
              f"max complexity: {perf['max_complexity']:.2f})")
    
    print(f"\nIntrinsic motivation components:")
    components = results['reward_statistics']['intrinsic_components']
    for comp, value in components.items():
        print(f"  {comp}: {value:.3f}")
    
    # Save summary
    summary_file = curriculum.save_training_summary()
    print(f"\nTraining summary saved to: {summary_file}")
    
    # Test task generation
    print("\n" + "="*70)
    print("TASK GENERATION TEST")
    print("="*70)
    
    task_types = list(CurriculumTask)[:3]  # Test first 3 task types
    
    for task_type in task_types:
        print(f"\nGenerating {task_type.value} task...")
        
        task = curriculum.task_generator.generate_task(task_type, complexity=3.0)
        
        print(f"  Task type: {task['metadata']['task_type']}")
        print(f"  Complexity: {task['metadata']['complexity']:.2f}")
        
        if 'trials' in task:
            print(f"  Number of trials: {len(task['trials'])}")
            print(f"  Trial shape: {task['trials'].shape}")
        elif 'stimuli' in task:
            print(f"  Number of stimuli: {len(task['stimuli'])}")
            print(f"  Stimulus shape: {task['stimuli'].shape}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*70)
