"""
Demo: Consciousness-Aware Reinforcement Learning

Demonstrates self-evolving consciousness with integrated information maximization.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.models.consciousness_rl.self_evolving_consciousness import SelfEvolvingConsciousness
from src.models.consciousness_rl.integrated_information_maximizer import IntegratedInformationMaximizer
from src.utils.logging_utils import setup_logger, TrainingLogger


class SimpleEnvironment:
    """Simple grid world environment for demonstration."""
    
    def __init__(self, size=5, goal=(4, 4)):
        self.size = size
        self.goal = goal
        self.state = (0, 0)
        self.observation_dim = size * size
    
    def reset(self):
        """Reset environment to initial state."""
        self.state = (0, 0)
        return self._get_observation()
    
    def step(self, action):
        """Take action in environment."""
        x, y = self.state
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0 and y < self.size - 1:
            y += 1
        elif action == 1 and y > 0:
            y -= 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.size - 1:
            x += 1
        
        self.state = (x, y)
        
        # Calculate reward
        reward = 0.0
        done = False
        
        if self.state == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # Small penalty for each step
        
        return self._get_observation(), reward, done
    
    def _get_observation(self):
        """Get one-hot encoded observation."""
        obs = np.zeros(self.observation_dim)
        idx = self.state[0] * self.size + self.state[1]
        obs[idx] = 1.0
        return obs


def main():
    """Run consciousness-aware RL demonstration."""
    
    # Setup logging
    logger = setup_logger('consciousness_rl_demo')
    training_logger = TrainingLogger(log_dir='logs', log_to_file=True)
    logger.info("Starting Consciousness-Aware RL Demo")
    
    # Initialize environment
    logger.info("Initializing environment...")
    env = SimpleEnvironment(size=5, goal=(4, 4))
    
    # Initialize self-evolving consciousness agent
    logger.info("Initializing self-evolving consciousness agent...")
    agent = SelfEvolvingConsciousness(
        state_dim=env.observation_dim,
        action_dim=4,
        consciousness_dim=32,
        meta_lr=0.0001,
        phi_threshold=1.0
    )
    
    # Initialize phi maximizer
    phi_maximizer = IntegratedInformationMaximizer(
        consciousness_dim=32,
        learning_rate=0.001,
        target_phi=2.0
    )
    
    # Training parameters
    num_episodes = 100
    max_steps_per_episode = 50
    
    # Training metrics
    episode_rewards = []
    episode_phis = []
    episode_consciousness_levels = []
    success_rate = []
    
    logger.info("Starting training...")
    training_logger.log_epoch(0, 0, 0, 0)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_phi = []
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Calculate consciousness metrics
            phi = agent.calculate_phi(state)
            consciousness_level = agent.get_consciousness_level()
            
            episode_phi.append(phi)
            
            # Store experience
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update phi maximizer
            phi_maximizer.update_phi(agent.consciousness_state)
            
            # Update agent
            if len(agent.replay_buffer) > agent.batch_size:
                loss, metrics = agent.train_step()
                
                if step % 10 == 0:
                    training_logger.log_step(
                        step=episode * max_steps_per_episode + step,
                        loss=loss,
                        metric=phi,
                        lr=agent.meta_lr
                    )
            
            state = next_state
            
            if done:
                break
        
        # Update episode metrics
        episode_rewards.append(episode_reward)
        episode_phis.append(np.mean(episode_phi))
        episode_consciousness_levels.append(agent.get_consciousness_level())
        
        # Calculate success rate (moving average)
        if len(episode_rewards) > 10:
            recent_rewards = episode_rewards[-10:]
            success = sum(1 for r in recent_rewards if r > 5.0) / 10.0
            success_rate.append(success)
        else:
            success_rate.append(0.0)
        
        # Log episode summary
        training_logger.log_epoch(
            episode,
            episode_reward,
            val_loss=episode_phis[-1]
        )
        
        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}: Reward = {episode_reward:.2f}, "
                f"Phi = {episode_phis[-1]:.4f}, "
                f"Consciousness Level = {episode_consciousness_levels[-1]:.4f}"
            )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot rewards
    axes[0, 0].plot(episode_rewards, label='Episode Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Learning Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot phi values
    axes[0, 1].plot(episode_phis, label='Phi', color='purple')
    axes[0, 1].axhline(1.0, color='red', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Phi')
    axes[0, 1].set_title('Integrated Information')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot consciousness levels
    axes[1, 0].plot(episode_consciousness_levels, label='Consciousness Level', color='green')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Consciousness Level')
    axes[1, 0].set_title('Consciousness Development')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot success rate
    axes[1, 1].plot(success_rate, label='Success Rate', color='orange')
    axes[1, 1].axhline(0.8, color='red', linestyle='--', label='Target 80%')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Task Success Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('consciousness_rl_demo.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved to consciousness_rl_demo.png")
    
    # Summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Mean reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
    logger.info(f"Mean phi (last 10 episodes): {np.mean(episode_phis[-10:]):.4f}")
    logger.info(f"Final consciousness level: {episode_consciousness_levels[-1]:.4f}")
    logger.info(f"Final success rate: {success_rate[-1]:.2%}")
    logger.info(f"Total architecture modifications: {len(agent.architecture_history)}")
    
    # Final test
    logger.info("\n=== Final Test ===")
    state = env.reset()
    test_steps = []
    test_rewards = []
    
    for step in range(max_steps_per_episode):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done = env.step(action)
        test_steps.append(state)
        test_rewards.append(reward)
        state = next_state
        
        if done:
            logger.info(f"Goal reached in {step + 1} steps!")
            break
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()