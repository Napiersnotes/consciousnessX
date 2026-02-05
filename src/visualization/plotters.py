"""
Visualization plotters for consciousness simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Tuple
import torch


class QuantumStatePlotter:
    """
    Plotter for visualizing quantum states and dynamics.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize QuantumStatePlotter.

        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize

    def plot_quantum_state(self, quantum_state: np.ndarray, title: str = "Quantum State"):
        """
        Plot quantum state amplitudes.

        Args:
            quantum_state: Complex quantum state vector
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plot probability amplitudes
        probabilities = np.abs(quantum_state) ** 2
        ax1.bar(range(len(probabilities)), probabilities)
        ax1.set_xlabel("Basis State")
        ax1.set_ylabel("Probability")
        ax1.set_title("Probability Amplitudes")

        # Plot phase angles
        phases = np.angle(quantum_state)
        ax2.plot(phases, "o-")
        ax2.set_xlabel("Basis State")
        ax2.set_ylabel("Phase (rad)")
        ax2.set_title("Phase Angles")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_bloch_sphere(
        self, quantum_state: np.ndarray, title: str = "Bloch Sphere Representation"
    ):
        """
        Plot quantum state on Bloch sphere (for single qubit).

        Args:
            quantum_state: 2-element complex quantum state vector
            title: Plot title
        """
        if len(quantum_state) != 2:
            raise ValueError("Bloch sphere requires 2-element state vector")

        # Compute Bloch sphere coordinates
        alpha, beta = quantum_state[0], quantum_state[1]

        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta) - np.angle(alpha)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Create 3D plot
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[0, x],
                    y=[0, y],
                    z=[0, z],
                    mode="lines+markers",
                    line=dict(color="red", width=5),
                    marker=dict(size=8),
                )
            ]
        )

        # Add sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.1, showscale=False))

        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            width=700,
            height=700,
        )

        fig.show()

    def plot_quantum_dynamics(
        self, quantum_states: List[np.ndarray], title: str = "Quantum State Dynamics"
    ):
        """
        Plot evolution of quantum state over time.

        Args:
            quantum_states: List of quantum state vectors
            title: Plot title
        """
        n_states = len(quantum_states)
        n_basis = len(quantum_states[0])

        # Compute probabilities over time
        probabilities = np.array([np.abs(state) ** 2 for state in quantum_states])

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=probabilities.T,
                colorscale="Viridis",
                x=list(range(n_states)),
                y=list(range(n_basis)),
                colorbar=dict(title="Probability"),
            )
        )

        fig.update_layout(title=title, xaxis_title="Time Step", yaxis_title="Basis State")

        fig.show()

    def plot_coherence_evolution(
        self, coherence_history: List[float], title: str = "Quantum Coherence Evolution"
    ):
        """
        Plot evolution of quantum coherence over time.

        Args:
            coherence_history: List of coherence values
            title: Plot title
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(coherence_history))),
                y=coherence_history,
                mode="lines",
                name="Coherence",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title=title, xaxis_title="Time Step", yaxis_title="Coherence", yaxis=dict(range=[0, 1])
        )

        fig.show()


class MicrotubuleVisualizer:
    """
    Visualizer for microtubule structures and dynamics.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize MicrotubuleVisualizer.

        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize

    def visualize_microtubule(
        self,
        positions: np.ndarray,
        quantum_states: Optional[np.ndarray] = None,
        title: str = "Microtubule Structure",
    ):
        """
        Visualize microtubule structure.

        Args:
            positions: Array of tubulin positions (N, 3)
            quantum_states: Optional quantum states for color coding
            title: Plot title
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Determine colors based on quantum states
        if quantum_states is not None:
            colors = np.abs(quantum_states)
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2], c=colors, cmap="viridis", s=50
            )
            plt.colorbar(scatter, label="Quantum State Amplitude")
        else:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="blue", s=50)

        # Draw connections between tubulins
        for i in range(len(positions) - 1):
            ax.plot(
                [positions[i, 0], positions[i + 1, 0]],
                [positions[i, 1], positions[i + 1, 1]],
                [positions[i, 2], positions[i + 1, 2]],
                "gray",
                alpha=0.3,
            )

        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        ax.set_zlabel("Z (nm)")
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def visualize_microtubule_dynamics(
        self,
        positions: np.ndarray,
        quantum_states_history: List[np.ndarray],
        title: str = "Microtubule Dynamics",
    ):
        """
        Create animation of microtubule dynamics.

        Args:
            positions: Array of tubulin positions (N, 3)
            quantum_states_history: List of quantum state vectors over time
            title: Animation title
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        def update(frame):
            ax.clear()

            quantum_states = quantum_states_history[frame]
            colors = np.abs(quantum_states)

            ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2], c=colors, cmap="viridis", s=50
            )

            for i in range(len(positions) - 1):
                ax.plot(
                    [positions[i, 0], positions[i + 1, 0]],
                    [positions[i, 1], positions[i + 1, 1]],
                    [positions[i, 2], positions[i + 1, 2]],
                    "gray",
                    alpha=0.3,
                )

            ax.set_xlabel("X (nm)")
            ax.set_ylabel("Y (nm)")
            ax.set_zlabel("Z (nm)")
            ax.set_title(f"{title} - Frame {frame}")

        anim = FuncAnimation(
            fig, update, frames=len(quantum_states_history), interval=100, blit=False
        )

        plt.show()
        return anim

    def plot_tubulin_states(self, tubulin_states: np.ndarray, title: str = "Tubulin States"):
        """
        Plot tubulin states over time.

        Args:
            tubulin_states: Array of tubulin states (time, num_tubulins)
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        cax = ax.imshow(tubulin_states.T, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Tubulin Index")
        ax.set_title(title)

        plt.colorbar(cax, label="State")
        plt.tight_layout()
        plt.show()
