"""
Quantum circuit visualization utilities for ConsciousnessX.

Visualize quantum states, circuits, and measurements.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as patches


class QuantumVisualizer:
    """
    Visualize quantum circuits and states.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize quantum visualizer.

        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.figures = {}

    def plot_quantum_state(
        self, state: np.ndarray, title: str = "Quantum State", save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot quantum state amplitudes.

        Args:
            state: Quantum state vector
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        n_qubits = int(np.log2(len(state)))
        indices = np.arange(len(state))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plot amplitudes
        amplitudes = np.real(state)
        ax1.bar(indices, amplitudes, color="blue", alpha=0.7)
        ax1.set_xlabel("Basis State")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Real Amplitudes")
        ax1.set_xticks(indices)
        ax1.set_xticklabels([format(i, f"0{n_qubits}b") for i in indices], rotation=90)

        # Plot probabilities
        probabilities = np.abs(state) ** 2
        ax2.bar(indices, probabilities, color="green", alpha=0.7)
        ax2.set_xlabel("Basis State")
        ax2.set_ylabel("Probability")
        ax2.set_title("Measurement Probabilities")
        ax2.set_xticks(indices)
        ax2.set_xticklabels([format(i, f"0{n_qubits}b") for i in indices], rotation=90)

        fig.suptitle(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        self.figures["quantum_state"] = fig
        return fig

    def plot_bloch_sphere(
        self, state: np.ndarray, title: str = "Bloch Sphere", save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot single-qubit state on Bloch sphere.

        Args:
            state: Single-qubit state vector (length 2)
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        if len(state) != 2:
            raise ValueError("State must be a single qubit (length 2)")

        # Calculate Bloch sphere coordinates
        alpha, beta = state[0], state[1]

        # Compute Bloch sphere angles
        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta) - np.angle(alpha)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(sphere_x, sphere_y, sphere_z, color="lightblue", alpha=0.3)

        # Draw axes
        ax.plot([0, 1], [0, 0], [0, 0], "k-", linewidth=1)
        ax.plot([0, 0], [0, 1], [0, 0], "k-", linewidth=1)
        ax.plot([0, 0], [0, 0], [0, 1], "k-", linewidth=1)

        # Draw state vector
        ax.quiver(0, 0, 0, x, y, z, color="red", linewidth=3, arrow_length_ratio=0.1)

        # Label axes
        ax.text(1.1, 0, 0, "|x⟩", fontsize=12)
        ax.text(0, 1.1, 0, "|y⟩", fontsize=12)
        ax.text(0, 0, 1.1, "|0⟩", fontsize=12)
        ax.text(0, 0, -1.1, "|1⟩", fontsize=12)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        ax.set_box_aspect([1, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        self.figures["bloch_sphere"] = fig
        return fig

    def plot_circuit_diagram(
        self,
        gates: List[Tuple[str, int, Optional[List[int]]]],
        n_qubits: int,
        title: str = "Quantum Circuit",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot quantum circuit diagram.

        Args:
            gates: List of (gate_type, qubit, control_qubits)
            n_qubits: Number of qubits
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw qubit lines
        for q in range(n_qubits):
            ax.axhline(y=-q, color="black", linewidth=2)
            ax.text(-0.5, -q, f"q{q}", ha="right", va="center", fontsize=12)

        # Draw gates
        for i, (gate_type, qubit, controls) in enumerate(gates):
            x = i + 1

            if gate_type == "H":
                rect = Rectangle(
                    (x - 0.4, -qubit - 0.4),
                    0.8,
                    0.8,
                    facecolor="lightblue",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(x, -qubit, "H", ha="center", va="center", fontsize=12)

            elif gate_type == "X":
                rect = Rectangle(
                    (x - 0.4, -qubit - 0.4),
                    0.8,
                    0.8,
                    facecolor="lightgreen",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(x, -qubit, "X", ha="center", va="center", fontsize=12)

            elif gate_type == "Y":
                rect = Rectangle(
                    (x - 0.4, -qubit - 0.4),
                    0.8,
                    0.8,
                    facecolor="lightgreen",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(x, -qubit, "Y", ha="center", va="center", fontsize=12)

            elif gate_type == "Z":
                rect = Rectangle(
                    (x - 0.4, -qubit - 0.4),
                    0.8,
                    0.8,
                    facecolor="lightgreen",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(x, -qubit, "Z", ha="center", va="center", fontsize=12)

            elif gate_type == "CNOT":
                if controls:
                    for c in controls:
                        ax.plot([x, x], [-c, -qubit], "k-", linewidth=2)
                        circle = patches.Circle((x, -c), 0.15, facecolor="black", edgecolor="black")
                        ax.add_patch(circle)

                    rect = Rectangle(
                        (x - 0.4, -qubit - 0.4),
                        0.8,
                        0.8,
                        facecolor="lightgreen",
                        edgecolor="black",
                        linewidth=2,
                    )
                    ax.add_patch(rect)
                    ax.text(x, -qubit, "X", ha="center", va="center", fontsize=12)

            elif gate_type == "Measure":
                box = Rectangle(
                    (x - 0.4, -qubit - 0.4),
                    0.8,
                    0.8,
                    facecolor="lightcoral",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(box)
                ax.plot([x - 0.2, x + 0.2], [-qubit - 0.2, -qubit + 0.2], "k-", linewidth=2)
                ax.plot([x - 0.2, x + 0.2], [-qubit + 0.2, -qubit - 0.2], "k-", linewidth=2)

        ax.set_xlim(-1, len(gates) + 2)
        ax.set_ylim(-n_qubits + 0.5, 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        self.figures["circuit"] = fig
        return fig

    def plot_decoherence(
        self,
        t1_times: np.ndarray,
        t2_times: np.ndarray,
        time_points: np.ndarray,
        title: str = "Decoherence Times",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot T1 and T2 decoherence times.

        Args:
            t1_times: T1 relaxation times
            t2_times: T2 dephasing times
            time_points: Time points
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(time_points, t1_times, label="T1 (Relaxation)", linewidth=2, color="blue")
        ax.plot(time_points, t2_times, label="T2 (Dephasing)", linewidth=2, color="red")

        ax.set_xlabel("Time")
        ax.set_ylabel("Decoherence Time (μs)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        self.figures["decoherence"] = fig
        return fig

    def plot_fidelity_over_time(
        self,
        fidelities: np.ndarray,
        time_points: np.ndarray,
        target_fidelity: Optional[float] = None,
        title: str = "Gate Fidelity",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot gate fidelity over time.

        Args:
            fidelities: Fidelity values
            time_points: Time points
            target_fidelity: Optional target fidelity threshold
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(time_points, fidelities, linewidth=2, color="purple", label="Fidelity")

        if target_fidelity is not None:
            ax.axhline(
                target_fidelity, color="red", linestyle="--", label=f"Target ({target_fidelity})"
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Fidelity")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        self.figures["fidelity"] = fig
        return fig

    def plot_error_correction(
        self,
        logical_errors: np.ndarray,
        physical_errors: np.ndarray,
        time_points: np.ndarray,
        title: str = "Error Correction",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot error correction performance.

        Args:
            logical_errors: Logical error rates
            physical_errors: Physical error rates
            time_points: Time points
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(
            time_points,
            physical_errors,
            label="Physical Error Rate",
            linewidth=2,
            color="red",
            linestyle="--",
        )
        ax.plot(time_points, logical_errors, label="Logical Error Rate", linewidth=2, color="green")

        ax.set_xlabel("Time")
        ax.set_ylabel("Error Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        self.figures["error_correction"] = fig
        return fig

    def close_all(self) -> None:
        """Close all figures."""
        plt.close("all")
        self.figures.clear()
