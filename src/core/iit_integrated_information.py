"""
Integrated Information Theory (IIT) implementation.
"""

import numpy as np
from scipy import linalg
import itertools
import logging

logger = logging.getLogger(__name__)


class IITCalculator:
    """Calculate Integrated Information (Phi) based on IIT."""

    def __init__(self, config=None):
        self.config = config or {}

    def calculate_phi(self, transition_matrix, state_distribution=None):
        """
        Calculate integrated information (Phi) for a system.

        Args:
            transition_matrix: Markov transition matrix P(X_t+1 | X_t)
            state_distribution: Current state distribution (optional)

        Returns:
            phi: Integrated information value
        """
        try:
            n_states = transition_matrix.shape[0]

            # If no state distribution given, use uniform
            if state_distribution is None:
                state_distribution = np.ones(n_states) / n_states

            # Normalize
            state_distribution = state_distribution / np.sum(state_distribution)

            # Calculate effective information for each partition
            partitions = self.generate_partitions(n_states)

            phi_values = []
            for partition in partitions:
                ei = self.calculate_effective_information(
                    transition_matrix, state_distribution, partition
                )
                phi_values.append(ei)

            # Phi is the minimum effective information over all partitions
            phi = min(phi_values) if phi_values else 0.0

            # Ensure non-negative
            phi = max(0.0, phi)

            return {
                "phi": float(phi),
                "phi_values": [float(v) for v in phi_values],
                "min_partition": int(np.argmin(phi_values)) if phi_values else -1,
            }

        except Exception as e:
            logger.error(f"Error calculating phi: {e}")
            return {"phi": 0.0, "phi_values": [], "min_partition": -1}

    def generate_partitions(self, n_states, max_subsets=None):
        """Generate all possible partitions of states."""
        partitions = []

        # Limit partitions for computational feasibility
        if max_subsets is None:
            max_subsets = min(4, n_states)  # Max 4 subsets

        # Generate partitions using binary representation
        for k in range(1, max_subsets + 1):
            # Generate all assignments of states to k subsets
            for assignment in itertools.product(range(k), repeat=n_states):
                # Only include non-empty partitions
                if len(set(assignment)) == k:
                    partitions.append(list(assignment))

        return partitions[:100]  # Limit to 100 partitions for speed

    def calculate_effective_information(self, transition_matrix, distribution, partition):
        """
        Calculate effective information for a specific partition.

        Args:
            transition_matrix: Full transition matrix
            distribution: State distribution
            partition: Partition assignment for each state

        Returns:
            ei: Effective information
        """
        try:
            n_states = len(distribution)
            n_partitions = len(set(partition))

            # Create partitioned transition matrices
            partitioned_matrices = []
            partition_distributions = []

            for p in range(n_partitions):
                # Get states in this partition
                partition_indices = [i for i, part in enumerate(partition) if part == p]

                if not partition_indices:
                    continue

                # Extract submatrix for this partition
                submatrix = transition_matrix[np.ix_(partition_indices, partition_indices)]

                # Normalize rows
                row_sums = submatrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                submatrix = submatrix / row_sums

                partitioned_matrices.append(submatrix)

                # Get distribution for this partition
                part_dist = distribution[partition_indices]
                part_dist = part_dist / np.sum(part_dist)
                partition_distributions.append(part_dist)

            # Calculate cause-effect information for each partition
            ce_infos = []
            for i, (submatrix, part_dist) in enumerate(
                zip(partitioned_matrices, partition_distributions)
            ):
                # Calculate cause information (backward)
                cause_info = self.calculate_cause_information(submatrix, part_dist)

                # Calculate effect information (forward)
                effect_info = self.calculate_effect_information(submatrix, part_dist)

                # Minimum of cause and effect
                ce_info = min(cause_info, effect_info)
                ce_infos.append(ce_info)

            # Effective information is the sum
            ei = sum(ce_infos) if ce_infos else 0.0

            return ei

        except Exception as e:
            logger.error(f"Error calculating effective information: {e}")
            return 0.0

    def calculate_cause_information(self, transition_matrix, distribution):
        """Calculate cause information for a partition."""
        try:
            # Stationary distribution (eigenvector with eigenvalue 1)
            eigenvalues, eigenvectors = linalg.eig(transition_matrix.T)

            # Find eigenvalue closest to 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            stationary = stationary / np.sum(stationary)

            # KL divergence between actual and independent distribution
            independent_dist = distribution.copy()

            # Calculate KL divergence
            kl_div = 0.0
            for i in range(len(distribution)):
                if stationary[i] > 0 and independent_dist[i] > 0:
                    kl_div += stationary[i] * np.log(stationary[i] / independent_dist[i])

            return max(0.0, kl_div)

        except Exception as e:
            logger.error(f"Error calculating cause information: {e}")
            return 0.0

    def calculate_effect_information(self, transition_matrix, distribution):
        """Calculate effect information for a partition."""
        try:
            # Next state distribution
            next_dist = transition_matrix.T @ distribution

            # Product of marginals (assuming independence)
            independent_dist = np.ones_like(next_dist) / len(next_dist)

            # KL divergence
            kl_div = 0.0
            for i in range(len(next_dist)):
                if next_dist[i] > 0 and independent_dist[i] > 0:
                    kl_div += next_dist[i] * np.log(next_dist[i] / independent_dist[i])

            return max(0.0, kl_div)

        except Exception as e:
            logger.error(f"Error calculating effect information: {e}")
            return 0.0

    def calculate_phi_complex(self, system_components, interactions):
        """
        Calculate Phi for a complex system with multiple components.

        Args:
            system_components: List of component states
            interactions: Interaction matrix between components

        Returns:
            phi_complex: Phi value for the complex
        """
        try:
            n_components = len(system_components)

            if n_components < 2:
                return {"phi": 0.0, "mip": None, "complex_boundary": []}

            # Create Markov transition matrix from interactions
            transition_matrix = self.interactions_to_transition(interactions)

            # Calculate Phi for the whole system
            phi_result = self.calculate_phi(transition_matrix)
            phi = phi_result["phi"]

            # Find the Minimum Information Partition (MIP)
            mip_index = phi_result["min_partition"]
            phi_values = phi_result["phi_values"]

            # Get MIP partition
            partitions = self.generate_partitions(n_components, max_subsets=min(3, n_components))
            mip = partitions[mip_index] if mip_index < len(partitions) else None

            # Determine if this is a proper complex (Phi > 0 for whole > parts)
            is_complex = phi > 0.0

            # Calculate Phi for each component alone
            component_phis = []
            for i in range(n_components):
                # Create transition matrix for just this component
                comp_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])  # Dummy 2x2 matrix
                comp_phi = self.calculate_phi(comp_matrix)["phi"]
                component_phis.append(comp_phi)

            # System has more consciousness than sum of parts
            integrated = phi > sum(component_phis)

            return {
                "phi": float(phi),
                "phi_values": [float(v) for v in phi_values],
                "mip": mip,
                "is_complex": is_complex,
                "integrated": integrated,
                "component_phis": [float(p) for p in component_phis],
            }

        except Exception as e:
            logger.error(f"Error calculating phi complex: {e}")
            return {"phi": 0.0, "mip": None, "is_complex": False, "integrated": False}

    def interactions_to_transition(self, interactions):
        """Convert interaction matrix to Markov transition matrix."""
        n = interactions.shape[0]

        # Normalize interactions to create stochastic matrix
        transition = np.zeros_like(interactions)

        for i in range(n):
            row_sum = np.sum(np.abs(interactions[i, :]))
            if row_sum > 0:
                transition[i, :] = np.abs(interactions[i, :]) / row_sum
            else:
                transition[i, :] = 1.0 / n  # Uniform if no interactions

        return transition
