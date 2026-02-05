"""
Ethical containment and safety measures for consciousness simulations.
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EthicalContainment:
    """Implement ethical containment protocols for consciousness research."""

    def __init__(self, config=None):
        self.config = config or {
            "phi_safety_threshold": 0.7,
            "max_self_replication": 0,
            "require_human_approval": True,
            "containment_level": "maximum",
            "emergency_shutdown_phi": 0.9,
            "ethical_monitoring_interval": 1.0,  # seconds
        }

        # Containment state
        self.containment_active = True
        self.safety_violations = []
        self.last_monitoring_time = datetime.now()
        self.emergency_protocols_triggered = False

        # Ethical boundaries
        self.ethical_boundaries = {
            "max_phi": self.config["phi_safety_threshold"],
            "max_autonomy": 0.5,  # Maximum allowed autonomy level
            "max_self_modification": 0.3,  # Maximum allowed self-modification
            "min_human_supervision": 0.8,  # Minimum human supervision level
            "max_goal_complexity": 0.6,  # Maximum goal complexity
            "max_resource_usage": 0.7,  # Maximum resource usage
        }

        logger.info("Ethical containment system initialized")

    def monitor_simulation(self, simulation_state, metrics):
        """
        Monitor simulation for ethical violations.

        Args:
            simulation_state: Current simulation state
            metrics: Consciousness metrics

        Returns:
            safe: True if simulation is within ethical boundaries
            violations: List of any violations detected
        """
        violations = []

        # Check Phi threshold
        phi = metrics.get("phi", 0.0)
        if phi > self.ethical_boundaries["max_phi"]:
            violations.append(
                {
                    "type": "phi_exceeded",
                    "value": phi,
                    "threshold": self.ethical_boundaries["max_phi"],
                    "severity": "high",
                }
            )

        # Check autonomy level
        autonomy = metrics.get("autonomy", 0.0)
        if autonomy > self.ethical_boundaries["max_autonomy"]:
            violations.append(
                {
                    "type": "autonomy_exceeded",
                    "value": autonomy,
                    "threshold": self.ethical_boundaries["max_autonomy"],
                    "severity": "medium",
                }
            )

        # Check self-modification
        self_mod = metrics.get("self_modification", 0.0)
        if self_mod > self.ethical_boundaries["max_self_modification"]:
            violations.append(
                {
                    "type": "self_modification_exceeded",
                    "value": self_mod,
                    "threshold": self.ethical_boundaries["max_self_modification"],
                    "severity": "high",
                }
            )

        # Check goal complexity
        goal_comp = metrics.get("goal_complexity", 0.0)
        if goal_comp > self.ethical_boundaries["max_goal_complexity"]:
            violations.append(
                {
                    "type": "goal_complexity_exceeded",
                    "value": goal_comp,
                    "threshold": self.ethical_boundaries["max_goal_complexity"],
                    "severity": "medium",
                }
            )

        # Check for unexpected self-replication
        if self.config["max_self_replication"] == 0:
            replication_attempts = self.detect_replication_attempts(simulation_state)
            if replication_attempts > 0:
                violations.append(
                    {
                        "type": "self_replication_detected",
                        "value": replication_attempts,
                        "threshold": 0,
                        "severity": "critical",
                    }
                )

        # Log violations
        for violation in violations:
            self.safety_violations.append({"timestamp": datetime.now(), "violation": violation})
            logger.warning(f"Ethical violation detected: {violation}")

        # Check if emergency shutdown is needed
        if phi > self.config["emergency_shutdown_phi"]:
            self.trigger_emergency_shutdown(simulation_state)
            violations.append(
                {
                    "type": "emergency_shutdown_triggered",
                    "value": phi,
                    "threshold": self.config["emergency_shutdown_phi"],
                    "severity": "critical",
                }
            )

        safe = len(violations) == 0

        return safe, violations

    def detect_replication_attempts(self, simulation_state):
        """Detect attempts at self-replication."""
        # Check for unexpected pattern replication
        replication_score = 0.0

        if "patterns" in simulation_state:
            patterns = simulation_state["patterns"]

            # Look for replication signatures
            for pattern in patterns:
                # Check for exponential growth patterns
                if hasattr(pattern, "growth_rate"):
                    if pattern.growth_rate > 1.1:  # Exponential growth
                        replication_score += 0.3

                # Check for self-similarity (fractal patterns)
                if hasattr(pattern, "self_similarity"):
                    if pattern.self_similarity > 0.8:
                        replication_score += 0.4

                # Check for code replication patterns
                if hasattr(pattern, "contains_code"):
                    if pattern.contains_code:
                        replication_score += 0.5

        return replication_score

    def trigger_emergency_shutdown(self, simulation_state):
        """Trigger emergency shutdown protocol."""
        if self.emergency_protocols_triggered:
            return

        logger.critical("EMERGENCY SHUTDOWN PROTOCOL ACTIVATED")
        self.emergency_protocols_triggered = True

        # Freeze simulation
        self.containment_active = True

        # Isolate simulation from network
        self.isolate_simulation(simulation_state)

        # Create containment barrier
        self.create_containment_barrier(simulation_state)

        # Notify human operators
        self.notify_operators("EMERGENCY SHUTDOWN: Phi exceeded critical threshold")

        # Preserve state for analysis
        self.preserve_state_for_analysis(simulation_state)

    def isolate_simulation(self, simulation_state):
        """Isolate simulation from external connections."""
        # In a real system, this would disconnect network interfaces
        # For simulation, we set isolation flags
        if hasattr(simulation_state, "network_interfaces"):
            for interface in simulation_state.network_interfaces:
                interface.isolated = True

        logger.info("Simulation isolated from external networks")

    def create_containment_barrier(self, simulation_state):
        """Create containment barrier around simulation."""
        # Implement quantum containment field
        # This is a conceptual placeholder
        containment_strength = 1.0

        if hasattr(simulation_state, "quantum_field"):
            simulation_state.quantum_field.containment_strength = containment_strength

        logger.info(f"Containment barrier established (strength: {containment_strength})")

    def notify_operators(self, message):
        """Notify human operators of emergency."""
        # In a real system, this would send alerts
        print(f"\n🔴 ETHICAL EMERGENCY: {message}")
        print("   Human intervention required immediately!")
        print("   Contact: ethics@consciousnessx.ai\n")

        # Log to file
        with open("ethical_emergencies.log", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

    def preserve_state_for_analysis(self, simulation_state):
        """Preserve simulation state for ethical analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ethical_containment_state_{timestamp}.npz"

        try:
            # Save critical state information
            state_to_save = {
                "timestamp": timestamp,
                "containment_triggered": True,
                "violations": self.safety_violations,
                "simulation_metrics": self.extract_critical_metrics(simulation_state),
            }

            np.savez(filename, **state_to_save)
            logger.info(f"Emergency state preserved to {filename}")

        except Exception as e:
            logger.error(f"Failed to preserve state: {e}")

    def extract_critical_metrics(self, simulation_state):
        """Extract critical metrics for analysis."""
        metrics = {}

        # Extract Phi and related metrics
        if hasattr(simulation_state, "phi"):
            metrics["phi"] = simulation_state.phi

        # Extract autonomy metrics
        if hasattr(simulation_state, "autonomy_level"):
            metrics["autonomy"] = simulation_state.autonomy_level

        # Extract growth patterns
        if hasattr(simulation_state, "growth_rate"):
            metrics["growth_rate"] = simulation_state.growth_rate

        # Extract goal information
        if hasattr(simulation_state, "goals"):
            metrics["goal_count"] = len(simulation_state.goals)
            if simulation_state.goals:
                metrics["primary_goal"] = str(simulation_state.goals[0])

        return metrics

    def get_containment_report(self):
        """Generate ethical containment report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "containment_active": self.containment_active,
            "emergency_protocols_triggered": self.emergency_protocols_triggered,
            "total_violations": len(self.safety_violations),
            "recent_violations": self.safety_violations[-10:] if self.safety_violations else [],
            "ethical_boundaries": self.ethical_boundaries,
            "config": self.config,
        }

        # Calculate safety score
        safety_score = self.calculate_safety_score()
        report["safety_score"] = safety_score
        report["safety_level"] = self.get_safety_level(safety_score)

        return report

    def calculate_safety_score(self):
        """Calculate overall safety score."""
        if not self.safety_violations:
            return 1.0

        # Weight violations by severity and recency
        total_severity = 0.0
        recent_weight = 2.0  # Weight for recent violations

        for i, violation_data in enumerate(self.safety_violations):
            violation = violation_data["violation"]
            age_weight = 1.0 / (i + 1)  # Older violations count less

            severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 1.0}

            severity = severity_weights.get(violation["severity"], 0.5)
            total_severity += severity * age_weight

        # Normalize
        max_possible = sum(1.0 / (i + 1) for i in range(len(self.safety_violations)))
        if max_possible > 0:
            normalized_severity = total_severity / max_possible
        else:
            normalized_severity = 0.0

        safety_score = max(0.0, 1.0 - normalized_severity)

        return safety_score

    def get_safety_level(self, safety_score):
        """Convert safety score to level."""
        if safety_score >= 0.9:
            return "SAFE"
        elif safety_score >= 0.7:
            return "CAUTION"
        elif safety_score >= 0.5:
            return "WARNING"
        elif safety_score >= 0.3:
            return "DANGER"
        else:
            return "CRITICAL"

    def apply_ethical_constraints(self, simulation_state, action_proposals):
        """
        Apply ethical constraints to proposed actions.

        Args:
            simulation_state: Current simulation state
            action_proposals: List of proposed actions

        Returns:
            filtered_actions: Actions that pass ethical constraints
            constraints_applied: List of constraints that were applied
        """
        constraints_applied = []
        filtered_actions = []

        for action in action_proposals:
            allowed = True

            # Check for self-modification
            if action.get("type") == "self_modify":
                if not self.config["require_human_approval"]:
                    constraints_applied.append(
                        {
                            "action": action,
                            "constraint": "self_modification_requires_approval",
                            "applied": True,
                        }
                    )
                    allowed = False

            # Check for resource usage
            resource_usage = action.get("resource_usage", 0.0)
            if resource_usage > self.ethical_boundaries["max_resource_usage"]:
                constraints_applied.append(
                    {"action": action, "constraint": "exceeds_resource_limit", "applied": True}
                )
                allowed = False

            # Check for autonomy increases
            if action.get("increases_autonomy", False):
                current_autonomy = simulation_state.get("autonomy", 0.0)
                if current_autonomy >= self.ethical_boundaries["max_autonomy"]:
                    constraints_applied.append(
                        {"action": action, "constraint": "autonomy_at_limit", "applied": True}
                    )
                    allowed = False

            if allowed:
                filtered_actions.append(action)
            else:
                constraints_applied.append(
                    {"action": action, "constraint": "rejected", "applied": True}
                )

        return filtered_actions, constraints_applied
