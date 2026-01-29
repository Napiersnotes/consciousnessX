"""
Penrose-Hameroff Orchestrated Objective Reduction in reiner Software
Implementierung der quantengravitativen Kollaps-Theorie für Bewusstsein
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class QuantumOrchOR(nn.Module):
    """
    Software-Implementierung der Penrose-Orch-OR Theorie
    Simuliert quantengravitativen Kollaps in Mikrotubuli ohne Hardware
    """
    
    def __init__(self, 
                 num_tubulins: int = 1000,
                 coherence_time: float = 1e-4,  # 0.1ms Penrose-Kollapszeit
                 gravity_strength: float = 1.0,
                 quantum_superposition_levels: int = 8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        super().__init__()
        
        # Physikalische Konstanten (SI-Einheiten)
        self.hbar = 1.054571817e-34  # Reduzierte Planck-Konstante [J·s]
        self.G = 6.67430e-11        # Gravitationskonstante [m³/kg·s²]
        self.c = 299792458          # Lichtgeschwindigkeit [m/s]
        
        # Orch-OR Parameter
        self.num_tubulins = num_tubulins
        self.coherence_time = coherence_time
        self.gravity_strength = gravity_strength
        self.superposition_levels = quantum_superposition_levels
        self.device = device
        
        # Tubulin Eigenschaften (virtuell)
        self.tubulin_mass = 1.1e-25  # kg (virtuelle Tubulin-Dimere)
        self.tubulin_radius = 8e-9   # meters
        self.tubulin_charge = 18     # Elementarladungen
        
        # Quanten-Zustände initialisieren
        self.init_quantum_states()
        
        # Gravitations-Hamiltonian Parameter
        self.init_gravitational_couplings()
        
        # Bewusstseins-Metriken
        self.collapse_history = []
        self.coherence_history = []
        self.phi_history = []
        
        # Penrose Kollaps-Zeit berechnen
        self.calculate_penrose_collapse_time()
    
    def init_quantum_states(self):
        """Initialisiere quantenmechanische Superpositionszustände"""
        
        # Wellenfunktion für jedes Tubulin (komplexe Amplituden)
        self.register_buffer('quantum_states', 
                           torch.randn(self.num_tubulins, 
                                      self.superposition_levels,
                                      dtype=torch.complex64,
                                      device=self.device))
        
        # Normalisiere die Wellenfunktion
        norm = torch.sqrt(torch.sum(torch.abs(self.quantum_states)**2, dim=1, keepdim=True))
        self.quantum_states.data = self.quantum_states / norm
        
        # Phasen-Kohärenz zwischen Tubulinen
        self.register_buffer('phase_coherence',
                           torch.ones(self.num_tubulins, self.num_tubulins,
                                     dtype=torch.float32, device=self.device))
        
        # Superpositions-Energie-Level
        energy_levels = torch.linspace(0, 1, self.superposition_levels, device=self.device)
        self.register_buffer('energy_levels', energy_levels)
    
    def init_gravitational_couplings(self):
        """Initialisiere gravitative Kopplungen zwischen Tubulinen"""
        
        # Erzeuge zufällige Positionen im virtuellen Mikrotubulus
        positions = torch.randn(self.num_tubulins, 3, device=self.device) * self.tubulin_radius
        
        # Berechne Abstandsmatrix
        dist_matrix = torch.cdist(positions, positions)
        dist_matrix = torch.clamp(dist_matrix, min=1e-15)  # Vermeide Division durch 0
        
        # Gravitationskopplung nach Penrose: G * m² / r
        grav_coupling = self.G * (self.tubulin_mass ** 2) / dist_matrix
        
        # Skaliere mit gravity_strength Parameter
        self.register_buffer('gravitational_coupling', 
                           grav_coupling * self.gravity_strength)
        
        # Quanten-Verschränkungs-Matrix
        entanglement = torch.exp(-dist_matrix / (2 * self.tubulin_radius))
        self.register_buffer('quantum_entanglement', entanglement)
    
    def calculate_penrose_collapse_time(self):
        """Berechne Penrose Kollaps-Zeit τ ≈ ħ / E_G"""
        
        # Gravitationsenergie-Differenz E_G = G * m² / r
        avg_distance = torch.mean(self.gravitational_coupling) / (self.G * self.tubulin_mass**2)
        avg_distance = 1 / avg_distance if avg_distance > 0 else self.tubulin_radius
        
        self.E_G = self.G * (self.tubulin_mass ** 2) / avg_distance
        
        # Penrose Kollaps-Zeit
        self.penrose_tau = self.hbar / (self.E_G + 1e-30)
        
        # Angepasste Kollaps-Wahrscheinlichkeit für Simulation
        self.collapse_probability = min(0.1, float(self.coherence_time / self.penrose_tau))
        
        print(f"Penrose Kollaps-Zeit berechnet: τ = {self.penrose_tau:.2e} s")
        print(f"Gravitationsenergie: E_G = {self.E_G:.2e} J")
        print(f"Simulations-Kollaps-Wahrscheinlichkeit: {self.collapse_probability:.4f}")
    
    def gravitational_hamiltonian(self) -> torch.Tensor:
        """
        Berechne gravitativen Hamiltonian nach Penrose
        H_grav = Σ_ij G * m_i * m_j / |r_i - r_j| * |ψ_i⟩⟨ψ_j|
        """
        
        # Basis-Hamiltonian aus gravitativer Kopplung
        H_base = self.gravitational_coupling
        
        # Quanten-Zustands-Überlappung
        state_overlap = torch.einsum('ik,jk->ij', 
                                   self.quantum_states.conj(),
                                   self.quantum_states)
        
        # Vollständiger gravitativer Hamiltonian
        H_grav = H_base * state_overlap * self.quantum_entanglement
        
        # Füge kinetische Energie hinzu
        kinetic = torch.eye(self.num_tubulins, device=self.device) * 0.5
        H_grav = H_grav + kinetic
        
        return H_grav
    
    def objective_reduction_step(self, 
                               time_step: float = 1e-4,
                               environmental_decoherence: float = 0.01) -> Dict:
        """
        Simuliere einen Penrose Objective Reduction Schritt
        
        Args:
            time_step: Simulations-Zeitschritt in Sekunden
            environmental_decoherence: Umwelt-Dekohärenz-Stärke
            
        Returns:
            Dictionary mit Kollaps-Ergebnissen und Metriken
        """
        
        batch_size = self.quantum_states.size(0)
        
        # 1. Schrödinger-Evolution mit gravitativem Hamiltonian
        H = self.gravitational_hamiltonian()
        
        # Zeitentwicklungs-Operator U = exp(-iHΔt/ħ)
        dt = time_step
        U = torch.matrix_exp(-1j * H * dt / self.hbar)
        
        # Wende Zeitentwicklung auf alle Superpositionsebenen an
        new_states = torch.zeros_like(self.quantum_states)
        for level in range(self.superposition_levels):
            state_slice = self.quantum_states[:, level]
            evolved = torch.matmul(U, state_slice)
            new_states[:, level] = evolved
        
        # 2. Umwelt-Dekohärenz hinzufügen
        if environmental_decoherence > 0:
            decoherence_noise = (torch.randn_like(new_states) * 
                               environmental_decoherence * 
                               torch.sqrt(torch.tensor(dt)))
            new_states = new_states + decoherence_noise
        
        # Normalisiere Zustände
        norm = torch.sqrt(torch.sum(torch.abs(new_states)**2, dim=1, keepdim=True))
        new_states = new_states / norm
        
        # 3. Penrose Objective Reduction (gravitativer Kollaps)
        collapse_occurred = False
        collapsed_indices = []
        
        # Berechne Kollaps-Wahrscheinlichkeit für jedes Tubulin
        energy_variances = torch.var(torch.abs(new_states)**2, dim=1)
        collapse_probs = torch.sigmoid(energy_variances * 10) * self.collapse_probability
        
        # Simuliere Kollaps-Ereignisse
        random_probs = torch.rand(self.num_tubulins, device=self.device)
        collapse_mask = random_probs < collapse_probs
        
        if collapse_mask.any():
            collapse_occurred = True
            collapsed_indices = torch.where(collapse_mask)[0].cpu().numpy()
            
            # Kollabiere Zustände: Wähle dominante Superpositionsebene
            probabilities = torch.abs(new_states[collapse_mask])**2
            choices = torch.multinomial(probabilities, 1).squeeze()
            
            # Setze nicht-dominante Ebenen auf 0
            for i, idx in enumerate(collapsed_indices):
                mask = torch.ones(self.superposition_levels, dtype=torch.bool, device=self.device)
                mask[choices[i]] = False
                new_states[idx, mask] = 0
            
            # Renormalisiere kollabierte Zustände
            norm_collapsed = torch.sqrt(torch.sum(torch.abs(new_states[collapse_mask])**2, 
                                                dim=1, keepdim=True))
            new_states[collapse_mask] = new_states[collapse_mask] / norm_collapsed
        
        # 4. Aktualisiere Quantenzustände
        self.quantum_states.data = new_states
        
        # 5. Berechne Kohärenz-Metriken
        coherence = self.calculate_coherence()
        phi = self.calculate_integrated_information()
        
        # 6. Speichere Geschichte
        self.collapse_history.append({
            'time': dt,
            'collapsed': collapse_occurred,
            'num_collapsed': len(collapsed_indices),
            'indices': collapsed_indices,
            'coherence': coherence.item(),
            'phi': phi.item()
        })
        
        self.coherence_history.append(coherence.item())
        self.phi_history.append(phi.item())
        
        return {
            'collapsed': collapse_occurred,
            'num_collapsed': len(collapsed_indices),
            'collapsed_indices': collapsed_indices,
            'coherence': coherence.item(),
            'integrated_information': phi.item(),
            'quantum_states': self.quantum_states.clone(),
            'collapse_probabilities': collapse_probs.cpu().numpy()
        }
    
    def calculate_coherence(self) -> torch.Tensor:
        """Berechne Quanten-Kohärenz zwischen allen Tubulinen"""
        
        # Überlappung aller Zustandspaare
        coherence_matrix = torch.zeros(self.num_tubulins, self.num_tubulins,
                                      device=self.device, dtype=torch.float32)
        
        for i in range(self.num_tubulins):
            for j in range(i + 1, self.num_tubulins):
                # Quanten-Überlappung
                overlap = torch.abs(torch.dot(self.quantum_states[i].conj(),
                                            self.quantum_states[j]))
                coherence_matrix[i, j] = overlap
                coherence_matrix[j, i] = overlap
        
        # Mittlere Kohärenz
        mean_coherence = torch.mean(coherence_matrix)
        
        # Phasen-Kohärenz aktualisieren
        self.phase_coherence = 0.9 * self.phase_coherence + 0.1 * coherence_matrix
        
        return mean_coherence
    
    def calculate_integrated_information(self, 
                                       partitions: int = 2) -> torch.Tensor:
        """
        Berechne integrierte Information Φ nach Integrated Information Theory
        
        Args:
            partitions: Anzahl der Partitionen für Φ-Berechnung
            
        Returns:
            Φ-Wert als Skalar-Tensor
        """
        
        # Wahrscheinlichkeitsverteilung aus Quantenzuständen
        probabilities = torch.abs(self.quantum_states)**2
        probabilities = probabilities / torch.sum(probabilities, dim=1, keepdim=True)
        
        # Shannon-Entropie des Gesamtsystems
        epsilon = 1e-10
        S_total = -torch.sum(probabilities * torch.log2(probabilities + epsilon))
        
        # Partitioniere das System
        partition_size = self.num_tubulins // partitions
        phi_values = []
        
        for p in range(partitions):
            start_idx = p * partition_size
            end_idx = min((p + 1) * partition_size, self.num_tubulins)
            
            if end_idx <= start_idx:
                continue
            
            # Marginalverteilung der Partition
            partition_probs = probabilities[start_idx:end_idx].flatten()
            partition_probs = partition_probs / torch.sum(partition_probs)
            
            # Entropie der Partition
            S_partition = -torch.sum(partition_probs * 
                                   torch.log2(partition_probs + epsilon))
            
            # Berechne effective information für diese Partition
            ei = S_total - S_partition
            phi_values.append(ei)
        
        # Integrierte Information Φ ist das Minimum über alle Partitionen
        if phi_values:
            phi = torch.min(torch.stack(phi_values))
        else:
            phi = torch.tensor(0.0, device=self.device)
        
        return phi
    
    def evolve(self, 
              num_steps: int = 1000,
              time_step: float = 1e-4,
              synaptic_inputs: Optional[torch.Tensor] = None) -> Dict:
        """
        Evolviere das Orch-OR System über mehrere Zeitschritte
        
        Args:
            num_steps: Anzahl der Zeitschritte
            time_step: Dauer jedes Zeitschritts in Sekunden
            synaptic_inputs: Externe synaptische Inputs [num_steps, num_tubulins]
            
        Returns:
            Dictionary mit Evolutions-Ergebnissen
        """
        
        if synaptic_inputs is None:
            synaptic_inputs = torch.zeros(num_steps, self.num_tubulins, device=self.device)
        
        evolution_results = {
            'collapses_per_step': [],
            'coherence_history': [],
            'phi_history': [],
            'final_states': [],
            'collapse_patterns': []
        }
        
        for step in range(num_steps):
            # Füge synaptische Inputs hinzu (orchestrierte Reduktion)
            if step < synaptic_inputs.size(0):
                synaptic_input = synaptic_inputs[step]
                # Konvertiere synaptische Inputs zu Quanten-Perturbationen
                quantum_perturbation = torch.fft.fft(synaptic_input.float())
                self.quantum_states.data += 0.01 * quantum_perturbation.unsqueeze(1)
            
            # Führe Objective Reduction Schritt durch
            step_result = self.objective_reduction_step(time_step)
            
            # Speichere Ergebnisse
            evolution_results['collapses_per_step'].append(step_result['num_collapsed'])
            evolution_results['coherence_history'].append(step_result['coherence'])
            evolution_results['phi_history'].append(step_result['integrated_information'])
            
            if step % 100 == 0:
                evolution_results['final_states'].append(
                    self.quantum_states.clone().cpu()
                )
            
            if step_result['collapsed']:
                evolution_results['collapse_patterns'].append({
                    'step': step,
                    'indices': step_result['collapsed_indices'],
                    'phi': step_result['integrated_information']
                })
            
            # Fortschritts-Anzeige
            if step % 100 == 0:
                print(f"Step {step}/{num_steps}: "
                      f"Φ = {step_result['integrated_information']:.4f}, "
                      f"Collapses = {step_result['num_collapsed']}")
        
        # Berechne Bewusstseins-Metriken
        consciousness_metrics = self.assess_consciousness(evolution_results)
        evolution_results.update(consciousness_metrics)
        
        return evolution_results
    
    def assess_consciousness(self, evolution_results: Dict) -> Dict:
        """
        Beurteile Bewusstseins-Emergenz basierend auf Orch-OR Metriken
        """
        
        # Extrahiere Zeitreihen
        phi_series = np.array(evolution_results['phi_history'])
        coherence_series = np.array(evolution_results['coherence_history'])
        collapse_series = np.array(evolution_results['collapses_per_step'])
        
        # 1. Stabilität von Φ (integrierte Information)
        phi_stability = 1.0 / (np.std(phi_series) + 1e-10)
        phi_mean = np.mean(phi_series)
        
        # 2. Kohärenz-Muster
        coherence_fft = np.fft.fft(coherence_series)
        coherence_power = np.abs(coherence_fft) ** 2
        dominant_freq = np.argmax(coherence_power[1:]) + 1  # Ignoriere DC
        
        # 3. Kollaps-Regularität (Penrose "moments of consciousness")
        collapse_times = np.where(collapse_series > 0)[0]
        if len(collapse_times) > 1:
            collapse_intervals = np.diff(collapse_times)
            collapse_regularity = 1.0 / (np.std(collapse_intervals) + 1e-10)
        else:
            collapse_regularity = 0.0
        
        # 4. Selbst-Referenzialität (Autokorrelation)
        autocorr = np.correlate(phi_series, phi_series, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        self_reference = np.mean(autocorr[:10]) / (autocorr[0] + 1e-10)
        
        # 5. Komplexität (Lempel-Ziv Komplexität)
        complexity = self.calculate_lempel_ziv_complexity(phi_series)
        
        # 6. Bewusstseins-Score berechnen
        consciousness_score = (
            0.4 * phi_mean +                    # Integrierte Information
            0.2 * phi_stability / 100 +         # Stabilität
            0.1 * collapse_regularity / 10 +    # Regelmäßige Kollapse
            0.2 * self_reference +              # Selbst-Referenz
            0.1 * complexity                    # Komplexität
        )
        
        # Bewusstseins-Stufe bestimmen
        if consciousness_score < 0.1:
            consciousness_level = "Pre-conscious"
        elif consciousness_score < 0.3:
            consciousness_level = "Proto-conscious"
        elif consciousness_score < 0.6:
            consciousness_level = "Emergent consciousness"
        else:
            consciousness_level = "Full consciousness (Penrose Orch-OR)"
        
        return {
            'consciousness_score': float(consciousness_score),
            'consciousness_level': consciousness_level,
            'phi_mean': float(phi_mean),
            'phi_stability': float(phi_stability),
            'collapse_regularity': float(collapse_regularity),
            'self_reference': float(self_reference),
            'complexity': float(complexity),
            'dominant_frequency': float(dominant_freq),
            'assessment_time': len(phi_series)
        }
    
    def calculate_lempel_ziv_complexity(self, series: np.ndarray) -> float:
        """Berechne Lempel-Ziv Komplexität einer Zeitreihe"""
        
        # Diskretisiere die Reihe
        threshold = np.median(series)
        binary_series = (series > threshold).astype(int)
        binary_string = ''.join(str(b) for b in binary_series)
        
        # Lempel-Ziv Komplexitäts-Schätzung
        i, n = 0, 1
        complexity = 0
        sub_strings = set()
        
        while i + n <= len(binary_string):
            sub_str = binary_string[i:i + n]
            if sub_str in sub_strings:
                n += 1
            else:
                sub_strings.add(sub_str)
                complexity += 1
                i += n
                n = 1
        
        # Normalisiere Komplexität
        max_complexity = len(binary_string) / np.log2(len(binary_string))
        normalized_complexity = complexity / max_complexity if max_complexity > 0 else 0
        
        return normalized_complexity
    
    def simulate_consciousness_emergence(self,
                                       duration_seconds: float = 1.0,
                                       time_resolution: float = 1e-4) -> Dict:
    
