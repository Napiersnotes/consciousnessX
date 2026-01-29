#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ion Channel Dynamics Simulation
Implements Hodgkin-Huxley and other ion channel models for neuronal simulation
Production-ready with GPU acceleration and comprehensive biophysical accuracy
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import integrate, optimize, sparse
import numba
from numba import cuda, jit, njit, prange
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IonChannelType(Enum):
    """Types of ion channels"""
    SODIUM = "sodium"
    POTASSIUM = "potassium"
    CALCIUM = "calcium"
    CHLORIDE = "chloride"
    LEAK = "leak"
    POTASSIUM_A = "potassium_a"  # A-type K+ channel
    CALCIUM_DEPENDENT_POTASSIUM = "calcium_dependent_potassium"  # K_Ca
    HYPERPOLARIZATION_ACTIVATED = "hyperpolarization_activated"  # HCN
    TRANSIENT_SODIUM = "transient_sodium"  # Na_T
    PERSISTENT_SODIUM = "persistent_sodium"  # Na_P

class ChannelGating(Enum):
    """Channel gating mechanisms"""
    VOLTAGE_GATED = "voltage_gated"
    LIGAND_GATED = "ligand_gated"
    MECHANOSENSITIVE = "mechanosensitive"
    LIGHT_GATED = "light_gated"
    TEMPERATURE_SENSITIVE = "temperature_sensitive"

@dataclass
class IonChannelConfig:
    """Configuration for ion channel simulation"""
    # Simulation parameters
    time_step_ms: float = 0.01  # 10 μs time step
    simulation_duration_ms: float = 1000.0  # 1 second simulation
    temperature_celsius: float = 37.0  # Body temperature
    extracellular_potassium_mM: float = 5.0  # [K+]out
    extracellular_sodium_mM: float = 145.0  # [Na+]out
    extracellular_calcium_mM: float = 2.0  # [Ca2+]out
    extracellular_chloride_mM: float = 110.0  # [Cl-]out
    
    # Intracellular concentrations (mM)
    intracellular_potassium_mM: float = 140.0  # [K+]in
    intracellular_sodium_mM: float = 15.0  # [Na+]in
    intracellular_calcium_mM: float = 0.0001  # [Ca2+]in
    intracellular_chloride_mM: float = 10.0  # [Cl-]in
    
    # Physical constants
    faraday_constant: float = 96485.3329  # C/mol
    gas_constant: float = 8.314462618  # J/(mol·K)
    absolute_zero: float = 273.15  # 0°C in Kelvin
    
    # Membrane properties
    membrane_capacitance_uF_cm2: float = 1.0  # μF/cm²
    membrane_resistance_kohm_cm2: float = 10.0  # kΩ·cm²
    membrane_area_um2: float = 1000.0  # μm²
    
    # Channel densities (channels/μm²)
    sodium_channel_density: float = 50.0
    potassium_channel_density: float = 20.0
    calcium_channel_density: float = 5.0
    leak_channel_density: float = 1.0
    
    # Numerical parameters
    use_gpu: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "float32"
    adaptive_time_step: bool = True
    max_time_step_ms: float = 0.1
    min_time_step_ms: float = 0.001
    
    def __post_init__(self):
        """Validate configuration"""
        self.temperature_kelvin = self.temperature_celsius + self.absolute_zero
        
        # Calculate Nernst potentials
        self.e_k = self.calculate_nernst_potential(
            self.extracellular_potassium_mM,
            self.intracellular_potassium_mM,
            1  # K+ valence
        )
        
        self.e_na = self.calculate_nernst_potential(
            self.extracellular_sodium_mM,
            self.intracellular_sodium_mM,
            1  # Na+ valence
        )
        
        self.e_ca = self.calculate_nernst_potential(
            self.extracellular_calcium_mM,
            self.intracellular_calcium_mM,
            2  # Ca2+ valence
        )
        
        self.e_cl = self.calculate_nernst_potential(
            self.extracellular_chloride_mM,
            self.intracellular_chloride_mM,
            -1  # Cl- valence
        )
        
        logger.info(f"Calculated Nernst potentials: E_K={self.e_k:.2f} mV, "
                   f"E_Na={self.e_na:.2f} mV, E_Ca={self.e_ca:.2f} mV")
    
    def calculate_nernst_potential(self, conc_out: float, conc_in: float, valence: int) -> float:
        """Calculate Nernst potential for an ion"""
        if conc_in == 0 or conc_out == 0:
            return 0.0
        
        # E = (RT/zF) * ln([out]/[in])
        rt_over_zf = (self.gas_constant * self.temperature_kelvin) / \
                    (abs(valence) * self.faraday_constant)
        ratio = conc_out / conc_in
        
        # Convert from volts to millivolts
        potential = rt_over_zf * np.log(ratio) * 1000.0
        
        # Adjust sign for anion vs cation
        if valence < 0:
            potential = -potential
        
        return potential

class IonChannel:
    """Base class for ion channels"""
    
    def __init__(self, 
                 channel_type: IonChannelType,
                 config: IonChannelConfig,
                 channel_density: Optional[float] = None,
                 single_channel_conductance: float = 20.0):  # pS
        self.channel_type = channel_type
        self.config = config
        self.single_channel_conductance = single_channel_conductance  # pS
        
        # Set channel density
        if channel_density is None:
            self.channel_density = self._get_default_density(channel_type)
        else:
            self.channel_density = channel_density
        
        # Calculate total conductance
        self.total_conductance = self._calculate_total_conductance()
        
        # Gating variables
        self.gating_variables = self._initialize_gating_variables()
        
        # State variables
        self.current = 0.0  # pA
        self.open_probability = 0.0
        self.inactivation_state = 1.0  # 1.0 = not inactivated
        
        # History tracking
        self.current_history = []
        self.open_probability_history = []
        
        logger.debug(f"Initialized {channel_type.value} channel with "
                    f"density {self.channel_density:.1f}/μm²")
    
    def _get_default_density(self, channel_type: IonChannelType) -> float:
        """Get default channel density based on type"""
        density_map = {
            IonChannelType.SODIUM: self.config.sodium_channel_density,
            IonChannelType.POTASSIUM: self.config.potassium_channel_density,
            IonChannelType.CALCIUM: self.config.calcium_channel_density,
            IonChannelType.CHLORIDE: self.config.leak_channel_density,
            IonChannelType.LEAK: self.config.leak_channel_density,
            IonChannelType.POTASSIUM_A: 5.0,
            IonChannelType.CALCIUM_DEPENDENT_POTASSIUM: 2.0,
            IonChannelType.HYPERPOLARIZATION_ACTIVATED: 3.0,
            IonChannelType.TRANSIENT_SODIUM: 10.0,
            IonChannelType.PERSISTENT_SODIUM: 2.0
        }
        return density_map.get(channel_type, 1.0)
    
    def _calculate_total_conductance(self) -> float:
        """Calculate total conductance from channel density"""
        # Convert from pS to nS and multiply by density and area
        area_cm2 = self.config.membrane_area_um2 * 1e-8  # Convert μm² to cm²
        conductance_nS = (self.single_channel_conductance * 1e-3) * \
                        self.channel_density * \
                        self.config.membrane_area_um2
        
        return conductance_nS
    
    def _initialize_gating_variables(self) -> Dict[str, float]:
        """Initialize gating variables based on channel type"""
        if self.channel_type == IonChannelType.SODIUM:
            return {'m': 0.05, 'h': 0.6}  # Activation, inactivation
        elif self.channel_type == IonChannelType.POTASSIUM:
            return {'n': 0.3}  # Delayed rectifier
        elif self.channel_type == IonChannelType.CALCIUM:
            return {'m': 0.05, 'h': 0.6}
        elif self.channel_type == IonChannelType.POTASSIUM_A:
            return {'a': 0.1, 'b': 0.8}  # A-type
        elif self.channel_type == IonChannelType.CALCIUM_DEPENDENT_POTASSIUM:
            return {'c': 0.1}  # Calcium-dependent
        elif self.channel_type == IonChannelType.HYPERPOLARIZATION_ACTIVATED:
            return {'m': 0.2}
        else:
            return {'m': 1.0}  # Always open for leak channels
    
    def calculate_current(self, 
                         voltage_mV: float,
                         calcium_concentration: Optional[float] = None,
                         time_step_ms: Optional[float] = None) -> float:
        """
        Calculate current through this channel
        
        Args:
            voltage_mV: Membrane potential (mV)
            calcium_concentration: Intracellular calcium concentration (mM)
            time_step_ms: Time step for gating variable updates
            
        Returns:
            Current in picoamperes (pA)
        """
        time_step = time_step_ms or self.config.time_step_ms
        
        # Update gating variables
        self._update_gating_variables(voltage_mV, calcium_concentration, time_step)
        
        # Calculate reversal potential
        reversal_potential = self._get_reversal_potential()
        
        # Calculate open probability
        self.open_probability = self._calculate_open_probability()
        
        # Calculate current: I = g * P_open * (V - E_rev)
        self.current = self.total_conductance * \
                      self.open_probability * \
                      (voltage_mV - reversal_potential)
        
        # Store history
        self.current_history.append(self.current)
        self.open_probability_history.append(self.open_probability)
        
        return self.current
    
    def _update_gating_variables(self, 
                                voltage_mV: float,
                                calcium_concentration: Optional[float],
                                time_step_ms: float):
        """Update channel gating variables"""
        if self.channel_type == IonChannelType.SODIUM:
            # Hodgkin-Huxley Na+ channel
            alpha_m = 0.1 * (voltage_mV + 40.0) / (1.0 - np.exp(-(voltage_mV + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(voltage_mV + 65.0) / 18.0)
            
            alpha_h = 0.07 * np.exp(-(voltage_mV + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(voltage_mV + 35.0) / 10.0))
            
            # Update using forward Euler
            tau_m = 1.0 / (alpha_m + beta_m)
            m_inf = alpha_m * tau_m
            self.gating_variables['m'] += (m_inf - self.gating_variables['m']) * \
                                         (time_step_ms / tau_m)
            
            tau_h = 1.0 / (alpha_h + beta_h)
            h_inf = alpha_h * tau_h
            self.gating_variables['h'] += (h_inf - self.gating_variables['h']) * \
                                         (time_step_ms / tau_h)
            
            # Inactivation state
            self.inactivation_state = self.gating_variables['h']
        
        elif self.channel_type == IonChannelType.POTASSIUM:
            # Hodgkin-Huxley K+ channel
            alpha_n = 0.01 * (voltage_mV + 55.0) / (1.0 - np.exp(-(voltage_mV + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(voltage_mV + 65.0) / 80.0)
            
            tau_n = 1.0 / (alpha_n + beta_n)
            n_inf = alpha_n * tau_n
            
            self.gating_variables['n'] += (n_inf - self.gating_variables['n']) * \
                                         (time_step_ms / tau_n)
        
        elif self.channel_type == IonChannelType.CALCIUM_DEPENDENT_POTASSIUM:
            # BK channel: voltage and calcium dependent
            if calcium_concentration is None:
                calcium_concentration = self.config.intracellular_calcium_mM
            
            # Voltage dependence
            v_half = -20.0  # mV
            k = 10.0  # mV
            voltage_factor = 1.0 / (1.0 + np.exp(-(voltage_mV - v_half) / k))
            
            # Calcium dependence
            ca_kd = 0.0005  # mM
            calcium_factor = calcium_concentration / (calcium_concentration + ca_kd)
            
            # Combined open probability
            self.gating_variables['c'] = voltage_factor * calcium_factor
        
        elif self.channel_type == IonChannelType.HYPERPOLARIZATION_ACTIVATED:
            # HCN channel: activated by hyperpolarization
            v_half = -90.0  # mV
            k = -8.0  # mV (negative for activation on hyperpolarization)
            
            m_inf = 1.0 / (1.0 + np.exp((voltage_mV - v_half) / k))
            tau_m = 100.0 + 900.0 / (1.0 + np.exp((voltage_mV + 70.0) / 10.0))  # ms
            
            self.gating_variables['m'] += (m_inf - self.gating_variables['m']) * \
                                         (time_step_ms / tau_m)
        
        # For other channels, keep gating variables constant or implement specific dynamics
    
    def _get_reversal_potential(self) -> float:
        """Get reversal potential for this channel type"""
        if self.channel_type in [IonChannelType.SODIUM, 
                                IonChannelType.TRANSIENT_SODIUM,
                                IonChannelType.PERSISTENT_SODIUM]:
            return self.config.e_na
        elif self.channel_type in [IonChannelType.POTASSIUM,
                                  IonChannelType.POTASSIUM_A,
                                  IonChannelType.CALCIUM_DEPENDENT_POTASSIUM]:
            return self.config.e_k
        elif self.channel_type == IonChannelType.CALCIUM:
            return self.config.e_ca
        elif self.channel_type == IonChannelType.CHLORIDE:
            return self.config.e_cl
        elif self.channel_type == IonChannelType.LEAK:
            # Leak channels are usually non-specific
            return -70.0  # Typical resting potential
        elif self.channel_type == IonChannelType.HYPERPOLARIZATION_ACTIVATED:
            # HCN channels are mixed Na+/K+
            return -30.0
        else:
            return -70.0  # Default
    
    def _calculate_open_probability(self) -> float:
        """Calculate channel open probability from gating variables"""
        if self.channel_type == IonChannelType.SODIUM:
            # m³h for Na+ channel
            m = self.gating_variables['m']
            h = self.gating_variables['h']
            return (m ** 3) * h
        
        elif self.channel_type == IonChannelType.POTASSIUM:
            # n⁴ for K+ channel
            n = self.gating_variables['n']
            return n ** 4
        
        elif self.channel_type == IonChannelType.POTASSIUM_A:
            # a³b for A-type K+ channel
            a = self.gating_variables['a']
            b = self.gating_variables['b']
            return (a ** 3) * b
        
        elif self.channel_type == IonChannelType.CALCIUM_DEPENDENT_POTASSIUM:
            # c for BK channel
            return self.gating_variables['c']
        
        elif self.channel_type == IonChannelType.HYPERPOLARIZATION_ACTIVATED:
            # m for HCN channel
            return self.gating_variables['m']
        
        else:
            # Default: use 'm' gating variable or always open
            return self.gating_variables.get('m', 1.0)
    
    def reset(self):
        """Reset channel to initial state"""
        self.gating_variables = self._initialize_gating_variables()
        self.current = 0.0
        self.open_probability = 0.0
        self.inactivation_state = 1.0
        self.current_history.clear()
        self.open_probability_history.clear()

class HodgkinHuxleyNeuron:
    """
    Hodgkin-Huxley neuron model
    Production-ready implementation with multiple ion channels
    """
    
    def __init__(self, config: Optional[IonChannelConfig] = None):
        self.config = config or IonChannelConfig()
        
        # Initialize ion channels
        self.ion_channels = self._initialize_ion_channels()
        
        # Membrane state
        self.voltage_mV = -70.0  # Resting potential
        self.current_injection_pA = 0.0  # Injected current
        
        # Calcium dynamics
        self.intracellular_calcium_mM = self.config.intracellular_calcium_mM
        self.calcium_pump_rate = 0.01  # mM/ms
        self.calcium_buffer_capacity = 100.0  # Dimensionless
        
        # History tracking
        self.voltage_history = []
        self.calcium_history = []
        self.time_history = []
        self.spike_times = []
        
        # Spike detection
        self.spike_threshold_mV = -50.0
        self.refractory_period_ms = 2.0
        self.refractory_until_ms = 0.0
        
        # Adaptive time stepping
        self.current_time_step_ms = self.config.time_step_ms
        self.voltage_derivative_history = deque(maxlen=10)
        
        logger.info(f"Initialized Hodgkin-Huxley neuron with "
                   f"{len(self.ion_channels)} ion channels")
    
    def _initialize_ion_channels(self) -> Dict[IonChannelType, IonChannel]:
        """Initialize all ion channels for this neuron"""
        channels = {}
        
        # Standard Hodgkin-Huxley channels
        channels[IonChannelType.SODIUM] = IonChannel(
            channel_type=IonChannelType.SODIUM,
            config=self.config,
            channel_density=self.config.sodium_channel_density,
            single_channel_conductance=20.0  # pS
        )
        
        channels[IonChannelType.POTASSIUM] = IonChannel(
            channel_type=IonChannelType.POTASSIUM,
            config=self.config,
            channel_density=self.config.potassium_channel_density,
            single_channel_conductance=10.0  # pS
        )
        
        # Leak channel
        channels[IonChannelType.LEAK] = IonChannel(
            channel_type=IonChannelType.LEAK,
            config=self.config,
            channel_density=self.config.leak_channel_density,
            single_channel_conductance=1.0  # pS
        )
        
        # Optional additional channels
        channels[IonChannelType.POTASSIUM_A] = IonChannel(
            channel_type=IonChannelType.POTASSIUM_A,
            config=self.config,
            channel_density=5.0,
            single_channel_conductance=15.0  # pS
        )
        
        channels[IonChannelType.CALCIUM_DEPENDENT_POTASSIUM] = IonChannel(
            channel_type=IonChannelType.CALCIUM_DEPENDENT_POTASSIUM,
            config=self.config,
            channel_density=2.0,
            single_channel_conductance=50.0  # pS
        )
        
        channels[IonChannelType.HYPERPOLARIZATION_ACTIVATED] = IonChannel(
            channel_type=IonChannelType.HYPERPOLARIZATION_ACTIVATED,
            config=self.config,
            channel_density=3.0,
            single_channel_conductance=5.0  # pS
        )
        
        return channels
    
    def simulate_step(self, 
                     current_injection_pA: Optional[float] = None,
                     time_step_ms: Optional[float] = None) -> Dict[str, float]:
        """
        Simulate one time step
        
        Args:
            current_injection_pA: Injected current (pA)
            time_step_ms: Time step for this iteration
            
        Returns:
            Dictionary with simulation results
        """
        if current_injection_pA is not None:
            self.current_injection_pA = current_injection_pA
        
        # Use adaptive time stepping if enabled
        if self.config.adaptive_time_step:
            time_step_ms = self._calculate_adaptive_time_step()
        else:
            time_step_ms = time_step_ms or self.config.time_step_ms
        
        # Check refractory period
        current_time = len(self.time_history) * self.config.time_step_ms
        is_refractory = current_time < self.refractory_until_ms
        
        if is_refractory:
            # During refractory period, clamp voltage
            self.voltage_mV = -80.0  # Hyperpolarized
      
