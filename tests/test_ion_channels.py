#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ion_channel_dynamics.py
Production-ready test suite with comprehensive coverage
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from virtual_bio.ion_channel_dynamics import (
    IonChannelConfig,
    IonChannel,
    IonChannelType,
    HodgkinHuxleyNeuron,
    ChannelGating
)

class TestIonChannelConfig(unittest.TestCase):
    """Test ion channel configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = IonChannelConfig()
        
        self.assertEqual(config.temperature_celsius, 37.0)
        self.assertAlmostEqual(config.time_step_ms, 0.01)
        self.assertAlmostEqual(config.extracellular_sodium_mM, 145.0)
        self.assertAlmostEqual(config.intracellular_sodium_mM, 15.0)
        
        # Nernst potentials should be calculated
        self.assertIsNotNone(config.e_na)
        self.assertIsNotNone(config.e_k)
        self.assertIsNotNone(config.e_ca)
        self.assertIsNotNone(config.e_cl)
        
        # Temperatures should be converted
        self.assertAlmostEqual(config.temperature_kelvin, 310.15)
    
    def test_nernst_potential_calculation(self):
        """Test Nernst potential calculation"""
        config = IonChannelConfig(
            temperature_celsius=20.0,
            extracellular_potassium_mM=10.0,
            intracellular_potassium_mM=100.0
        )
        
        # K+ Nernst potential at 20°C
        # E_K = (RT/zF) * ln([K+]out/[K+]in)
        # R = 8.314, T = 293.15K, z = 1, F = 96485
        # Should be negative since [K+]in > [K+]out
        
        self.assertLess(config.e_k, 0)
        
        # Rough calculation: E_K ≈ -58 mV * log10([K+]in/[K+]out) at 20°C
        expected = -58 * np.log10(100.0/10.0)
        self.assertAlmostEqual(config.e_k, expected, delta=5.0)
    
    def test_invalid_config(self):
        """Test invalid configuration raises errors"""
        # Temperature too low
        with self.assertRaises(ValueError):
            config = IonChannelConfig(temperature_celsius=-300)

class TestIonChannel(unittest.TestCase):
    """Test ion channel base class"""
    
    def setUp(self):
        self.config = IonChannelConfig()
        self.na_channel = IonChannel(
            channel_type=IonChannelType.SODIUM,
            config=self.config,
            channel_density=50.0,
            single_channel_conductance=20.0
        )
    
    def test_channel_initialization(self):
        """Test channel initialization"""
        self.assertEqual(self.na_channel.channel_type, IonChannelType.SODIUM)
        self.assertEqual(self.na_channel.channel_density, 50.0)
        self.assertEqual(self.na_channel.single_channel_conductance, 20.0)
        
        # Should have gating variables
        self.assertIn('m', self.na_channel.gating_variables)
        self.assertIn('h', self.na_channel.gating_variables)
        
        # Initial state
        self.assertEqual(self.na_channel.current, 0.0)
        self.assertEqual(self.na_channel.open_probability, 0.0)
        self.assertEqual(self.na_channel.inactivation_state, 1.0)
    
    def test_total_conductance_calculation(self):
        """Test total conductance calculation"""
        # Total conductance = single channel conductance * density * area
        single_conductance_ps = 20.0  # pS
        density = 50.0  # channels/μm²
        area = self.config.membrane_area_um2  # μm²
        
        # Expected: 20 pS * 50/μm² * 1000 μm² = 1,000,000 pS = 1000 nS
        expected_conductance = single_conductance_ps * density * area * 1e-3  # nS
        
        self.assertAlmostEqual(self.na_channel.total_conductance, 
                             expected_conductance, 
                             delta=expected_conductance * 0.01)
    
    def test_current_calculation(self):
        """Test current calculation"""
        # At resting potential (-70 mV), sodium current should be inward (negative)
        voltage = -70.0
        current = self.na_channel.calculate_current(voltage, time_step_ms=0.01)
        
        # Sodium reversal potential is positive (~+55 mV)
        # So at -70 mV, current should be negative (inward)
        self.assertLess(current, 0.0)
        
        # Current history should be updated
        self.assertEqual(len(self.na_channel.current_history), 1)
        self.assertEqual(len(self.na_channel.open_probability_history), 1)
    
    def test_gating_variable_dynamics(self):
        """Test gating variable updates"""
        initial_m = self.na_channel.gating_variables['m']
        initial_h = self.na_channel.gating_variables['h']
        
        # Calculate current at resting potential
        self.na_channel.calculate_current(-70.0, time_step_ms=1.0)
        
        final_m = self.na_channel.gating_variables['m']
        final_h = self.na_channel.gating_variables['h']
        
        # Gating variables should change
        self.assertNotEqual(initial_m, final_m)
        self.assertNotEqual(initial_h, final_h)
        
        # Values should remain between 0 and 1
        self.assertGreaterEqual(final_m, 0.0)
        self.assertLessEqual(final_m, 1.0)
        self.assertGreaterEqual(final_h, 0.0)
        self.assertLessEqual(final_h, 1.0)
    
    def test_open_probability_calculation(self):
        """Test open probability calculation"""
        # Set gating variables to known values
        self.na_channel.gating_variables['m'] = 0.5
        self.na_channel.gating_variables['h'] = 0.5
        
        # For Na+ channel: P_open = m³h
        expected = 0.5**3 * 0.5 = 0.0625
        
        # Calculate current to trigger open probability calculation
        self.na_channel.calculate_current(-70.0, time_step_ms=0.01)
        
        self.assertAlmostEqual(self.na_channel.open_probability, expected, delta=1e-10)
    
    def test_channel_reset(self):
        """Test channel reset functionality"""
        # Modify channel state
        self.na_channel.calculate_current(-70.0, time_step_ms=1.0)
        
        # Store modified values
        modified_m = self.na_channel.gating_variables['m']
        modified_current = self.na_channel.current
        modified_history_len = len(self.na_channel.current_history)
        
        # Reset channel
        self.na_channel.reset()
        
        # Should return to initial state
        self.assertEqual(self.na_channel.current, 0.0)
        self.assertEqual(self.na_channel.open_probability, 0.0)
        self.assertEqual(self.na_channel.inactivation_state, 1.0)
        
        # History should be cleared
        self.assertEqual(len(self.na_channel.current_history), 0)
        self.assertEqual(len(self.na_channel.open_probability_history), 0)
        
        # Gating variables should be reinitialized
        self.assertNotEqual(self.na_channel.gating_variables['m'], modified_m)
        self.assertEqual(self.na_channel.gating_variables['m'], 0.05)  # Initial value

class TestHodgkinHuxleyNeuron(unittest.TestCase):
    """Test Hodgkin-Huxley neuron model"""
    
    def setUp(self):
        self.neuron = HodgkinHuxleyNeuron()
    
    def test_neuron_initialization(self):
        """Test neuron initialization"""
        self.assertEqual(len(self.neuron.ion_channels), 6)  # Default channels
        self.assertAlmostEqual(self.neuron.voltage_mV, -70.0)  # Resting potential
        
        # Should have required channels
        self.assertIn(IonChannelType.SODIUM, self.neuron.ion_channels)
        self.assertIn(IonChannelType.POTASSIUM, self.neuron.ion_channels)
        self.assertIn(IonChannelType.LEAK, self.neuron.ion_channels)
        
        # History should be empty
        self.assertEqual(len(self.neuron.voltage_history), 0)
        self.assertEqual(len(self.neuron.calcium_history), 0)
        self.assertEqual(len(self.neuron.time_history), 0)
        self.assertEqual(len(self.neuron.spike_times), 0)
    
    def test_single_step_simulation(self):
        """Test single simulation step"""
        # Initial state
        initial_voltage = self.neuron.voltage_mV
        initial_calcium = self.neuron.intracellular_calcium_mM
        
        # Simulate one step with no current injection
        result = self.neuron.simulate_step(current_injection_pA=0.0)
        
        # Should return results dictionary
        self.assertIsInstance(result, dict)
        self.assertIn('voltage_mV', result)
        self.assertIn('calcium_mM', result)
        self.assertIn('time_ms', result)
        self.assertIn('dvdt_mV_ms', result)
        self.assertIn('spike_detected', result)
        
        # Voltage should change slightly (towards resting potential)
        self.assertNotEqual(self.neuron.voltage_mV, initial_voltage)
        
        # Calcium might change
        self.assertNotEqual(self.neuron.intracellular_calcium_mM, initial_calcium)
        
        # History should be updated
        self.assertEqual(len(self.neuron.voltage_history), 1)
        self.assertEqual(len(self.neuron.calcium_history), 1)
        self.assertEqual(len(self.neuron.time_history), 1)
        
        # No spike should be detected at resting potential
        self.assertFalse(result['spike_detected'])
    
    def test_current_injection_response(self):
        """Test response to current injection"""
        # Inject depolarizing current
        result = self.neuron.simulate_step(current_injection_pA=200.0)
        
        # Voltage should increase (become less negative)
        self.assertGreater(self.neuron.voltage_mV, -70.0)
        
        # dvdt should be positive (depolarizing)
        self.assertGreater(result['dvdt_mV_ms'], 0.0)
    
    def test_action_potential_generation(self):
        """Test action potential generation"""
        # Simulate with sustained depolarizing current
        spike_detected = False
        
        for i in range(1000):  # Simulate 1000 steps (10 ms at 0.01 ms steps)
            result = self.neuron.simulate_step(current_injection_pA=500.0)
            
            if result['spike_detected']:
                spike_detected = True
                break
        
        # With sufficient current, should generate a spike
        self.assertTrue(spike_detected)
        self.assertGreater(len(self.neuron.spike_times), 0)
        
        # Voltage should show characteristic spike shape
        voltages = np.array(self.neuron.voltage_history)
        self.assertGreater(np.max(voltages), -50.0)  # Should exceed threshold
        self.assertLess(np.min(voltages[-10:]), -70.0)  # Should repolarize
    
    def test_refractory_period(self):
        """Test refractory period after spike"""
        # Generate a spike
        for i in range(500):
            result = self.neuron.simulate_step(current_injection_pA=500.0)
            if result['spike_detected']:
                break
        
        # Store time of first spike
        first_spike_time = self.neuron.spike_times[0] if self.neuron.spike_times else 0
        
        # Continue simulation with current injection
        # Should not spike again immediately due to refractory period
        second_spike_detected = False
        for i in range(100):  # 1 ms after first spike
            result = self.neuron.simulate_step(current_injection_pA=500.0)
            if result['spike_detected'] and result['time_ms'] > first_spike_time:
                second_spike_detected = True
                break
        
        # Should not detect second spike within refractory period
        self.assertFalse(second_spike_detected)
        
        # Check refractory flag in results
        if len(self.neuron.voltage_history) > 0:
            # Get recent results
            recent_time = self.neuron.time_history[-1]
            is_refractory = recent_time < self.neuron.refractory_until_ms
            
            if is_refractory:
                # During refractory period, voltage should be hyperpolarized
                self.assertLess(self.neuron.voltage_mV, -70.0)
    
    def test_calcium_dynamics(self):
        """Test calcium concentration dynamics"""
        initial_calcium = self.neuron.intracellular_calcium_mM
        
        # Simulate with voltage that activates calcium channels
        for i in range(100):
            self.neuron.simulate_step(current_injection_pA=200.0)
        
        final_calcium = self.neuron.intracellular_calcium_mM
        
        # Calcium concentration should change
        self.assertNotEqual(final_calcium, initial_calcium)
        
        # Calcium should remain non-negative
        self.assertGreaterEqual(final_calcium, 0.0)
        
        # Calcium history should be recorded
        self.assertEqual(len(self.neuron.calcium_history), 100)
    
    def test_neuron_reset(self):
        """Test neuron reset functionality"""
        # Modify neuron state
        for i in range(100):
            self.neuron.simulate_step(current_injection_pA=100.0)
        
        # Store modified values
        modified_voltage = self.neuron.voltage_mV
        modified_calcium = self.neuron.intracellular_calcium_mM
        modified_history_len = len(self.neuron.voltage_history)
        
        # Reset neuron
        self.neuron.reset()
        
        # Should return to initial state
        self.assertEqual(self.neuron.voltage_mV, -70.0)
        self.assertEqual(self.neuron.current_injection_pA, 0.0)
        self.assertEqual(self.neuron.intracellular_calcium_mM, 
                        self.neuron.config.intracellular_calcium_mM)
        self.assertEqual(self.neuron.refractory_until_ms, 0.0)
        
        # History should be cleared
        self.assertEqual(len(self.neuron.voltage_history), 0)
        self.assertEqual(len(self.neuron.calcium_history), 0)
        self.assertEqual(len(self.neuron.time_history), 0)
        self.assertEqual(len(self.neuron.spike_times), 0)
        
        # Channels should be reset
        for channel in self.neuron.ion_channels.values():
            self.assertEqual(channel.current, 0.0)
            self.assertEqual(len(channel.current_history), 0)
    
    def test_full_simulation(self):
        """Test complete simulation run"""
        # Define a current protocol
        def current_protocol(t_ms):
            if 10.0 <= t_ms < 60.0:  # 50 ms current pulse
                return 200.0  # pA
            return 0.0
        
        # Run simulation
        results = self.neuron.run_simulation(
            duration_ms=100.0,
            current_protocol=current_protocol
        )
        
        # Should return results dictionary
        self.assertIsInstance(results, dict)
        
        # Should have expected keys
        expected_keys = ['time_ms', 'voltage_mV', 'calcium_mM', 
                        'spike_times_ms', 'channel_currents', 'num_steps', 'config']
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Arrays should have correct length
        self.assertEqual(len(results['time_ms']), results['num_steps'])
        self.assertEqual(len(results['voltage_mV']), results['num_steps'])
        self.assertEqual(len(results['calcium_mM']), results['num_steps'])
        
        # Spike times should be within simulation duration
        for spike_time in results['spike_times_ms']:
            self.assertGreaterEqual(spike_time, 0.0)
            self.assertLessEqual(spike_time, 100.0)
        
        # Channel currents should be recorded
        self.assertIsInstance(results['channel_currents'], dict)
        self.assertGreater(len(results['channel_currents']), 0)

class TestAdaptiveTimeStepping(unittest.TestCase):
    """Test adaptive time stepping functionality"""
    
    def test_adaptive_time_step_calculation(self):
        """Test adaptive time step calculation"""
        config = IonChannelConfig(
            adaptive_time_step=True,
            min_time_step_ms=0.001,
            max_time_step_ms=0.1,
            time_step_ms=0.01
        )
        
        neuron = HodgkinHuxleyNeuron(config)
        
        # Initially should use configured time step
        self.assertAlmostEqual(neuron.current_time_step_ms, config.time_step_ms)
        
        # Simulate with varying dynamics to trigger adaptive stepping
        for i in range(10):
            if i < 5:
                current = 0.0  # Resting
            else:
                current = 500.0  # Depolarizing
            
            result = neuron.simulate_step(current_injection_pA=current)
            
            # Time step should remain within bounds
            self.assertGreaterEqual(result['time_step_ms'], config.min_time_step_ms)
            self.assertLessEqual(result['time_step_ms'], config.max_time_step_ms)
        
        # Time step should have changed from initial value
        self.assertNotEqual(neuron.current_time_step_ms, config.time_step_ms)

# Performance tests (only run if requested)
@unittest.skipUnless(os.getenv('RUN_PERFORMANCE_TESTS'), 
                    "Performance tests disabled by default")
class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_large_network_performance(self):
        """Test performance with network of neurons"""
        import time
        
        from virtual_bio.ion_channel_dynamics import IonChannelNetwork
        
        # Create network
        network = IonChannelNetwork(
            num_neurons=100,
            num_compartments=1
        )
        
        # Run simulation
        start_time = time.time()
        
        results = network.simulate_network(
            duration_ms=100.0,
            input_currents=None
        )
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        max_time = 30.0  # 30 seconds
        self.assertLess(elapsed, max_time, 
                       f"Network simulation took {elapsed:.2f}s, expected < {max_time}s")
        
        # Should have results
        self.assertIn('statistics', results)
        self.assertGreater(results['statistics']['total_spikes'], 0)
        
        print(f"100-neuron network simulation: {elapsed:.2f}s")
        print(f"Total spikes: {results['statistics']['total_spikes']}")
        print(f"Mean firing rate: {results['statistics']['mean_firing_rate_hz']:.2f} Hz")

if __name__ == '__main__':
    unittest.main(verbosity=2)
