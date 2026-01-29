#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for microtubule_simulator.py
Production-ready test suite with comprehensive coverage
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.microtubule_simulator import (
    MicrotubuleSimulator,
    MicrotubuleConfig,
    MicrotubuleLattice,
    QuantumCoherenceSimulator,
    QuantumState
)

class TestMicrotubuleConfig(unittest.TestCase):
    """Test configuration class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = MicrotubuleConfig()
        
        self.assertEqual(config.num_protofilaments, 13)
        self.assertEqual(config.num_tubulins_per_filament, 100)
        self.assertAlmostEqual(config.tubulin_spacing_nm, 8.0)
        self.assertAlmostEqual(config.temperature_k, 310.0)
        self.assertAlmostEqual(config.hbar, 1.054571817e-34)
        
        # Should auto-detect device
        self.assertIn(config.device, ["cpu", "cuda"])
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MicrotubuleConfig(
            num_protofilaments=10,
            num_tubulins_per_filament=50,
            simulation_duration_s=0.1,
            use_gpu=False
        )
        
        self.assertEqual(config.num_protofilaments, 10)
        self.assertEqual(config.num_tubulins_per_filament, 50)
        self.assertAlmostEqual(config.simulation_duration_s, 0.1)
        self.assertEqual(config.use_gpu, False)
        self.assertEqual(config.device, "cpu")
    
    def test_invalid_config(self):
        """Test invalid configuration raises errors"""
        with self.assertRaises(ValueError):
            MicrotubuleConfig(precision="invalid")
        
        with self.assertRaises(ValueError):
            MicrotubuleConfig(num_protofilaments=0)

class TestMicrotubuleLattice(unittest.TestCase):
    """Test microtubule lattice creation"""
    
    def setUp(self):
        self.config = MicrotubuleConfig(
            num_protofilaments=3,  # Small for testing
            num_tubulins_per_filament=5
        )
        self.lattice = MicrotubuleLattice(self.config)
    
    def test_lattice_initialization(self):
        """Test lattice is initialized correctly"""
        self.assertEqual(self.lattice.num_tubulins, 15)  # 3 * 5
        
        # Check positions array shape
        self.assertEqual(self.lattice.positions.shape, (15, 3))
        
        # Check adjacency matrix shape
        self.assertEqual(self.lattice.adjacency_matrix.shape, (15, 15))
        
        # Check quantum states
        self.assertEqual(len(self.lattice.quantum_states), 15)
        
        # Check dipole moments
        self.assertEqual(self.lattice.dipole_moments.shape, (15, 3))
    
    def test_lattice_positions(self):
        """Test lattice positions are reasonable"""
        positions = self.lattice.positions
        
        # All positions should be finite
        self.assertTrue(np.all(np.isfinite(positions)))
        
        # Z-coordinates should increase
        z_coords = positions[:, 2]
        for i in range(len(z_coords) - 1):
            self.assertLessEqual(z_coords[i], z_coords[i + 1])
    
    def test_adjacency_matrix(self):
        """Test adjacency matrix properties"""
        adj = self.lattice.adjacency_matrix
        
        # Should be symmetric
        self.assertTrue((adj != adj.T).nnz == 0)
        
        # No self-connections
        self.assertEqual(adj.diagonal().sum(), 0)
        
        # Should have some connections
        self.assertGreater(adj.nnz, 0)

class TestQuantumCoherenceSimulator(unittest.TestCase):
    """Test quantum coherence calculations"""
    
    def setUp(self):
        self.config = MicrotubuleConfig()
        self.simulator = QuantumCoherenceSimulator(self.config)
    
    def test_decoherence_rates(self):
        """Test decoherence rate calculation"""
        rates = self.simulator.decoherence_rates
        
        # Should have expected keys
        expected_keys = ['thermal', 'viscous', 'electromagnetic', 'total']
        for key in expected_keys:
            self.assertIn(key, rates)
        
        # Rates should be positive
        for rate in rates.values():
            self.assertGreater(rate, 0)
        
        # Total should be sum of components
        components = sum([rates[k] for k in ['thermal', 'viscous', 'electromagnetic']])
        self.assertAlmostEqual(rates['total'], components, delta=components*1e-10)
    
    def test_calculate_coherence_time(self):
        """Test coherence time calculation"""
        # Test with typical energy
        energy = 1e-30  # 1e-30 J
        collapse_time = self.simulator.calculate_coherence_time(energy)
        
        self.assertGreater(collapse_time, 0)
        self.assertLess(collapse_time, 10)  # Should be less than 10 seconds
        
        # Test with zero energy (should return infinity)
        collapse_time = self.simulator.calculate_coherence_time(0)
        self.assertTrue(np.isinf(collapse_time))
        
        # Test with negative energy (absolute value should be used)
        collapse_time_neg = self.simulator.calculate_coherence_time(-1e-30)
        collapse_time_pos = self.simulator.calculate_coherence_time(1e-30)
        self.assertAlmostEqual(collapse_time_neg, collapse_time_pos)

class TestMicrotubuleSimulator(unittest.TestCase):
    """Test main simulator"""
    
    def setUp(self):
        # Use small configuration for faster tests
        self.config = MicrotubuleConfig(
            num_protofilaments=3,
            num_tubulins_per_filament=10,
            simulation_duration_s=0.01,  # 10 ms
            time_step_s=1e-4            # 0.1 ms
        )
        self.simulator = MicrotubuleSimulator(self.config)
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertIsNotNone(self.simulator.lattice)
        self.assertIsNotNone(self.simulator.coherence_sim)
        self.assertEqual(self.simulator.current_time, 0.0)
        self.assertEqual(self.simulator.steps_completed, 0)
    
    def test_simulation_step(self):
        """Test single simulation step"""
        initial_states = self.simulator.lattice.quantum_states.copy()
        
        # Run one step
        self.simulator._simulation_step(0)
        
        # Time should advance
        self.assertAlmostEqual(self.simulator.current_time, self.config.time_step_s)
        
        # Steps should increment
        self.assertEqual(self.simulator.steps_completed, 1)
        
        # History should be recorded
        self.assertEqual(len(self.simulator.time_history), 1)
        self.assertEqual(len(self.simulator.state_history), 1)
        self.assertEqual(len(self.simulator.metrics_history), 1)
        
        # States might change (but not guaranteed)
        final_states = self.simulator.lattice.quantum_states
        # At least test they have same shape
        self.assertEqual(initial_states.shape, final_states.shape)
    
    def test_full_simulation(self):
        """Test complete simulation"""
        results = self.simulator.simulate()
        
        # Should have expected keys
        expected_keys = ['simulation_time', 'quantum_states', 'metrics', 
                        'final_state', 'positions', 'adjacency_matrix', 'summary', 'config']
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check simulation completed expected number of steps
        expected_steps = int(self.config.simulation_duration_s / self.config.time_step_s)
        self.assertEqual(results['summary']['total_steps'], expected_steps)
        
        # Check time arrays
        time_array = results['simulation_time']
        self.assertEqual(len(time_array), expected_steps)
        self.assertAlmostEqual(time_array[-1], self.config.simulation_duration_s, 
                             delta=self.config.time_step_s*2)
        
        # Check summary statistics
        summary = results['summary']
        self.assertIn('final_coherence', summary)
        self.assertIn('integrated_information', summary)
        self.assertIn('consciousness_level', summary)
        
        # Coherence should be between 0 and 1
        self.assertGreaterEqual(summary['final_coherence'], 0)
        self.assertLessEqual(summary['final_coherence'], 1)
        
        # Φ should be non-negative
        self.assertGreaterEqual(summary['integrated_information'], 0)
    
    def test_calculate_integrated_information(self):
        """Test Φ calculation"""
        # Create artificial state history
        num_steps = 100
        num_tubulins = 30
        
        # Simple pattern: alternating blocks
        states = np.zeros((num_steps, num_tubulins))
        for i in range(num_steps):
            if i % 20 < 10:
                states[i, :num_tubulins//2] = 1  # First half active
            else:
                states[i, num_tubulins//2:] = 1  # Second half active
        
        self.simulator.state_history = list(states)
        
        phi = self.simulator._calculate_integrated_information()
        
        # Φ should be between 0 and 1 for this simple pattern
        self.assertGreaterEqual(phi, 0)
        self.assertLessEqual(phi, 1)
    
    def test_save_load_results(self):
        """Test saving and loading results"""
        # Run a small simulation
        results = self.simulator.simulate()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.simulator.save_results(tmp_path)
            
            # Load results
            loaded_results = MicrotubuleSimulator.load_results(tmp_path)
            
            # Check key structures are preserved
            self.assertIn('time_history', loaded_results)
            self.assertIn('state_history', loaded_results)
            self.assertIn('config', loaded_results)
            
            # Check arrays match
            original_time = np.array(results['simulation_time'])
            loaded_time = np.array(loaded_results['time_history'])
            
            np.testing.assert_array_almost_equal(original_time, loaded_time)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestConsciousnessAssessment(unittest.TestCase):
    """Test consciousness assessment functions"""
    
    def test_assess_consciousness_level(self):
        """Test consciousness level assessment"""
        from core.microtubule_simulator import MicrotubuleSimulator
        
        simulator = MicrotubuleSimulator()
        
        # Test different Φ values
        test_cases = [
            (0.05, "Pre-conscious"),
            (0.15, "Proto-conscious"),
            (0.4, "Emergent consciousness"),
            (0.8, "Full consciousness")
        ]
        
        for phi, expected_level in test_cases:
            level = simulator._assess_consciousness_level(phi)
            self.assertEqual(level, expected_level)
    
    def test_analyze_collapse_patterns(self):
        """Test collapse pattern analysis"""
        from core.microtubule_simulator import MicrotubuleSimulator
        
        simulator = MicrotubuleSimulator()
        
        # Create artificial state history with collapses
        num_steps = 100
        num_tubulins = 10
        
        states = np.zeros((num_steps, num_tubulins))
        
        # Add collapses at regular intervals
        for step in range(num_steps):
            if step % 20 == 0:  # Collapse every 20 steps
                # Change some states
                states[step, :5] = 1
            elif step > 0:
                # Carry forward previous state
                states[step] = states[step-1]
        
        simulator.state_history = list(states)
        simulator.config.time_step_s = 0.001  # 1 ms steps
        
        stats = simulator._analyze_collapse_patterns()
        
        # Should have expected keys
        expected_keys = ['mean_collapse_interval', 'std_collapse_interval',
                        'total_collapses', 'collapse_rate', 'regularity_index']
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # For our pattern, should have collapses
        self.assertGreater(stats['total_collapses'], 0)

# Performance tests (only run if requested)
@unittest.skipUnless(os.getenv('RUN_PERFORMANCE_TESTS'), 
                    "Performance tests disabled by default")
class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_large_simulation_performance(self):
        """Test performance with larger simulation"""
        import time
        
        config = MicrotubuleConfig(
            num_protofilaments=13,
            num_tubulins_per_filament=100,
            simulation_duration_s=0.1,
            time_step_s=1e-4
        )
        
        simulator = MicrotubuleSimulator(config)
        
        start_time = time.time()
        results = simulator.simulate()
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        # Adjust threshold based on hardware
        max_time = 30.0  # 30 seconds
        self.assertLess(elapsed, max_time, 
                       f"Simulation took {elapsed:.2f}s, expected < {max_time}s")
        
        # Check real-time factor
        simulated_time = results['summary']['total_time']
        real_time_factor = simulated_time / elapsed
        print(f"Real-time factor: {real_time_factor:.2f}")
        
        # Should be faster than real-time for useful simulation
        self.assertGreater(real_time_factor, 1.0,
                          f"Real-time factor {real_time_factor:.2f} <= 1.0")

if __name__ == '__main__':
    unittest.main(verbosity=2)
