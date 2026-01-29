"""
Unit tests for IonChannelDynamics
"""
import pytest
import numpy as np
from src.virtual_bio.ion_channel_dynamics import IonChannelDynamics


class TestIonChannelDynamics:
    """Test suite for IonChannelDynamics class"""
    
    @pytest.fixture
    def ion_channel(self):
        """Create an IonChannelDynamics instance"""
        return IonChannelDynamics(
            num_channels=100,
            voltage_range=(-80, 40),
            dt=0.01
        )
    
    def test_initialization(self, ion_channel):
        """Test ion channel initialization"""
        assert ion_channel.num_channels == 100
        assert ion_channel.voltage_range == (-80, 40)
        assert ion_channel.dt == 0.01
    
    def test_simulate_voltage_step(self, ion_channel):
        """Test voltage step simulation"""
        initial_voltage = -80.0
        final_voltage = 0.0
        duration = 10.0
        
        result = ion_channel.simulate_voltage_step(
            initial_voltage, 
            final_voltage, 
            duration
        )
        
        assert result is not None
        assert 'voltage' in result
        assert 'current' in result
        assert 'gating_variables' in result
        assert len(result['voltage']) > 0
    
    def test_compute_conductance(self, ion_channel):
        """Test conductance computation"""
        ion_channel.set_voltage(-80.0)
        conductance = ion_channel.compute_conductance()
        
        assert conductance >= 0
        assert isinstance(conductance, float)
    
    def test_simulate_action_potential(self, ion_channel):
        """Test action potential simulation"""
        result = ion_channel.simulate_action_potential()
        
        assert result is not None
        assert 'voltage' in result
        assert 'current' in result
        assert len(result['voltage']) > 0
        # Check for characteristic spike
        assert np.max(result['voltage']) > 0
        assert np.min(result['voltage']) < -70
    
    def test_update_gating_variables(self, ion_channel):
        """Test gating variables update"""
        ion_channel.set_voltage(0.0)
        ion_channel.update_gating_variables()
        
        assert ion_channel.gating_variables is not None
        assert ion_channel.gating_variables.shape == (ion_channel.num_channels, 3)
    
    def test_compute_current(self, ion_channel):
        """Test current computation"""
        ion_channel.set_voltage(0.0)
        current = ion_channel.compute_current()
        
        assert isinstance(current, float)
    
    def test_reset_channels(self, ion_channel):
        """Test channel reset"""
        ion_channel.simulate_voltage_step(-80.0, 0.0, 10.0)
        ion_channel.reset_channels()
        
        assert ion_channel.current_voltage == -80.0
        assert ion_channel.time == 0.0