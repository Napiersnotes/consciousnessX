#!/usr/bin/env python3
"""
Simulation command line interface.
"""

import argparse
import yaml
import numpy as np
from datetime import datetime
import logging

def simulate_main(args):
    """Main simulation function."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    logger.info(f"Starting consciousness simulation for {args.duration} seconds")
    
    # Import and run simulation
    try:
        from src.core.microtubule_simulator import MicrotubuleSimulator
        from src.core.penrose_gravitational_collapse import PenroseCollapse
        from src.virtual_bio.ion_channel_dynamics import IonChannelDynamics
        
        # Initialize simulators
        mt_sim = MicrotubuleSimulator()
        penrose_sim = PenroseCollapse()
        ion_sim = IonChannelDynamics()
        
        # Run simulation
        time_steps = int(args.duration / 0.0001)
        results = []
        
        for step in range(time_steps):
            # Update all simulations
            mt_state = mt_sim.update()
            collapse_state = penrose_sim.compute_collapse(mt_state)
            ion_state = ion_sim.update(collapse_state)
            
            # Combine results
            combined_state = {
                'microtubule': mt_state,
                'collapse': collapse_state,
                'ion_channels': ion_state,
                'timestamp': step * 0.0001
            }
            results.append(combined_state)
            
            if step % 1000 == 0:
                logger.info(f"Step {step}/{time_steps}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"simulation_results_{timestamp}.npz"
        np.savez(output_file, results=np.array(results, dtype=object))
        
        logger.info(f"Simulation complete. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consciousness simulations")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--duration", type=float, default=1.0, help="Simulation duration (s)")
    args = parser.parse_args()
    simulate_main(args)
