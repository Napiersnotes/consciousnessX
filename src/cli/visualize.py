#!/usr/bin/env python3
"""
Visualization command line interface.
"""

import argparse
import numpy as np
import logging
import webbrowser
from pathlib import Path


def visualize_main(args):
    """Main visualization function."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.dashboard:
        logger.info("Launching consciousness dashboard...")

        # Import dashboard
        from src.visualization.consciousness_dashboard import launch_dashboard

        # Load data if provided
        data = None
        if args.data:
            data = np.load(args.data, allow_pickle=True)

        # Launch dashboard
        launch_dashboard(data=data)
    else:
        # Basic command line visualization
        if args.data:
            data = np.load(args.data, allow_pickle=True)
            print(f"Data shape: {data['results'].shape}")

            # Show basic statistics
            results = data["results"]
            print(f"Number of time steps: {len(results)}")

            # Extract some sample metrics
            if len(results) > 0:
                sample = results[0]
                print("\nSample state keys:", list(sample.keys()))

                if "microtubule" in sample:
                    print(f"Microtubule states: {sample['microtubule']['states'].shape}")

                if "collapse" in sample:
                    print(f"Collapse probability: {sample['collapse']['collapse_probability']:.4f}")

        logger.info("Use --dashboard flag for interactive visualization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize simulation results")
    parser.add_argument("--data", help="Data file to visualize")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard")
    args = parser.parse_args()
    visualize_main(args)
