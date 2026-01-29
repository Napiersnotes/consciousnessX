import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="ConsciousnessX CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Simulation command
    subparsers.add_parser("simulate", help="Run consciousness simulation")
    
    # Visualization command  
    subparsers.add_parser("visualize", help="Launch dashboard")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        from src.core.microtubule_simulator import MicrotubuleSimulator
        simulator = MicrotubuleSimulator()
        results = simulator.simulate()
        print(f"Simulation complete. Φ = {results['summary']['integrated_information']:.4f}")
    
    elif args.command == "visualize":
        from src.visualization.consciousness_dashboard import ConsciousnessDashboard
        dashboard = ConsciousnessDashboard()
        dashboard.run()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
