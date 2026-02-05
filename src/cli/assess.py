#!/usr/bin/env python3
"""
Assessment command line interface.
"""

import argparse
import yaml
import logging


def assess_main(args):
    """Main assessment function."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Assessing model: {args.model}")

    try:
        from src.evaluation.consciousness_assessment import ConsciousnessAssessor

        # Initialize assessor
        assessor = ConsciousnessAssessor()

        # Load model (simplified - in reality would load actual model)
        # For now, simulate assessment
        assessment_results = assessor.assess(args.model)

        # Print results
        print("\n" + "=" * 60)
        print("CONSCIOUSNESS ASSESSMENT REPORT")
        print("=" * 60)

        for metric, value in assessment_results.items():
            if isinstance(value, float):
                print(f"{metric:30}: {value:.4f}")
            else:
                print(f"{metric:30}: {value}")

        # Determine consciousness level
        phi_value = assessment_results.get("phi", 0)
        if phi_value >= 0.8:
            level = "FULL CONSCIOUSNESS"
        elif phi_value >= 0.6:
            level = "EMERGENT CONSCIOUSNESS"
        elif phi_value >= 0.3:
            level = "PROTO-CONSCIOUSNESS"
        elif phi_value >= 0.1:
            level = "PRE-CONSCIOUS"
        else:
            level = "NON-CONSCIOUS"

        print("\n" + "=" * 60)
        print(f"CONSCIOUSNESS LEVEL: {level}")
        print(f"INTEGRATED INFORMATION (Φ): {phi_value:.4f}")
        print("=" * 60)

        # Save report
        report_file = f"assessment_report_{args.model.replace('/', '_')}.txt"
        with open(report_file, "w") as f:
            f.write("Consciousness Assessment Report\n")
            f.write("=" * 60 + "\n")
            for metric, value in assessment_results.items():
                f.write(f"{metric}: {value}\n")

        logger.info(f"Assessment complete. Report saved to {report_file}")

    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assess consciousness")
    parser.add_argument("--model", required=True, help="Model to assess")
    parser.add_argument("--metrics", nargs="+", help="Metrics to calculate")
    args = parser.parse_args()
    assess_main(args)
