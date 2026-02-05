#!/usr/bin/env python3
"""
Training command line interface.
"""

import argparse
import yaml
import logging
from datetime import datetime


def train_main(args):
    """Main training function."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load curriculum if specified
    curriculum_config = {}
    if args.curriculum:
        with open(args.curriculum, "r") as f:
            curriculum_config = yaml.safe_load(f)

    logger.info(f"Starting consciousness training for {args.epochs} epochs")

    try:
        from src.training.consciousness_curriculum import ConsciousnessCurriculum

        # Initialize curriculum
        curriculum = ConsciousnessCurriculum(config=curriculum_config)

        # Run training
        results = []
        for epoch in range(args.epochs):
            epoch_result = curriculum.train_epoch(epoch)
            results.append(epoch_result)

            if epoch % 100 == 0:
                logger.info(
                    f"Epoch {epoch}: Consciousness level = {epoch_result.get('consciousness_level', 0):.3f}"
                )

                # Save checkpoint
                if epoch % 1000 == 0:
                    checkpoint_file = f"checkpoint_epoch_{epoch}.pt"
                    curriculum.save_checkpoint(checkpoint_file)
                    logger.info(f"Checkpoint saved to {checkpoint_file}")

        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model = f"consciousness_model_{timestamp}.pt"
        curriculum.save_checkpoint(final_model)

        logger.info(f"Training complete. Model saved to {final_model}")

        # Print summary
        if results:
            final_level = results[-1].get("consciousness_level", 0)
            logger.info(f"Final consciousness level: {final_level:.3f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train consciousness models")
    parser.add_argument("--curriculum", help="Training curriculum file")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    args = parser.parse_args()
    train_main(args)
