#!/usr/bin/env python3
"""
Face Analysis App - Main Entry Point

Main script for running the face classification pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.detector import FaceDetector
from src.classifier import FaceClassifier
from src.reviewer import launch_reviewer
from src.checkpoint import CheckpointManager

# Setup logging
def setup_logging(log_level: str = "DEBUG"):
    """Setup logging configuration."""
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Main log file
    log_file = log_dir / f"processing_{Path(__file__).stem}.log"

    # Error log file
    error_log_file = log_dir / "errors.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Setup error log handler
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Add error handler to root logger
    logging.getLogger().addHandler(error_handler)

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Creating config from config.yaml.example")

        # Try to load example config
        example_config = Path("config.yaml.example")
        if example_config.exists():
            with open(example_config, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            raise FileNotFoundError(f"No config file found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")
    return config


def validate_config(config: dict) -> bool:
    """
    Validate configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_paths = ['source_directory', 'output_directory', 'database_path']

    for path_key in required_paths:
        if path_key not in config['paths']:
            logger.error(f"Missing required config: paths.{path_key}")
            return False

        path = config['paths'][path_key]

        if path_key == 'source_directory':
            if not Path(path).exists():
                logger.error(f"Source directory does not exist: {path}")
                return False

    return True


def mode_rebuild_embeddings(config: dict):
    """
    Rebuild sample embeddings from sample images.
    Uses the standalone generate_embeddings.py script.

    Args:
        config: Configuration dictionary
    """
    logger.info("=== Mode: Rebuild Embeddings ===")

    import subprocess

    script_path = Path(__file__).parent / "src" / "generate_embeddings.py"

    if not script_path.exists():
        logger.error(f"Embeddings script not found: {script_path}")
        sys.exit(1)

    logger.info(f"Running embeddings generation script: {script_path}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True
        )

        # Print output
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)

        logger.info("✓ Sample embeddings generated successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to generate sample embeddings")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        sys.exit(1)


def mode_scan(config: dict, resume: bool = False):
    """
    Scan and classify files.

    Args:
        config: Configuration dictionary
        resume: Resume from checkpoint
    """
    logger.info("=== Mode: Scan ===")

    # Validate config
    if not validate_config(config):
        logger.error("Invalid configuration")
        sys.exit(1)

    # Initialize classifier
    classifier = FaceClassifier(
        source_dir=config['paths']['source_directory'],
        output_dir=config['paths']['output_directory'],
        samples_dir=config['paths']['samples_directory'],
        embeddings_dir=config['paths']['embeddings_directory'],
        database_path=config['paths']['database_path'],
        model_name=config['recognition']['model_name'],
        detector_backend=config['recognition']['detector_backend'],
        enforce_detection=config['recognition']['enforce_detection'],
        confidence_threshold=config['recognition']['confidence_threshold'],
        parallel_workers=config['processing']['parallel_workers'],
        video_sample_fps=config['processing']['video_sample_fps'],
        checkpoint_interval=config['processing']['checkpoint_interval'],
    )

    # Load sample embeddings
    logger.info("Loading sample embeddings...")
    if not classifier.recognizer.load_sample_embeddings():
        logger.error("Failed to load sample embeddings. Run --mode rebuild-embeddings first.")
        sys.exit(1)

    # Process all files
    logger.info("Starting file processing...")
    results = classifier.process_all(resume=resume)

    # Print summary
    logger.info("=== Processing Summary ===")
    logger.info(f"Total files: {results['total']}")
    logger.info(f"Processed: {results['processed']}")
    logger.info(f"Errors: {results['errors']}")
    logger.info(f"No faces: {results['no_faces']}")
    logger.info(f"Classifications: {results['classifications']}")


def mode_review(config: dict):
    """
    Launch GUI reviewer.

    Args:
        config: Configuration dictionary
    """
    logger.info("=== Mode: Review ===")

    launch_reviewer(
        database_path=config['paths']['database_path'],
        output_dir=config['paths']['output_directory'],
        model_name=config['recognition']['model_name'],
        detector_backend=config['recognition']['detector_backend']
    )


def mode_full(config: dict, resume: bool = False):
    """
    Run full pipeline (scan + review).

    Args:
        config: Configuration dictionary
        resume: Resume from checkpoint
    """
    logger.info("=== Mode: Full Pipeline ===")

    # Step 1: Scan
    mode_scan(config, resume=resume)

    # Step 2: Review
    mode_review(config)


def mode_resume(config: dict):
    """
    Resume processing from checkpoint.

    Args:
        config: Configuration dictionary
    """
    logger.info("=== Mode: Resume ===")

    checkpoint_manager = CheckpointManager(str(Path(config['paths']['database_path']).parent))

    if not checkpoint_manager.has_checkpoint():
        logger.error("No checkpoint found. Run --mode scan first.")
        sys.exit(1)

    mode_scan(config, resume=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Face Analysis App - Automatic face classification system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample embeddings
  python main.py --mode rebuild-embeddings

  # Scan and classify all files
  python main.py --mode scan

  # Review low-confidence detections
  python main.py --mode review

  # Run full pipeline
  python main.py --mode full

  # Resume from checkpoint
  python main.py --mode resume

  # Use custom config
  python main.py --mode scan --config my_config.yaml
        """
    )

    parser.add_argument(
        '--mode',
        choices=['rebuild-embeddings', 'scan', 'review', 'full', 'resume'],
        required=True,
        help='Execution mode'
    )

    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='DEBUG',
        help='Logging level (default: DEBUG)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Execute mode
    try:
        if args.mode == 'rebuild-embeddings':
            mode_rebuild_embeddings(config)

        elif args.mode == 'scan':
            mode_scan(config, resume=False)

        elif args.mode == 'review':
            mode_review(config)

        elif args.mode == 'full':
            mode_full(config, resume=False)

        elif args.mode == 'resume':
            mode_resume(config)

        logger.info("=== Execution completed successfully ===")

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
