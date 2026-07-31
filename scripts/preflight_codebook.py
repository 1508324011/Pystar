import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pystar._codebook_contracts import compile_codebook_contract
from pystar.infrastructure import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="PyStar shared codebook preflight")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    compiled = compile_codebook_contract(config.codebook)
    debug_path = compiled.write_debug_csv(Path(config.pipeline.output.directory))
    print(f"Preflight complete: {debug_path}")


if __name__ == "__main__":
    main()
