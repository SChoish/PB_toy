"""Run both PathBridger concept figure variants."""

from .codes_numerical import main as run_numerical
from .codes_nn import main as run_nn


def main() -> None:
    """Generate both concept-figure variants."""
    print("----- Numerical -----")
    run_numerical()
    print("\n----- Tiny NN -----")
    run_nn()


if __name__ == "__main__":
    main()
