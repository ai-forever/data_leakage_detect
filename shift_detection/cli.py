#!/usr/bin/env python
"""
Shift Detection Command Line Interface

CLI for running baseline membership inference attacks.
"""

from shift_detection.run_attack import main as run_attack_main


def main():
    """Entry point for shift-attack CLI command."""
    # The run_attack.py main() function already handles all the argument parsing
    # We just need to call it directly
    run_attack_main()


if __name__ == "__main__":
    main()
