"""FastSegmentator command-line dispatcher.

Exposes a single `FastSegmentator` command that delegates to one of the two
inference entry points, each keeping its own flags:

    FastSegmentator totalseg -i <in> -o <out> --task total
    FastSegmentator nnunet   -i <in> -o <out> --model_path <model>

Run `FastSegmentator <command> --help` for command-specific options.
"""

import sys

COMMANDS = ("totalseg", "nnunet")

USAGE = """FastSegmentator <command> [options]

Commands:
  totalseg   Config-driven TotalSegmentator inference (CT + MR, select via --task)
  nnunet     Generic nnU-Net model-folder inference (select weights via --model_path)

Run 'FastSegmentator <command> --help' for command-specific options.
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    if command not in COMMANDS:
        sys.exit(
            f"ERROR: unknown command '{command}'. "
            f"Expected one of {', '.join(COMMANDS)}. Run 'FastSegmentator --help'."
        )

    # Drop the sub-command token so the delegated argparse sees a clean argv.
    sys.argv = [f"FastSegmentator {command}"] + sys.argv[2:]

    if command == "totalseg":
        from totalseg_infer import main as run
    else:  # "nnunet"
        from nnunet_infer_nii import main as run
    run()


if __name__ == "__main__":
    main()
