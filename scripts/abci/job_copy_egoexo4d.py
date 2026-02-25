"""ABCI job submission script for copying EgoExo4D takes.

Generates a PBS script that runs ``copy_egoexo4d_takes.sh`` on an ABCI
compute node and submits it via ``qsub``.

Usage::

    # Preview the generated PBS script
    python scripts/abci/job_copy_egoexo4d.py --dry-run

    # Submit the job
    python scripts/abci/job_copy_egoexo4d.py

    # Custom walltime and job name
    python scripts/abci/job_copy_egoexo4d.py --walltime 3:00:00 --job-name copy-ee4d-v1
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import pathlib
import subprocess
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CopyEgoExo4DConfig:
    """ABCI PBS job configuration for copying EgoExo4D takes."""

    # PBS queue name.
    queue: str = "rt_HC"
    # ABCI project group.
    project: str = "gch51606"
    # Number of compute nodes (-l select=N).
    num_nodes: int = 1
    # Wall-clock time limit (HH:MM:SS).
    walltime: str = "6:00:00"
    # Job name shown in qstat. Auto-generated if not set.
    job_name: str | None = None
    # Resource type for reserved queues (-v RTYPE=). Only set when using a reservation.
    rtype: str | None = None
    # Directory for generated PBS scripts and job logs.
    job_dir: pathlib.Path = pathlib.Path("jobs")
    # Print the generated script and exit without submitting.
    dry_run: bool = False

    # ---- Data paths ----
    # DATA_ROOT for dataset paths.
    data_root: str = "/groups/gch51606/takehiko.ohkawa/tmp_data"
    # Source takes directory in ABCI shared storage.
    src_takes_dir: str = "/groups/gch51606/dataset/egoexo4d/raw/takes"

    # ---- Environment ----
    # Conda environment prefix. Set to empty string to skip conda activation.
    conda_prefix: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_pbs_script(config: CopyEgoExo4DConfig, job_name: str) -> str:
    """Build the full PBS shell script as a string."""
    log_path = config.job_dir / f"{job_name}.log"

    # --- PBS header ---
    lines = [
        "#!/bin/sh",
        f"#PBS -q {config.queue}",
        f"#PBS -P {config.project}",
        f"#PBS -l select={config.num_nodes}",
        f"#PBS -l walltime={config.walltime}",
        f"#PBS -N {job_name}",
        "#PBS -j oe",
        f"#PBS -o {log_path}",
    ]
    if config.rtype is not None:
        lines.append(f"#PBS -v RTYPE={config.rtype}")
    lines.append("")

    # --- Conda activation block ---
    if config.conda_prefix:
        conda_block = textwrap.dedent(f"""\
            # ---- Conda activate ----
            if [ -f "${{HOME}}/miniconda3/etc/profile.d/conda.sh" ]; then
                . "${{HOME}}/miniconda3/etc/profile.d/conda.sh"
            elif [ -f "${{HOME}}/anaconda3/etc/profile.d/conda.sh" ]; then
                . "${{HOME}}/anaconda3/etc/profile.d/conda.sh"
            fi
            eval "$(conda shell.bash hook)" 2>/dev/null || true
            conda activate "{config.conda_prefix}"
            echo "[INFO]: Python: $(python -V), $(which python)"
        """)
    else:
        conda_block = textwrap.dedent("""\
            # ---- No conda prefix specified; using system Python ----
            echo "[INFO]: Python: $(python -V), $(which python)"
        """)

    # --- Body ---
    body = textwrap.dedent(f"""\
        set -eu

        echo "=== Job start: $(date) ==="
        echo "=== Host: $(hostname) ==="

        {conda_block}
        # ---- Module setup ----
        if command -v module >/dev/null 2>&1; then
            module load gcc-toolset/10 2>/dev/null || true
            echo "[INFO]: Modules loaded"
        fi

        # ---- Working directory ----
        if [ -n "${{PBS_O_WORKDIR:-}}" ]; then
            cd "$PBS_O_WORKDIR"
        else
            cd "$(dirname "$0")/.."
        fi
        echo "[INFO]: Working directory: $(pwd)"

        # ---- Environment variables ----
        export DATA_ROOT="{config.data_root}"
        export SRC_TAKES_DIR="{config.src_takes_dir}"
        echo "[INFO]: DATA_ROOT=${{DATA_ROOT}}"
        echo "[INFO]: SRC_TAKES_DIR=${{SRC_TAKES_DIR}}"

        # ---- Run copy script ----
        bash scripts/abci/copy_egoexo4d_takes.sh

        echo "=== Job end: $(date) ==="
    """)

    return "\n".join(lines) + body


def submit_job(script_path: pathlib.Path) -> str:
    """Submit a PBS script via qsub. Returns the job ID."""
    try:
        result = subprocess.run(
            ["qsub", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.error("qsub not found. Are you running on ABCI? Use --dry-run to preview locally.")
        raise SystemExit(1)

    if result.returncode != 0:
        logger.error(f"qsub failed (exit {result.returncode}):\n{result.stderr.strip()}")
        raise SystemExit(result.returncode)

    job_id = result.stdout.strip()
    return job_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        import tyro
    except ImportError:
        logger.error("tyro is required: pip install tyro")
        raise SystemExit(1)

    config = tyro.cli(CopyEgoExo4DConfig)

    # Generate job name
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = config.job_name or f"copy-ee4d-{now}"

    # Ensure job directory exists
    config.job_dir.mkdir(parents=True, exist_ok=True)

    # Render and write PBS script
    script_content = render_pbs_script(config, job_name)
    script_path = config.job_dir / f"{job_name}.sh"
    script_path.write_text(script_content)
    logger.info(f"PBS script written to {script_path}")

    if config.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — generated script: {script_path}")
        print(f"{'='*60}\n")
        print(script_content)
        return

    # Submit
    job_id = submit_job(script_path)
    logger.info(f"Job submitted: {job_id}")
    logger.info(f"Log file: {config.job_dir / f'{job_name}.log'}")


if __name__ == "__main__":
    main()
