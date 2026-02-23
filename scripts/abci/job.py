"""ABCI job submission script for InternVideo2 multi-modality training.

Generates a PBS script from CLI parameters and submits it via ``qsub``.

Usage::

    python scripts/abci_job.py [OPTIONS]

Examples::

    # Default: 1-node Stage2 motion training
    python scripts/abci_job.py --queue rt_HG --job-name motion-train

    # Multi-node with custom config
    python scripts/abci_job.py --queue rt_HG --num-nodes 2 \
        --config scripts/pretraining/stage2/1B/config.py

    # Dry run (print script without submitting)
    python scripts/abci_job.py --dry-run

    # Extra env vars
    python scripts/abci_job.py --env WANDB_MODE=offline CUDA_LAUNCH_BLOCKING=1
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import pathlib
import subprocess
import sys
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
class ABCIJobConfig:
    """ABCI PBS job configuration for InternVideo2 training."""

    # PBS queue name (e.g. rt_HG, rt_AF, rt_AG) or reservation ID.
    queue: str = "rt_HG"
    # ABCI project group.
    project: str = "INSERT_GROUP_HERE"
    # Number of compute nodes (-l select=N).
    num_nodes: int = 1
    # Wall-clock time limit (HH:MM:SS).
    walltime: str = "72:00:00"
    # Job name shown in qstat. Auto-generated if not set.
    job_name: str | None = None
    # Resource type for reserved queues (-v RTYPE=). Only set when using a reservation.
    rtype: str | None = None
    # Directory for generated PBS scripts and job logs.
    job_dir: pathlib.Path = pathlib.Path("jobs")
    # Print the generated script and exit without submitting.
    dry_run: bool = False
    # Extra environment variables injected into the PBS script (KEY=VALUE).
    env: tuple[str, ...] = ()

    # ---- Training options ----
    # Path to training config (relative to project root).
    config: str = "scripts/pretraining/stage2/1B_motion/config.py"
    # Override output directory. Auto-generated if not set.
    output_dir: str | None = None
    # Number of dataloader workers per GPU.
    num_workers: int = 0
    # Extra arguments passed to tasks/pretrain.py.
    extra_args: tuple[str, ...] = ()

    # ---- Environment paths ----
    # Conda environment prefix. Set to empty string to skip conda activation.
    conda_prefix: str = ""
    # CUDA module to load (e.g. cuda/12.2, cuda/12.1).
    cuda_module: str = "cuda/12.2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_env_entries(entries: tuple[str, ...]) -> None:
    """Ensure every --env entry has a ``=`` sign."""
    for entry in entries:
        if "=" not in entry:
            logger.error(f"Invalid --env value '{entry}': expected KEY=VALUE format.")
            raise SystemExit(1)


def render_pbs_script(config: ABCIJobConfig, job_name: str) -> str:
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

    # --- Output directory ---
    if config.output_dir:
        output_dir = config.output_dir
    else:
        config_parent = str(pathlib.Path(config.config).parent)
        output_dir = f"{config_parent}/outputs/{job_name}"

    # --- Extra args ---
    extra = ""
    for arg in config.extra_args:
        extra += f" {arg}"

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

    # --- Env var exports ---
    env_block = ""
    for entry in config.env:
        key, _, value = entry.partition("=")
        env_block += f'export {key}="{value}"\n'
    if env_block:
        env_block = f"# ---- Extra environment variables ----\n{env_block}\n"

    # --- Body ---
    body = textwrap.dedent(f"""\
        set -eu

        echo "=== Job start: $(date) ==="
        echo "=== Host: $(hostname) ==="

        {conda_block}
        # ---- Module setup ----
        if command -v module >/dev/null 2>&1; then
            module load gcc-toolset/10 2>/dev/null || module load gcc/12.2.0 2>/dev/null || true
            module load {config.cuda_module}
            echo "[INFO]: Modules loaded"
        fi

        NVCC_PATH="$(command -v nvcc || true)"
        if [ -n "${{NVCC_PATH}}" ]; then
            CUDA_HOME="$(dirname "$(dirname "${{NVCC_PATH}}")")"
            export CUDA_HOME
            echo "[INFO]: CUDA_HOME=${{CUDA_HOME}}"
        fi

        # ---- Working directory ----
        if [ -n "${{PBS_O_WORKDIR:-}}" ]; then
            cd "$PBS_O_WORKDIR"
        else
            cd "$(dirname "$0")/.."
        fi
        echo "[INFO]: Working directory: $(pwd)"

        # ---- GPU check ----
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi || true
        else
            echo "[WARN]: nvidia-smi not found"
        fi

        # ---- Distributed training setup ----
        NNODES="${{PBS_NUM_NODES:-{config.num_nodes}}}"
        NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
        MASTER_PORT=$((12000 + RANDOM % 20000))

        export MASTER_PORT
        export OMP_NUM_THREADS=1
        export PYTHONPATH="${{PYTHONPATH:+${{PYTHONPATH}}:}}${{PWD}}"
        export WANDB_MODE="${{WANDB_MODE:-online}}"

        # Suppress noisy logs
        export PYTHONWARNINGS="ignore"
        export TRANSFORMERS_VERBOSITY="error"
        export DATASETS_VERBOSITY="error"
        export NCCL_DEBUG="WARN"
        export DS_LOG_LEVEL="WARNING"
        export TORCH_DISTRIBUTED_DEBUG="OFF"

        {env_block}echo "[INFO]: Nodes=${{NNODES}}, GPUs/node=${{NUM_GPUS}}, MASTER_PORT=${{MASTER_PORT}}"
        echo "[INFO]: Config: {config.config}"
        echo "[INFO]: Output: {output_dir}"

        # ---- Run training ----
        if [ "${{NNODES}}" -gt 1 ]; then
            # Multi-node: use mpirun to distribute torchrun across nodes
            MASTER_ADDR=$(head -n 1 "${{PBS_NODEFILE}}")
            export MASTER_ADDR
            echo "[INFO]: Multi-node training: MASTER_ADDR=${{MASTER_ADDR}}"

            mpirun -np "${{NNODES}}" --npernode 1 --bind-to none \\
                torchrun \\
                    --nnodes="${{NNODES}}" \\
                    --nproc_per_node="${{NUM_GPUS}}" \\
                    --rdzv_backend=c10d \\
                    --rdzv_endpoint="${{MASTER_ADDR}}:${{MASTER_PORT}}" \\
                    tasks/pretrain.py \\
                    {config.config} \\
                    output_dir "{output_dir}" \\
                    num_workers {config.num_workers}{extra}
        else
            # Single-node
            torchrun \\
                --nproc_per_node="${{NUM_GPUS}}" \\
                --master_port="${{MASTER_PORT}}" \\
                tasks/pretrain.py \\
                {config.config} \\
                output_dir "{output_dir}" \\
                num_workers {config.num_workers}{extra}
        fi

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

    config = tyro.cli(ABCIJobConfig)

    validate_env_entries(config.env)

    # Generate job name
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = config.job_name or f"iv2-{now}"

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
