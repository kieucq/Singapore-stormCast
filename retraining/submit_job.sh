#!/usr/bin/env bash
#
# Submit a shell script as a non-interactive PBS batch job on ASPIRE2A.
#
# Basic usage:
#
#   bash submit_job.sh JOB_NAME TARGET_SCRIPT [TARGET_ARGUMENTS...]
#
# Example:
#
#   bash retraining/submit_job.sh \
#     prepare-data \
#     retraining/prepare_data.sh \
#     1995-01-01 1995-01-31 \
#     1995-02-01 1995-02-07
#
# Resource defaults:
#
#   Project:   17001770
#   Queue:     normal
#   CPUs:      4
#   Memory:    64G
#   GPUs:      0
#   Walltime:  08:00:00
#
# Override resources by setting environment variables:
#
#   WALLTIME=12:00:00 NCPUS=8 MEM=96G \
#   bash retraining/submit_job.sh \
#     prepare-regression \
#     retraining/prepare_regression_data.sh
#
# Other examples:
#
#   PROJECT=17001770 QUEUE=normal \
#   bash retraining/submit_job.sh my-job path/to/script.sh
#
# Useful PBS commands:
#
#   qstat -u "$USER"
#       List your queued and running jobs.
#
#   qstat -f JOB_ID
#       Show detailed information for one job.
#
#   tail -F LOG_FILE
#       Follow the job log. Unlike `tail -f`, `tail -F` also waits for the
#       file to be created and survives file replacement.
#
#   qdel JOB_ID
#       Cancel a queued or running job.
#
#   qselect -u "$USER"
#       Print only your job IDs.
#
# Help:
#
#   bash retraining/submit_job.sh --help
#

set -euo pipefail


usage() {
    cat <<'EOF'
Submit a shell script as a non-interactive PBS job on ASPIRE2A.

Usage:
  submit_job.sh JOB_NAME TARGET_SCRIPT [TARGET_ARGUMENTS...]
  submit_job.sh --help

Example:
  bash retraining/submit_job.sh \
    prepare-data \
    retraining/prepare_data.sh \
    1995-01-01 1995-01-31 \
    1995-02-01 1995-02-07

Default resources:
  PROJECT=17001770
  QUEUE=normal
  NCPUS=4
  MEM=64G
  NGPUS=0
  WALLTIME=08:00:00

Override resources:
  WALLTIME=12:00:00 NCPUS=8 MEM=96G \
  bash retraining/submit_job.sh \
    prepare-data \
    retraining/prepare_data.sh \
    1995-01-01 1995-01-31 \
    1995-02-01 1995-02-07

Useful commands after submission:
  qstat -u "$USER"       List your jobs
  qstat -f JOB_ID        Inspect one job
  tail -F LOG_FILE       Follow the job output
  qdel JOB_ID            Cancel the job

Logs are written to:
  ~/scratch/retraining/logs/
EOF
}


if [[ $# -lt 2 ]]; then
    echo "Error: expected a job name and target script." >&2
    echo >&2
    usage >&2
    exit 2
fi


JOB_NAME="$1"
TARGET_SCRIPT="$(realpath "$2")"

shift 2
TARGET_ARGS=("$@")

# Store the complete command as a Bash array so arguments containing spaces
# or special characters remain correctly quoted in the submitted job.
TARGET_COMMAND=(
    bash
    "$TARGET_SCRIPT"
    "${TARGET_ARGS[@]}"
)

TARGET_COMMAND_DECL="$(declare -p TARGET_COMMAND)"

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "Error: target script does not exist:" >&2
    echo "  $TARGET_SCRIPT" >&2
    exit 1
fi


SUBMIT_SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_ROOT="$(
    cd "$SUBMIT_SCRIPT_DIR/.."
    pwd
)"


# PBS resources. Set environment variables before invoking this script to
# override any of these defaults.
PROJECT="${PROJECT:-17001770}"
QUEUE="${QUEUE:-normal}"
NCPUS="${NCPUS:-4}"
MEM="${MEM:-64G}"
NGPUS="${NGPUS:-0}"
WALLTIME="${WALLTIME:-08:00:00}"


LOG_DIR="$HOME/scratch/retraining/logs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/${JOB_NAME}_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"


echo "Submitting PBS job"
echo "------------------"
echo "Job name:      $JOB_NAME"
echo "Target script: $TARGET_SCRIPT"
echo "Arguments:     ${TARGET_ARGS[*]:-(none)}"
echo "Project root:  $PROJECT_ROOT"
echo "Project:       $PROJECT"
echo "Queue:         $QUEUE"
echo "CPUs:          $NCPUS"
echo "Memory:        $MEM"
echo "GPUs:          $NGPUS"
echo "Walltime:      $WALLTIME"
echo "Log file:      $LOG_FILE"
echo


JOB_ID="$(
    qsub <<EOF
#!/usr/bin/env bash
#PBS -N $JOB_NAME
#PBS -P $PROJECT
#PBS -q $QUEUE
#PBS -l select=1:ncpus=$NCPUS:mem=$MEM:ngpus=$NGPUS
#PBS -l walltime=$WALLTIME
#PBS -j oe
#PBS -o $LOG_FILE

set -euo pipefail

module load miniforge3/25.3.1
eval "\$(conda shell.bash hook)"
conda activate stormcast

cd "$PROJECT_ROOT"

echo "PBS JOB"
echo "======="
echo "Job ID:     \$PBS_JOBID"
echo "Host:       \$(hostname)"
echo "Started:    \$(date)"
echo "Working dir: \$(pwd)"
echo "Script:      $TARGET_SCRIPT"
echo "Arguments:   ${TARGET_ARGS[*]:-(none)}"
echo

$TARGET_COMMAND_DECL

echo "Command:"
printf '  %q' "\${TARGET_COMMAND[@]}"
printf '\n\n'

"\${TARGET_COMMAND[@]}"

echo
echo "Job completed successfully."
echo "Finished: \$(date)"
EOF
)"


echo "Job submitted successfully."
echo
echo "Job ID:"
echo "  $JOB_ID"
echo
echo "Useful commands:"
echo
echo "  # List all your jobs"
echo "  qstat -u \"$USER\""
echo
echo "  # Inspect this job"
echo "  qstat -f \"$JOB_ID\""
echo
echo "  # Follow this job's log"
echo "  tail -F \"$LOG_FILE\""
echo
echo "  # Cancel this job"
echo "  qdel \"$JOB_ID\""
echo
echo "Log file:"
echo "  $LOG_FILE"