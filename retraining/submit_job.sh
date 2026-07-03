#!/usr/bin/env bash
#
# Submit a shell script as a non-interactive PBS job on ASPIRE2A.
#
# Usage:
#   bash retraining/submit_job.sh JOB_NAME SCRIPT
#
# Example:
#   bash retraining/submit_job.sh prepare-regression \
#     retraining/prepare_regression_data.sh
#
# Optional resource overrides:
#   WALLTIME=12:00:00 MEM=96G \
#     bash retraining/submit_job.sh prepare-regression \
#     retraining/prepare_regression_data.sh
#

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 JOB_NAME SCRIPT" >&2
    exit 2
fi

JOB_NAME="$1"
TARGET_SCRIPT="$(realpath "$2")"

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "Script does not exist: $TARGET_SCRIPT" >&2
    exit 1
fi

SUBMIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SUBMIT_SCRIPT_DIR/.." && pwd)"

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

echo "Job ID:  \$PBS_JOBID"
echo "Host:    \$(hostname)"
echo "Started: \$(date)"
echo "Script:  $TARGET_SCRIPT"
echo

bash "$TARGET_SCRIPT"

echo
echo "Finished: \$(date)"
EOF
)"

echo "Submitted job: $JOB_ID"
echo "Log file:"
echo "  $LOG_FILE"