#!/usr/bin/env bash
#SBATCH -A CHANGE_ME
#SBATCH -J rt10_strict100
#SBATCH -o robotwin-strict100-%j.out
#SBATCH -e robotwin-strict100-%j.err
#SBATCH -p CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH -t 0-10:00:00

set -euo pipefail
BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
exec "${BUNDLE_ROOT}/scripts/run_policy_strict100.sh"
