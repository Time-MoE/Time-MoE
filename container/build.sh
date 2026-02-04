#!/usr/bin/env bash
set -euo pipefail

DEF="$(dirname "$0")/warp_moe.def"
OUT="$(dirname "$0")/warp_moe.sif"

echo "[INFO] Building: ${OUT} from ${DEF}"
echo "[INFO] You may need sudo or fakeroot depending on cluster policy."

apptainer build "${OUT}" "${DEF}"
echo "[OK] Built: ${OUT}"
