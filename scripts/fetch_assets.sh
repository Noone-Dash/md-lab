#!/usr/bin/env bash
# Fetch the external assets that are NOT vendored in git (licences/size).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Martini 3 force field (coarse-grained track)"
mkdir -p assets/martini
BASE="https://raw.githubusercontent.com/marrink-lab/martini-forcefields/main/martini_forcefields/regular/v3.0.0/gmx_files"
for f in martini_v3.0.0.itp \
         martini_v3.0.0_phospholipids_v1.itp \
         martini_v3.0.0_solvents_v1.itp \
         martini_v3.0.0_ions_v1.itp; do
  [ -s "assets/martini/$f" ] || curl -fsSL "$BASE/$f" -o "assets/martini/$f"
  printf "    %-42s %s bytes\n" "$f" "$(wc -c < "assets/martini/$f")"
done

echo "==> 3Dmol.js is vendored in viewer/static/ (already in the repo)"
echo "==> done. GROMACS itself is external — see SETUP.md"
