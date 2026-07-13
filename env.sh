# source me:  source env.sh
# Resolves the environment WITHOUT assuming anyone's directory layout.

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# --- GROMACS: only set GMX_ROOT if you have a prefix install.
# If `gmx` is already on PATH (e.g. after `module load gromacs`), nothing to do.
if ! command -v gmx >/dev/null 2>&1 && [ -z "${GMX_ROOT:-}" ]; then
  for _p in /usr/local/gromacs /opt/gromacs "$HOME/gromacs" "$HOME/opt/gromacs"; do
    for _c in "$_p" "$_p"-*; do
      if [ -x "$_c/bin/gmx" ]; then export GMX_ROOT="$_c"; break 2; fi
    done
  done
fi
[ -n "${GMX_ROOT:-}" ] && [ -f "$GMX_ROOT/bin/GMXRC" ] && . "$GMX_ROOT/bin/GMXRC"

# --- python venv, if one exists next to the repo
[ -d "$_here/.venv" ] && . "$_here/.venv/bin/activate"

echo "Run  python -m labkit.doctor  to check the environment."
