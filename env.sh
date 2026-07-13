# Source me:  source env.sh
# Sets up GROMACS + the labkit Python venv for an interactive session.
export GMX_ROOT="${GMX_ROOT:-/home/v_u/Documents/tools/opt/gromacs-2026.2}"
source "$GMX_ROOT/bin/GMXRC" >/dev/null 2>&1

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$LAB_DIR/.venv" ]; then
  source "$LAB_DIR/.venv/bin/activate"
fi

echo "GROMACS lab ready:"
echo "  gmx           -> $(command -v gmx)"
echo "  python        -> $(command -v python)"
echo "  ./labctl.py list        # see simulations"
echo "  python viewer/app.py    # launch the web UI (http://127.0.0.1:5000)"
