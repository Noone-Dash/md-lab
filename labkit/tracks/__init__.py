"""Non-GROMACS simulation tracks (each emits the standard run.json manifest)."""

from .cell import CellRD
from .openmm_md import OpenMMImplicit
from .qmmm import QMMM
from .reaction import ReactionScan

TRACKS = {t.key: t for t in [OpenMMImplicit(), QMMM(), ReactionScan(), CellRD()]}


def list_tracks():
    return [t.meta() for t in TRACKS.values()]


def get_track(key):
    return TRACKS[key]
