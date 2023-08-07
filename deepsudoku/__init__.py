import os
import pathlib

DIFFICULTIES = ['easy', 'medium', 'hard', 'extreme', 'insane',]

PACKAGE_PATH = pathlib.Path(os.path.abspath(__file__)).parent
REPO_PATH = PACKAGE_PATH.parent


PATH_TO_TDOKU_BIN = REPO_PATH / "tdoku/build/libtdoku_shared.so"

TDOKU_AVAILABLE = os.path.exists(PATH_TO_TDOKU_BIN)



