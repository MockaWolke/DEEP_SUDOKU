import os
import pathlib

DIFFICULTIES = ['easy', 'medium', 'hard', 'extreme', 'insane',]

PACKAGE_PATH = pathlib.Path(os.path.abspath(__file__)).parent
REPO_PATH = PACKAGE_PATH.parent
