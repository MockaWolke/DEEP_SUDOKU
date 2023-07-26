import json
import sys
import time
import numpy as np
from deepsudoku import DIFFICULTIES
from hashlib import sha256

from Sudoku.board import Board
from Sudoku.generator import Generator


def generate(verbose="True", sidelen='9', difficulty="easy", count=1):
    """Generate one or more Sudoku puzzles.
    You can specify the size and difficulty of the puzzles."""
    
    assert difficulty in DIFFICULTIES, f"difficulty must be in {DIFFICULTIES}"

    outputs = []
    gen = Generator(sidelen, difficulty)
    start = time.time()
    for iteration in range(1, count + 1):
        result = gen.generate_one()
        outputs.append(result)
    end = time.time()

    if verbose:
        print("Summary statistics:", file=sys.stderr)
        puzzle_str = "puzzles that are" if count > 1 else "puzzle that is"
        print("Generated {3} '{0}' {2} {1}x{1}."
              .format(difficulty, sidelen, puzzle_str, count),
              file=sys.stderr)
        for index, result in enumerate(outputs, 1):
            board = result['puzzle']
            empty = len(board.get_empty_cells())
            filled = len(board.get_filled_cells())
            total = board.size
            print("Puzzle {}: Empty={} ({:.0%}), Filled={} ({:.0%})."
                  .format(index,
                          empty, empty / total,
                          filled, filled / total),
                  file=sys.stderr)
        print("Generation time: {:.1f} seconds total ({:.1f} secs/puzzle)."
              .format(end - start, (end - start) / count),
              file=sys.stderr)

    return_values = []

    for result in outputs:

        puzzle_array = np.array(
            result['puzzle'].values(), np.uint8).reshape(int(sidelen), -1)
        result_array = np.array(
            result['solution'].values(), np.uint8).reshape(int(sidelen), -1)

        return_values.append((puzzle_array, result_array))

    return return_values
