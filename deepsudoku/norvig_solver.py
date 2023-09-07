"""Code from https://github.com/brotskydotcom/sudoku-generator/tree/develop"""
# MIT License
#
# Copyright (c) 2019 Daniel Brotsky
#
# Portions copyright (c) 2017 by Peter Norvig and Naoki Shibuya
# (See https://towardsdatascience.com/peter-norvigs-sudoku-solver-25779bb349ce)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Dict

from deepsudoku.utils import Board


class NorvigSolver:
    """This solver was originally published by Peter Norvig.
    It was later updated to py3 by Naoki Shibuya.
    See https://towardsdatascience.com/peter-norvigs-sudoku-solver-25779bb349ce
    for the code and an explanation of how it works. Small adaptations have been made
    to adapt for multiple sizes of puzzle and to invoke the solver from the generator.
    """
    # digits, row labels and col labels up to 16 (max puzzle size)
    all_digits = '123456789ABCDEFG'
    all_rows = 'ABCDEFGHIJKLMNOP'
    all_cols = 'abcdefghijklmnop'

    def __init__(self, board: Board):
        """Captures the board so you can solve it non-destructively."""
        self.digits = self.all_digits[:board.side_length]
        self.rows = self.all_rows[:board.side_length]
        self.cols = self.all_cols[:board.side_length]
        self.squares = self.cross(self.rows, self.cols)
        self.unit_list = ([self.cross(self.rows, c) for c in self.cols] +
                          [self.cross(r, self.cols) for r in self.rows] +
                          [self.cross(rs, cs)
                           for rs in self.rank_groups(self.rows, board.rank, board.side_length)
                           for cs in self.rank_groups(self.cols, board.rank, board.side_length)])
        self.units = dict((s, [u for u in self.unit_list if s in u]) for s in self.squares)
        self.peers = dict((s, set(sum(self.units[s], [])) - {s}) for s in self.squares)
        self.values = self.board2values(board)
        self.solution = {}

    def board2values(self, board):
        return self.parse_grid(''.join([str(c) for c in board.cells]))

    def values2board(self, values):
        try:
            return Board([self.all_digits.index(values[s][0])+1 for s in self.squares])
        except ValueError:
            return {}

    def is_solution(self):
        return all(len(self.values[s]) == 1 for s in self.squares)

    def cross(self, aa: [str], bb: [str]):
        """Cross product of elements in aa and elements in bb."""
        return [a + b for a in aa for b in bb]

    def rank_groups(self, indices: str, step: int, end: int) -> [str]:
        return [indices[start:start + step] for start in range(0, end, step)]

    def parse_grid(self, grid: str) -> Dict[str, str]:
        """Convert grid to a dict of possible values, {square: digits}, or
        return {} if a contradiction is detected."""
        # To start, every square can be any digit; then assign values from the grid.
        values = dict((s, self.digits) for s in self.squares)
        for s, d in self.grid_values(grid).items():
            if d in self.digits and not self.assign(values, s, d):
                return {}  # (Fail if we can't assign d to square s.)
        return values

    def grid_values(self, grid: str) -> Dict[str, str]:
        """Convert grid into a dict of {square: char} with '0' or '.' for empties."""
        chars = [c for c in grid if c in self.digits or c in '0.-']
        if len(chars) != len(self.squares):
            raise ValueError("Grid contains illegal characters")
        return dict(zip(self.squares, chars))

    def assign(self, values: Dict[str, str], s: str, d: str):
        """Eliminate all the other values (except d) from values[s] and propagate.
        Return values, except return {} if a contradiction is detected."""
        other_values = values[s].replace(d, '')
        if all(self.eliminate(values, s, d2) for d2 in other_values):
            return values
        else:
            return {}

    def eliminate(self, values: Dict[str, str], s: str, d: str):
        """Eliminate d from values[s]; propagate when values or places <= 2.
        Return values, except return {} if a contradiction is detected."""
        if d not in values[s]:
            return values  # Already eliminated
        values[s] = values[s].replace(d, '')
        # (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
        if len(values[s]) == 0:
            return {}  # Contradiction: removed last value
        elif len(values[s]) == 1:
            d2 = values[s]
            if not all(self.eliminate(values, s2, d2) for s2 in self.peers[s]):
                return {}
        # (2) If a unit u is reduced to only one place for a value d, then put it there.
        for u in self.units[s]:
            places = [s for s in u if d in values[s]]
            if len(places) == 0:
                return {}  # Contradiction: no place for this value
            elif len(places) == 1:
                # d can only be in one place in unit; assign it there
                if not self.assign(values, places[0], d):
                    return {}
        return values

    def can_solve(self):
        values = self.search(self.values)
        if values:
            self.solution = self.values2board(values)
            return True
        return False

    def search(self, values: Dict[str, str]) -> Dict[str, str]:
        """Using depth-first search and propagation, try all possible values."""
        if not values:
            return {}  # Failed earlier
        if all(len(values[s]) == 1 for s in self.squares):
            return values  # Solved!
        # Chose the unfilled square s with the fewest possibilities
        n, s = min((len(values[s]), s) for s in self.squares if len(values[s]) > 1)
        return next(filter(len, (self.search(self.assign(values.copy(), s, d)) for d in values[s])), {})
