from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from typing import List, IO


def string_to_array(string: str):

    string = string.replace(".", "0")

    return np.array([int(c) for c in string], np.uint8).reshape(
        int(np.sqrt(len(string))), -1
    )


def visualize_sudoku(sudoko, return_fig=False, figsize=(6, 6), dpi=100):

    if isinstance(sudoko, str):

        field = string_to_array(sudoko)

    elif isinstance(sudoko, np.ndarray):

        if sudoko.shape != (9, 9):
            raise ValueError("False Shape")

        field = sudoko

    white = np.ones_like(field).astype(np.float32)

    cmap = colors.LinearSegmentedColormap.from_list("", ["green", "white"])

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(white, cmap=cmap, vmin=0, vmax=1)
    plt.axis("off")
    # Draw grid lines
    for i in range(10):
        if i % 3 == 0:
            plt.vlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=2)
            plt.hlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=2)
        else:
            plt.vlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=1)
            plt.hlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=1)

    # now plot the numbers of field
    for i in range(9):
        for j in range(9):
            if field[i, j] != 0:
                plt.text(
                    j,
                    i,
                    str(field[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                )

    if return_fig:
        return fig

    plt.show()


class Cell:
    """Each cell in a Sudoku board has a row, a column, a tile, and a value.
    For convenience, a cell can be annotated with its _context_, which is
    all of the other cells in its row, column, and tile (so called because
    those are the cells that constrain its value)."""

    def __init__(self, row: int, col: int, tile: int, value: int = 0):
        """Each cell is initialized with its row, column, and tile.
        If a value is specified, that's remembered, otherwise the cell
        is initialized with a 0 value (meaning blank).  The context
        of each cell is initialized to empty; it is expected to be
        used as a cache by the board containing the cell.
        No error checking is done of any values.
        """
        self.row = row
        self.col = col
        self.tile = tile
        self.value = value
        self.context = []

    def __repr__(self):
        """The (almost) eval-able form of a cell is its constructor (with its id for specificity)."""
        return "<Cell(row={},col={},tile={},value={}) at {}>".format(
            self.row, self.col, self.tile, self.value, id(self)
        )

    def __str__(self):
        """In traditional Sudoku, each cell is represented by its value as a single character.
        This implementation assumes a puzzle side_length no bigger than 5 (so 25 as max cell value).
        """
        if self.value == 0:
            return "-"
        elif self.value < 10:
            return str(self.value)
        else:
            return chr(ord("A") + self.value - 10)


class Board:
    """A Board is a Sudoku puzzle.  It contains cells organized into rows, columns, and
    _tiles_ (the sub-squares which also have to contain each possible value).  Cells
    are either _filled_ (with a value) or _empty_ (not yet filled); in this implementation
    we use a 0 as the value in an empty cell.  This implementation organizes puzzle cells
    left-to-right along each row, and the rows are read top-to-bottom.  Tiles are also
    organized in the same order.  The _rank_ of the puzzle is the number of squares
    along the edge of each tile, hence a 9x9 puzzle has a side_length of 3.  Each horizontal
    collection of aligned tiles is called a _band_, and each vertical collection of
    aligned tiles is called a _stack_.
    """

    def __init__(self, values: [int]):
        """Create a board with the given initial cell values.
        Cell values are filled left-to-right, top-to-bottom from the given array.
        Basic range checking is done on the values to make sure that the puzzle
        is well-formed, but it's not checked to see if it has a solution.
        """
        # validate arguments
        self.size = len(values)
        if self.size == 16:
            self.rank = 2
            self.side_length = 4
        elif self.size == 81:
            self.rank = 3
            self.side_length = 9
        elif self.size == 256:
            self.rank = 4
            self.side_length = 16
        else:
            raise ValueError(
                "Side length must be 4, 9, or 16 (containing 16, 81, or 256 values)"
            )
        if max(values) > self.side_length or min(values) < 0:
            raise ValueError(
                "Each cell value must be in the range 0 to {}".format(self.side_length)
            )
        # create the indices
        self.rows: List[List[Cell]] = [[] for _ in range(self.side_length)]
        self.columns: List[List[Cell]] = [[] for _ in range(self.side_length)]
        self.tiles: List[List[Cell]] = [[] for _ in range(self.side_length)]
        self.cells: List[Cell] = []
        # create initialized cells and put them in their row, column and tile
        value_index = 0
        for row in range(0, self.side_length):
            for col in range(0, self.side_length):
                tile = self.rank * (row // self.rank) + (col // self.rank)
                cell = Cell(row, col, tile, values[value_index])
                self.rows[row].append(cell)
                self.columns[col].append(cell)
                self.tiles[tile].append(cell)
                self.cells.append(cell)
                value_index += 1
        # for each cell, pre-compute its context (the cells in its row, column, or tile)
        for cell in self.cells:
            cell.context = self.get_context(cell)

    @classmethod
    def from_file(cls, file: IO):
        """Create a new board from an open file stream.
        The named file should contain a puzzle's worth of cell values separated
        by whitespace, where each value is a single digit from 0 (for empty) to
        the side length of the puzzle, using the values from 1-9 and A-G.
        The values are inserted into the puzzle rows from left to right
        and top to bottom, and newlines are treated the same as any other
        whitespace, so it's easy to format the file using one line per row.  The
        puzzle in the file is checked to make sure it's a valid puzzle, so
        if you have any typos the construction will throw an error.
        """
        return Board([int(x, base=17) for x in file.read().split()])

    def get_filled_cells(self) -> [Cell]:
        """Return all the cells with values."""
        return [c for c in self.cells if c.value != 0]

    def get_empty_cells(self) -> [Cell]:
        """Return all the cells with no value."""
        return [c for c in self.cells if c.value == 0]

    def get_context(self, cell: Cell) -> {Cell}:
        """Return the set of cells that share a row, column or tile with this cell."""
        return {
            c
            for c in self.rows[cell.row]
            + self.columns[cell.col]
            + self.tiles[cell.tile]
            if c is not cell
        }

    def get_excluded(self, cell) -> {Cell}:
        """Return the set of values this cell *cannot* have (because of cells in its context)."""
        return {c.value for c in cell.context if c.value != 0}

    # get all the possible values for cell, that is, the complement of the excluded values
    def get_possibles(self, cell) -> {Cell}:
        """Return the set of values this cell *can* have (considering cells in its context)."""
        excluded = self.get_excluded(cell)
        return {c for c in range(1, self.side_length + 1) if c not in excluded}

    def get_density(self, cell) -> float:
        """Return the percentage of filled cells in this cell's context."""
        return len([x for x in cell.context if x.value != 0]) // len(cell.context)

    def swap_row_values(self, row1, row2, allow_different_tiles=False):
        """Swap the values of corresponding cells in the given rows.
        This is not solution-preserving if the rows are not in the same tile,
        so you have to specifically provide for that if that's what you want.
        """
        if not allow_different_tiles and row1 // self.rank != row2 // self.rank:
            raise ValueError(
                "Swapping rows from different tiles is not solution-preserving."
            )
        rows = self.rows
        for col in range(0, len(self.rows[row2])):
            rows[row2][col].value, rows[row1][col].value = (
                rows[row1][col].value,
                rows[row2][col].value,
            )

    def swap_column_values(self, col1, col2, allow_different_tiles=False):
        """Swap the values of corresponding cells in the given columns.
        This is not solution-preserving if the rows are not in the same tile,
        so you have to specifically provide for that if that's what you want.
        """
        if not allow_different_tiles and col1 // self.rank != col2 // self.rank:
            raise ValueError(
                "Swapping rows from different tiles is not solution-preserving."
            )
        columns = self.columns
        for row in range(0, len(self.columns[col2])):
            columns[col2][row].value, columns[col1][row].value = (
                columns[col1][row].value,
                columns[col2][row].value,
            )

    def swap_stack_values(self, stack1, stack2):
        """Swap the values of corresponding cells in all the tiles in the given stacks.
        (A _stack_ is a complete set of vertically-aligned tiles.)
        """
        for offset in range(0, self.rank):
            self.swap_column_values(
                stack1 * self.rank + offset, stack2 * self.rank + offset, True
            )

    def swap_band_values(self, band1, band2):
        """Swap the values of corresponding cells in all the tiles in the given bands.
        (A _band_ is a complete set of horizontally-aligned tiles.)
        """
        for offset in range(0, self.rank):
            self.swap_row_values(
                band1 * self.rank + offset, band2 * self.rank + offset, True
            )

    def copy(self) -> Board:
        """Return a deep copy of this board."""
        return Board(values=[c.value for c in self.cells])

    def values(self):
        """Return the array of values in this board.
        Values are returned left-to-right, top-to-bottom, and can be used to recreate the board.
        """
        return [cell.value for cell in self.cells]

    def html(self) -> str:
        """Return an HTML representation of this board."""
        html = "<table>"
        for row in self.rows:
            row_string = "<tr>"
            for cell in row:
                row_string += (
                    '<td align="center" style="text-align:center">'
                    + str(cell)
                    + "</td>"
                )
            row_string += "</tr>"
            html += row_string
        html += "</table>"
        return html

    def ascii(self):
        """Return a human-readable ascii fixed-width representation of the board."""
        output = ""
        for row_num, row in enumerate(self.rows, 1):
            for col_num, cell in enumerate(row, 1):
                output += str(cell) + " "
                if col_num < self.side_length:
                    if col_num % self.rank == 0:
                        output += "| "
                else:
                    output += "\n"
            if row_num < self.side_length:
                if row_num % self.rank == 0:
                    output += ("--" * self.rank + "+-") * (self.rank - 1)
                    output += "--" * self.rank + "\n"
        return output

    def __str__(self):
        """The printable form of a board is its ascii form."""
        return self.ascii()

    def __repr__(self):
        """The (almost) eval-able form of a board is its constructor (with its id for specificity)."""
        return "<Board({values}) at {id}>".format(id=id(self), values=self.values())

