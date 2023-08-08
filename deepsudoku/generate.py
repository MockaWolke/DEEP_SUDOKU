from deepsudoku import REPO_PATH, TDOKU_AVAILABLE, PATH_TO_TDOKU_BIN
from deepsudoku.utils import Board
import random
import tqdm
import multiprocessing
import warnings
from deepsudoku.norvig_solver import NorvigSolver
import numpy as np
from typing import Dict, Set, Tuple, List


def generate2(field: Dict[Tuple[int, int], int], rows: Dict[int, Set[int]], cols: Dict[int, Set[int]], blocks: Dict[Tuple[int, int], Set[int]], idx: List[Tuple[int, int]]) -> bool:
    
    if not idx:
        return True
    
    y, x = idx[0]
    block_tuple = (y//3, x//3)
    
    # Find possible numbers for current cell
    choose_from = list(rows[y].intersection(cols[x]).intersection(blocks[block_tuple]))
    
    if not choose_from:
        return False
    
    random.shuffle(choose_from)
    
    for number in choose_from:
        rows[y].discard(number)
        cols[x].discard(number)
        blocks[block_tuple].discard(number)
        
        if generate2(field, rows, cols, blocks, idx[1:]):
            field[(y, x)] = number
            return True
        
        # Restore numbers if the current number doesn't lead to a solution
        rows[y].add(number)
        cols[x].add(number)
        blocks[block_tuple].add(number)
    
    return False

def construct_puzzle_solution() -> List[int]:
    
    field = {}
    numbers = set(range(1, 10))
    
    # Initialize possible numbers for each row, column, and block
    rows = {i: numbers.copy() for i in range(9)}
    cols = {i: numbers.copy() for i in range(9)}
    blocks = {(i, j): numbers.copy() for i in range(3) for j in range(3)}
    index = list(np.ndindex(9, 9))
    
    generate2(field, rows, cols, blocks, index)
    
    return [field[(y,x)] for y in range(9) for x in range(9)]
    

def construct_puzzle_solution2() -> List[int]:
    size = 3
    Sudoku = np.empty((size**2, size**2, size**2), dtype=np.uint8)
    Sudoku[:] = np.ones(size**2)
    recursive_removal(Sudoku, list(np.ndindex(Sudoku.shape[:2])))
    return convert_to_sudoku(Sudoku).flatten()


# my implementation 2
def convert_to_sudoku(board):
    r = np.zeros(board.shape[:2])
    for y, x in np.ndindex(board.shape[:2]):
        cardinality = np.sum(board[y,x])
        if cardinality == 1:
            r[y,x] = np.nonzero(board[y,x])[0] + 1
        else:
            r[y,x] = 0
    return r



def recursive_removal(board, indices):

    if not indices:
        # board is finished
        return True

    # find index of field with largest number of constraints, ie. smallest number of possible values
    (ind_, min_) = ((-1,-1), board.shape[2]+1)
    for y, x in indices:
        cardinality = np.sum(board[y,x])
        if cardinality == 0:
            # if there is a field with zero possibilities, we failed
            #print("There are 0 possibilities at ", y, " ", x)
            return False
        elif cardinality < min_:
            (ind_, min_) = ((y,x), cardinality)
    

    # ind_ contains the index of the field with the smallest number of constraints
    indices.remove(ind_)
    # we pick a possible value at random, remember it and remove it from all constrainees
    # then we recursively proceed

    inds = np.nonzero(board[ind_])[0]
    # for a random pick, we shuffle
    np.random.shuffle(inds)

    board[ind_] = 0

    #print(inds, " @ ", ind_)

    for ind in inds:

        # set constrainees to false:
        y_, x_ = ind_
        by = (y_ // size) * size #bolck coordinates
        bx = (x_ // size) * size #block coordinates

        #backup rows/columns if we have to revert changes, see below
        column_backup = np.copy(board[y_,:,ind])
        row_backup = np.copy(board[:,x_,ind])
        block_backup = np.copy(board[by:by+size, bx:bx+size, ind])
        
        board[y_,:,ind] = 0 #column
        board[:,x_,ind] = 0 #row
        board[by:by+size, bx:bx+size, ind] = 0 #block

        # set this field to ind
        board[ind_][ind] = 1

        #print_sod(board)


        if recursive_removal(board, indices):
            return True


        # we failed :(
        # revert changes
        #print("now reverting changes @ ", ind_)
        # Problem: We cannot just revert by re-allowing ind for all the other fields, because these may already have been constrained.
        # Therefore we have to use the backups from above
        board[y_,:,ind] = column_backup
        board[:,x_,ind] = row_backup
        board[by:by+size, bx:bx+size, ind] = block_backup
        board[ind_][ind] = 0
    
    # all indices failed, therefore we have to backtrack.
    indices.append(ind_)
    board[ind_][inds]
    return False



    




class Solver():
    
    def __init__(self, use_tdoku = False) -> None:
    
        
        self.mode = "tdoku" if use_tdoku else "norvig"
    
        if use_tdoku:
            
            if not TDOKU_AVAILABLE:
                
                raise ValueError(f"TDOKU BINARY at {PATH_TO_TDOKU_BIN} does not exist.")
            
            from deepsudoku.tdoku_solver import TdokuSolver
            
            self.tdoku_solver = TdokuSolver()
            

    def is_solvable(self, board):

        if self.mode == "tdoku":
            
            return self.tdoku_solver.is_solvable(board)
        
        
        return NorvigSolver(self.board).can_solve()



class Generator:
    """
    Foundation of code from https://github.com/brotskydotcom/sudoku-generator/tree/develop
    Generate a problem puzzle by removing values from a solved puzzle,
    making sure that removal of the value doesn't introduce another solution.

    We sample a random solution.

    We then pick squares whose values should be removed.  We do this
    in two sequences:

    First, we examine each square in the puzzle in a random sequence, and if
    the immediate neighbors of that square (row, column, and tile) constrain
    it completely, we remove its value.  The result of this step is always
    "easy" to solve, because there is always at least one square which can
    only have one value.

    Second, we take all the remaining squares that have values
    and we sort them from most-constrained to least-constrained (with
    random ordering where there are ties).  For each of these squares,
    we see if the puzzle can be solved with a different value in the
    square.  If not, we remove the square's value.

    The more squares we remove in each step, the harder the puzzle should be
    to solve.  But because the first few squares removed in the second step
    are most likely to be those who are completely constrained by their
    neighbors, stopping the first step before it has tried to remove values
    from every square means the second step will first try to complete the
    first step.
    """

    # fraction of single-valued and random squares to remove based on the difficulty level
    thresholds = {
        "easy": {"4": (6, 0), "9": (27, 0), "16": (64, 0)},
        "medium": {"4": (9, 1), "9": (41, 5), "16": (96, 16)},
        "hard": {"4": (12, 2), "9": (54, 10), "16": (128, 33)},
        "extreme": {"4": (16, 3), "9": (81, 15), "16": (184, 49)},
        "insane": {"4": (16, 4), "9": (81, 20), "16": (256, 66)},
    }

    def __init__(
        self,
        side_length: str,
        difficulty: str,
        solver,
        upper_bound_missing_digist: int = None,
    ):

        """The starting point for the generator must be a solvable puzzle."""
        board = None

        self.solver = solver

        if difficulty in self.thresholds.keys():
            thresholds = self.thresholds[difficulty][side_length]
            self.first_cutoff, self.second_cutoff = thresholds
        else:
            raise ValueError("Unknown difficulty level: {}".format(difficulty))

        if side_length != "9":
            raise NotImplementedError()

        if upper_bound_missing_digist is not None:

            if upper_bound_missing_digist > sum(
                self.thresholds[difficulty][side_length]
            ):

                warnings.warn(
                    f"The upper_bound_missing_digist of {upper_bound_missing_digist} is to high for the diffuctly {difficulty}."
                )

            if difficulty == "easy":

                self.first_cutoff = upper_bound_missing_digist

            else:
                
                self.second_cutoff = upper_bound_missing_digist - self.first_cutoff

    def remove_values_1(self, board, cutoff):
        """Do the first pass at removing values from cells.
        Pick up to cutoff cells at random. Then remove each cell's value if
        it's the only possible value for that cell.
        """
        cells = board.get_filled_cells()
        random.shuffle(cells)
        for cell in cells:
            if cutoff <= 0:
                return
            if len(board.get_possibles(cell)) == 1:
                cell.value = 0
                cutoff -= 1

    def reduce_pass_2(self, board, cutoff, factor_in_density):
        """Do the second pass at removing values from cells.
        Pick up to cutoff cells sorted from highest to lowest density
        (breaking ties randomly).  Then remove each cell's value if doing so
        doesn't lead to another possible solution.
        """
        if factor_in_density:

            ranked_cells = [(x, board.get_density(x)) for x in board.get_filled_cells()]

            random.shuffle(ranked_cells)
            cells = [
                x[0] for x in sorted(ranked_cells, key=lambda x: x[1], reverse=True)
            ]
        else:

            cells = [x for x in board.get_filled_cells()]
            random.shuffle(cells)

        for cell in cells:
            if cutoff <= 0:
                return
            original = cell.value
            # for every other possible cell value, see if the board is solvable
            # if it is, then restore the original value so it isn't removed.
            for x in [val for val in board.get_possibles(cell) if val != original]:
                cell.value = x
                if self.is_solvable(board):
                    cell.value = original
                    break
            if cell.value != original:
                cell.value = 0
                cutoff -= 1

    def board_as_string(self, board):

        return "".join([str(c) for c in board.cells]).replace("-", ".")

    def is_solvable(self, board):

        return self.solver.is_solvable(board)

    def generate_one(self, factor_in_density: bool = True) -> dict:
        """Generate a new puzzle and solution.
        The returned dictionary has a 'puzzle' entry and a 'solution' entry.
        """

        solution = construct_puzzle_solution()

        board = Board(solution.copy())

        self.remove_values_1(board, self.first_cutoff)
        self.reduce_pass_2(board, self.second_cutoff, factor_in_density)

        solution = "".join([str(c) for c in solution])
        puzzle = self.board_as_string(board)

        return puzzle, solution


def generate_game(args):
    difficulty, factor_in_density, use_tdoku = args

    solver = Solver(use_tdoku)
    generator = Generator("9", difficulty, solver)
    return generator.generate_one(factor_in_density)


def generate_many_games(n_games, n_cpus, difficulty, factor_in_density, use_tdoku):
    with multiprocessing.Pool(n_cpus) as p:
        games = list(
            tqdm.tqdm(
                p.imap(generate_game, [(difficulty, factor_in_density, use_tdoku)] * n_games),
                total=n_games,
            )
        )
    return games

