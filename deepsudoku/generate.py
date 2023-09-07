"""Code to generate Sudokus"""
from deepsudoku import REPO_PATH, TDOKU_AVAILABLE, PATH_TO_TDOKU_BIN
from deepsudoku.utils import Board
import random
import tqdm
import multiprocessing
import warnings
from deepsudoku.norvig_solver import NorvigSolver
import numpy as np
from typing import Dict, Set, Tuple, List
import os
import glob
import pickle


def recu_generate(field: Dict[Tuple[int, int], int], rows: Dict[int, Set[int]], cols: Dict[int, Set[int]], blocks: Dict[Tuple[int, int], Set[int]], idx: List[Tuple[int, int]]) -> bool:
    """Recursion to generate a full field"""    
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
        
        if recu_generate(field, rows, cols, blocks, idx[1:]):
            field[(y, x)] = number
            return True
        
        # Restore numbers if the current number doesn't lead to a solution
        rows[y].add(number)
        cols[x].add(number)
        blocks[block_tuple].add(number)
    
    return False

def construct_puzzle_solution() -> List[int]:
    """Quickly find full field.  """
    field = {}
    numbers = set(range(1, 10))
    
    # Initialize possible numbers for each row, column, and block
    rows = {i: numbers.copy() for i in range(9)}
    cols = {i: numbers.copy() for i in range(9)}
    blocks = {(i, j): numbers.copy() for i in range(3) for j in range(3)}
    index = list(np.ndindex(9, 9))
    
    recu_generate(field, rows, cols, blocks, index)
    
    return [field[(y,x)] for y in range(9) for x in range(9)]
    



class Solver():
    """Solver wrapper to use either norvig or tdoku"""
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
        use_random_starting_point: bool = True,
    ):

        """The starting point for the generator must be a solvable puzzle."""
        board = None

        self.solver = solver
        self.use_random_starting_point = use_random_starting_point

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
                
                
        self.difficulty = difficulty
            
        self.starting_points = None
        
        if not self.use_random_starting_point:
            
            options = glob.glob(os.path.join(REPO_PATH,"saved_sudoku_junks/*.pkl"))
            
            chosen = random.choice(options)
            
            with open(chosen, "rb") as file:
                
                self.starting_points = pickle.load(file)
                
            
            
            
        

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

    def generate_one(self, factor_in_density: bool = True, cut_off:int = None) -> dict:
        """Generate a new puzzle and solution.
        The returned dictionary has a 'puzzle' entry and a 'solution' entry.
        """
        
        if self.use_random_starting_point:

         
            solution = construct_puzzle_solution()


        else: 
            
            solution = random.choice(self.starting_points)
            
            solution = [int(c) for c in solution]
            

        board = Board(solution.copy())

        if cut_off is None:

            self.remove_values_1(board, self.first_cutoff)
            
        else:
            
            self.remove_values_1(board, cut_off)
            
        
        if self.difficulty != "easy":
            self.reduce_pass_2(board, self.second_cutoff, factor_in_density)

        solution = "".join([str(c) for c in solution])
        puzzle = self.board_as_string(board)

        return puzzle, solution


def generate_game(args):

    difficulty, factor_in_density, use_tdoku, upper_bound = args

    solver = Solver(use_tdoku)
    generator = Generator("9", difficulty, solver, upper_bound_missing_digist =  upper_bound)
    return generator.generate_one(factor_in_density)


def generate_many_games(n_games, n_cpus, difficulty, factor_in_density, use_tdoku, upper_bound):
    
    with multiprocessing.Pool(n_cpus) as p:
        games = list(
            tqdm.tqdm(
                p.imap(generate_game, [(difficulty, factor_in_density, use_tdoku, upper_bound)] * n_games),
                total=n_games,
            )
        )
    
    return games

