from deepsudoku import REPO_PATH
from Sudoku.board import Board
from ctypes import *
import random
import tqdm
from Sudoku.board import Board
import multiprocessing
import warnings

class Solver:
    """Using https://github.com/t-dillon/tdoku"""
    def __init__(self):
        self.__tdoku = CDLL(REPO_PATH / "tdoku/build/libtdoku_shared.so")

        self.__solve = self.__tdoku.TdokuSolverDpllTriadSimd
        self.__solve.restype = c_ulonglong

        self.__constrain = self.__tdoku.TdokuConstrain
        self.__constrain.restype = c_bool

        self.__minimize = self.__tdoku.TdokuMinimize
        self.__minimize.restype = c_bool

    def Solve(self, puzzle):
        if type(puzzle) is str:
            puzzle = str.encode(puzzle)
        limit = c_ulonglong(1)
        config = c_ulong(0)
        solution = create_string_buffer(81)
        guesses = c_ulonglong(0)
        count = self.__solve(c_char_p(puzzle), limit, config, solution, pointer(guesses))
        if count:
            return count, solution.value.decode(), guesses.value
        else:
            return 0, "", guesses

    def Count(self, puzzle, limit=2):
        if type(puzzle) is str:
            puzzle = str.encode(puzzle)
        limit = c_ulonglong(limit)
        config = c_ulong(0)
        solution = create_string_buffer(81)
        guesses = c_ulonglong(0)
        count = self.__solve(c_char_p(puzzle), limit, config, solution, pointer(guesses))
        return count

    def Constrain(self, partial_puzzle):
        if type(partial_puzzle) is str:
            buffer = create_string_buffer(str.encode(partial_puzzle))
        else:
            buffer = create_string_buffer(partial_puzzle)
        pencilmark = c_bool(False)
        self.__constrain(pencilmark, buffer)
        return buffer.value

    def Minimize(self, non_minimal_puzzle):
        if type(non_minimal_puzzle) is str:
            buffer = create_string_buffer(str.encode(non_minimal_puzzle))
        else:
            buffer = create_string_buffer(non_minimal_puzzle)
        pencilmark = c_bool(False)
        monotonic = c_bool(False)
        self.__minimize(pencilmark, monotonic, buffer)
        return buffer.value


def construct_puzzle_solution():
    """Code from https://github.com/Kyubyong/sudoku/blob/master/generate_sudoku.py"""
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    while True:
        try:
            puzzle  = [[0]*9 for i in range(9)] # start with blank puzzle
            rows    = [set(range(1,10)) for i in range(9)] # set of available
            columns = [set(range(1,10)) for i in range(9)] #   numbers for each
            squares = [set(range(1,10)) for i in range(9)] #   row, column and square
            for i in range(9):
                for j in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i].intersection(columns[j]).intersection(squares[(i//3)*3 + j//3])
                    choice  = random.choice(list(choices))
        
                    puzzle[i][j] = choice
        
                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i//3)*3 + j//3].discard(choice)

            # success! every cell is filled.
            
            
            return [digit for row in puzzle for digit in row]
            
        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass

class Generator:
    """
    Foundation of code from https://github.com/brotskydotcom/sudoku-generator/tree/develop
    Generate a problem puzzle by removing values from a solved puzzle,
    making sure that removal of the value doesn't introduce another solution.

    Before we start removing squares, we pick one of a small set of solved
    puzzles and preform a sequence of randomly-chosen, correctness-preserving
    transformations on it.  (Without this step, we would have to have a large
    number of solutions to generate a large number of puzzles.)

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
        'easy': {'4': (6, 0), '9': (27, 0), '16': (64, 0)},
        'medium': {'4': (9, 1), '9': (41, 5), '16': (96, 16)},
        'hard': {'4': (12, 2), '9': (54, 10), '16': (128, 33)},
        'extreme': {'4': (16, 3), '9': (81, 15), '16': (184, 49)},
        'insane': {'4': (16, 4), '9': (81, 20), '16': (256, 66)}
    }

    def __init__(self, side_length: str, difficulty: str, solver, upper_bound_missing_digist : int = None):
        
        """The starting point for the generator must be a solvable puzzle."""
        board = None
        
        self.solver = solver
        

        
        if difficulty in self.thresholds.keys():
            thresholds = self.thresholds[difficulty][side_length]
            self.first_cutoff, self.second_cutoff = thresholds
        else:
            raise ValueError("Unknown difficulty level: {}".format(difficulty))
        
        if side_length != '9':
            raise NotImplementedError()
        
        if upper_bound_missing_digist is not None:
      
            if upper_bound_missing_digist > sum(self.thresholds[difficulty][side_length]):
                
                warnings.warn(f"The upper_bound_missing_digist of {upper_bound_missing_digist} is to high for the diffuctly {difficulty}.")

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
        
            ranked_cells = [(x, board.get_density(x)) for x in
                            board.get_filled_cells()]
        
            random.shuffle(ranked_cells)
            cells = [x[0] for x in
                    sorted(ranked_cells, key=lambda x: x[1], reverse=True)]
        else:
            
            cells = [ x for x in board.get_filled_cells()]
            random.shuffle(cells)
        
        
         
        for cell in cells:
            if cutoff <= 0:
                return
            original = cell.value
            # for every other possible cell value, see if the board is solvable
            # if it is, then restore the original value so it isn't removed.
            for x in [val for val in board.get_possibles(cell)
                                     if val != original]:
                cell.value = x
                if self.is_solvable(board):
                    cell.value = original
                    break
            if cell.value != original:
                cell.value = 0
                cutoff -= 1

    def board_as_string(self, board):

        return ''.join([str(c) for c in board.cells]).replace("-",".")

    def is_solvable(self, board):
        
        as_string = self.board_as_string(board)
        
        return bool(self.solver.Solve(as_string)[0])

    def generate_one(self, factor_in_density : bool = True) -> dict:
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
    difficulty, factor_in_density  = args
    
    solver = Solver()
    generator = Generator('9', difficulty, solver)
    return generator.generate_one(factor_in_density)

def generate_many_games(n_games, n_cpus, difficulty, factor_in_density):
    with multiprocessing.Pool(n_cpus) as p:
        games = list(tqdm.tqdm(p.imap(generate_game, [(difficulty, factor_in_density)] * n_games), total=n_games))
    return games



