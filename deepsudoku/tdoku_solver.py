from ctypes import *
from deepsudoku.utils import Board
from deepsudoku import PATH_TO_TDOKU_BIN


class TdokuSolver:
    """Very Fast TDOKU SOlVER Using https://github.com/t-dillon/tdoku.
    Code mostly taken from it"""

    def __init__(self):

        
        self.__tdoku = CDLL(PATH_TO_TDOKU_BIN)

        self.__solve = self.__tdoku.TdokuSolverDpllTriadSimd
        self.__solve.restype = c_ulonglong

        self.__constrain = self.__tdoku.TdokuConstrain
        self.__constrain.restype = c_bool

        self.__minimize = self.__tdoku.TdokuMinimize
        self.__minimize.restype = c_bool

    def board_as_string(self, board):

        return "".join([str(c) for c in board.cells]).replace("-", ".")

    def is_solvable(self, board):

        as_string = self.board_as_string(board)

        return bool(self.solver.Solve(as_string)[0])
        

    def Solve(self, puzzle):
        if type(puzzle) is str:
            puzzle = str.encode(puzzle)
        limit = c_ulonglong(1)
        config = c_ulong(0)
        solution = create_string_buffer(81)
        guesses = c_ulonglong(0)
        count = self.__solve(
            c_char_p(puzzle), limit, config, solution, pointer(guesses)
        )
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
        count = self.__solve(
            c_char_p(puzzle), limit, config, solution, pointer(guesses)
        )
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

