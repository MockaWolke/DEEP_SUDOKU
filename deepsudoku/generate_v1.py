import numpy as np
from typing import Dict, Set, Tuple, List


def generate(field: Dict[Tuple[int, int], int], rows: Dict[int, Set[int]], cols: Dict[int, Set[int]], blocks: Dict[Tuple[int, int], Set[int]], idx: List[Tuple[int, int]]) -> bool:
    
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


class Generator:
    #TODO implement very simple and fast generator for testing purposes
    def __init__(self, size):
        self.size = size
        #size**2 = row/ column length of sudoku

    def generate_one():
        # return size**2 x size**2 np array representing the sudoku field
        # with some missing ditits and the solution of the field
        pass
