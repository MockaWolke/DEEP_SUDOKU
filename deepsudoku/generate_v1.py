import numpy as np
from typing import Dict, Set, Tuple, List
import random


def generate(size: int, field: Dict[Tuple[int, int], int], rows: Dict[int, Set[int]], cols: Dict[int, Set[int]], blocks: Dict[Tuple[int, int], Set[int]], idx: List[Tuple[int, int]]) -> bool:
    
    if not idx:
        return True
    
    y, x = idx[0]
    block_tuple = (y//size, x//size)
    
    # Find possible numbers for current cell
    choose_from = list(rows[y].intersection(cols[x]).intersection(blocks[block_tuple]))
    
    if not choose_from:
        return False
    
    random.shuffle(choose_from)
    
    for number in choose_from:
        rows[y].discard(number)
        cols[x].discard(number)
        blocks[block_tuple].discard(number)
        
        if generate(size, field, rows, cols, blocks, idx[1:]):
            field[(y, x)] = number
            return True
        
        # Restore numbers if the current number doesn't lead to a solution
        rows[y].add(number)
        cols[x].add(number)
        blocks[block_tuple].add(number)
    
    return False

def construct_puzzle_solution(size):
    
    sq_size = np.square(size)
    field = {}
    numbers = set(range(1, sq_size+1))
    
    # Initialize possible numbers for each row, column, and block
    rows = {i: numbers.copy() for i in range(sq_size)}
    cols = {i: numbers.copy() for i in range(sq_size)}
    blocks = {(i, j): numbers.copy() for i in range(size) for j in range(size)}
    index = list(np.ndindex(sq_size, sq_size))

    generate(size, field, rows, cols, blocks, index)
    
    return np.array([[field[(y,x)] for x in range(sq_size)] for y in range(sq_size)], dtype=np.int32)


class Generator:
    #TODO implement very simple and fast generator for testing purposes
    def __init__(self, size):
        self.size = size
        #size**2 = row/ column length of sudoku

    def generate_one(self):
        # return size**2 x size**2 np array representing the sudoku field
        # with some missing ditits and the solution of the field
        sol = construct_puzzle_solution(self.size)

        # Stupid approach for removing fields so that the solution is still unique: just do pairwise multiplication
        # with a random permutation matrix
        P = np.eye(np.square(self.size), dtype=np.int32) 
        np.random.shuffle(P)  # shuffles rows

        return np.multiply(sol, 1-P), sol
