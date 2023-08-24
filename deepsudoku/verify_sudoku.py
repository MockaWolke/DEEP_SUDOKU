import numpy as np

def verify(sudoku):
    if not isinstance(sudoku, np.ndarray):
        try:
            sudoku = np.array(sudoku)
        except: 
            raise Exception("Can't convert sudoku to np array")
    assert len(sudoku.shape) == 2, "Sudoku is not square"
    assert sudoku.shape[0] == sudoku.shape[1], "Sudoku is not square"
    
    sq_size = sudoku.shape[0]
    size = np.sqrt(sq_size)
    assert size.is_integer(), "Invalid Sudoku size"
    size = size.astype(int)

    sum_ = np.sum(np.arange(sq_size+1))

    for ind in range(sq_size):
        if np.sum(sudoku[ind,:]) != sum_:
            return False
        if np.sum(sudoku[:,ind]) != sum_:
            return False
    
    for ind in np.ndindex((size,size)):
        if np.sum(sudoku[ ind[0]*size:ind[0]*size+size, ind[1]*size:ind[1]*size+size ]) != sum_:
            return False

    return True    