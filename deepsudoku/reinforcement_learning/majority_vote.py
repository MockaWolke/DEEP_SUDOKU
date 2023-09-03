import numpy as np
from functools import reduce

class RowPermutation:
    def __init__(self, from_, to_, band, band_size):
        self.from_ = from_ + band*band_size
        self.to_ = to_ + band*band_size
    
    def __call__(self, matrix):
        r = np.copy(matrix)
        r[[self.to_,self.from_]] = r[[self.from_,self.to_]]
        #print("RowPermutation(", self.from_,",",self.to_,"):", r)
        return r

class BandPermutation:
    def __init__(self, from_, to_, band_size):
        self.from_ = from_
        self.to_ = to_
        self.band_size = band_size
    
    def __call__(self, matrix):
        r = np.copy(matrix)
        r[self.from_*self.band_size:(self.from_+1)*self.band_size] = matrix[self.to_*self.band_size:(self.to_+1)*self.band_size]
        r[self.to_*self.band_size:(self.to_+1)*self.band_size] = matrix[self.from_*self.band_size:(self.from_+1)*self.band_size]
        #print("RowPermutation(", self.from_,",",self.to_,"):", r)
        return r

class TransposePermutation:
    def __init__(self):
        pass

    def __call__(self, matrix):
        #print("TransposePermutation(", "):", matrix.T)
        return matrix.T


class MirrorPermutation:
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, matrix):
        #print("RotateClockwisePermutation(", self.nturns, "):", np.rot90(matrix, k=-self.nturns))
        return np.flip(matrix, axis=self.axis)


class SwapDigitsPermutation:
    def __init__(self, dig1, dig2):
        self.dig1 = dig1 + 1
        self.dig2 = dig2 + 1
    
    def __call__(self, matrix):
        swapper = np.where(matrix==self.dig1, self.dig2, self.dig1)
        r = np.where((matrix==self.dig1) | (matrix==self.dig2), swapper, matrix)
        #print("SwapDigitsPermutation(", self.dig1,",",self.dig2,"):", r)
        return r



compose = lambda F: reduce(lambda f, g: lambda x: g(f(x)), F)

def random_permutation(band_size):
    inds = np.random.choice(np.arange(band_size), size=3, replace=False)
    return np.random.choice([
        RowPermutation(inds[0],inds[1], inds[2], band_size),
        BandPermutation(inds[0],inds[1], band_size),
        TransposePermutation(),
        #MirrorPermutation(np.random.choice(np.array([0,1]))),
        SwapDigitsPermutation(inds[0],inds[1])
    ], p=[0.25,0.25,0.15,0.35]) 

def create_permutations(matrix, count):
    assert matrix.shape[0] == matrix.shape[1], "matrix not square"
    band_size = np.sqrt(matrix.shape[0])
    assert band_size.is_integer(), "Invalid Sudoku size"
    band_size = band_size.astype(int)

    desired_perm_length = 2 + np.emath.logn(matrix.shape[0]*2, count).astype(int)

    permutations = {}
    permutations[tuple(map(tuple, matrix))] = [TransposePermutation(), TransposePermutation()]
    
    while len(permutations) < count:
        rp = [random_permutation(band_size) for i in range(desired_perm_length)]
        pm = tuple(map(tuple, compose(rp)(matrix)))
        permutations[pm] = rp

    return permutations


def majority_vote(pi, obs, nvoters):
    """Perform majority vote

    Keyword arguments:
    pi -- policy
    obs -- observation (a 2d numpy array representing the sudoku)
    nvoters -- number of voters for the majority vote

    """
    votes = {}
    perms = create_permutations(obs, nvoters)
    for perm, perm_function in perms.items():
        obs_ = np.array(perm) #a 2d numpy array representing the sudoku, with elements eg. between 0 and 9
        
        # gather discrete action act from policy
        y,x,n = pi(np.array(perm)[None, : ])
        
        empty_board = np.zeros_like(obs_)
        empty_board[y,x] = n+1

        # reverse transform on the empty board
        rev = compose(reversed(perm_function))(empty_board)

        # find only non-zero entry in rev, that is the actual action
        nzind_ = next(zip(*np.nonzero(rev)))
        if nzind_ + (rev[nzind_]-1,) not in votes:
            votes[nzind_ + (rev[nzind_]-1,)] = 1
        else:
            votes[nzind_ + (rev[nzind_]-1,)] += 1
    
    # find majority vote
    return max(votes, key=votes.get)