import numpy as np
from functools import reduce

class RowPermutation:
    def __init__(self, from_, to_):
        self.from_ = from_
        self.to_ = to_
    
    def __call__(self, matrix):
        r = np.copy(matrix)
        r[[self.to_,self.from_]] = r[[self.from_,self.to_]]
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

def random_permutation(msize):
    inds = np.random.choice(np.arange(msize), size=2, replace=False)
    return np.random.choice([
        RowPermutation(inds[0],inds[1]),
        TransposePermutation(),
        MirrorPermutation(np.random.choice(np.array([0,1]))),
        SwapDigitsPermutation(inds[0],inds[1])
    ], p=[0.4,0.1,0.1,0.4]) 

def create_permutations(matrix, count):
    assert matrix.shape[0] == matrix.shape[1], "matrix not square"
    desired_perm_length = 2 + np.emath.logn(matrix.shape[0]*2, count).astype(int)

    permutations = {}
    permutations[tuple(map(tuple, matrix))] = [TransposePermutation(), TransposePermutation()]
    
    while len(permutations) < count:
        rp = [random_permutation(matrix.shape[0]) for i in range(desired_perm_length)]
        pm = tuple(map(tuple, compose(rp)(matrix)))
        permutations[pm] = rp

    return permutations

def preprocessing(observation):
    return tf.one_hot(observation, 10, axis=-1)

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
        qs = pi(tf.expand_dims(preprocessing(np.array(perm)), 0))
        act = np.argmax(qs)

        # unravel discrete action and apply to an empty board
        empty_board = np.zeros_like(obs_)
        y,x,n = np.unravel_index(act,(9,9,9))
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