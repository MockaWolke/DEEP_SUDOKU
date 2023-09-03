import numpy as np
from multiprocessing import Pool
import torch

def row_permutation(matrix):
    
    
    from_, to_ = np.random.choice(3, 2, replace = False)
    block = np.random.randint(0,3)
    from_ += block * 3
    to_ += block * 3
    matrix = matrix.copy()
    matrix[[to_,from_]] = matrix[[from_,to_]]
    return matrix

def col_permutation(matrix):

    
    from_, to_ = np.random.choice(3, 2, replace = False)
    block = np.random.randint(0,3)
    from_ += block * 3
    to_ += block * 3
    matrix = matrix.copy()
    matrix[:,[to_,from_]] = matrix[:,[from_,to_]]
    return matrix

def transpose_permutation(matrix):
    
    return np.transpose(matrix.copy(), (1, 0, 2))

def flip_permutation(matrix):
    
    axis = np.random.randint(0, 2)
    return np.flip(matrix.copy(),axis)

def rotate_permutation(matrix):
    
    times = np.random.randint(1, 4)
    
    return np.rot90(matrix.copy(), times)


PERMUTATIONS = [row_permutation, col_permutation, transpose_permutation, flip_permutation, rotate_permutation]
WEIGHTS = [0.35, 0.35, 0.1, 0.1, 0.1]


def n_permutations(ar : np.ndarray, n : int) -> np.ndarray:

    ar = ar.copy()

    todo = np.random.choice(PERMUTATIONS, n, p = WEIGHTS)

    for func in todo:
        ar = func(ar)
        
    return ar

def parralel_permutations(ar : np.ndarray, n_different : int,  n_permuts_each : int, n_cpus : int) -> np.ndarray:
    
    positions = np.arange(81).reshape(9, 9)
    
    ar = np.stack((ar, positions), -1)
    
    if n_cpus > 1:
    
        with Pool(n_cpus) as p:


            permuts = list(
                    p.starmap(n_permutations, [(ar, n_permuts_each)] * n_different),
            )

    else:
        
        permuts = [n_permutations(ar, n_permuts_each) for _ in range(n_different)]
        
    
    stacked = np.stack(permuts)
    
    return stacked
        
def ensemble(obs: np.ndarray, agent, n_different : int,  n_permuts_each : int, n_cpus : int, mode = "average"):
    
    
    assert mode in ["average","majority"]

    per = parralel_permutations(obs, n_different,  n_permuts_each, n_cpus )


    states, positions = per[:,:,:,0], per[:,:,:,1]

    positions = positions.reshape(-1, 81)
    
    
    states_to_torch = torch.tensor(states).to("cuda")

    with torch.no_grad():

        probs = agent.get_action_probs(states_to_torch)

    probs = probs.cpu().reshape(-1, 81, 9)


    indices = torch.tensor(np.argsort(positions, axis = -1))



    result_torch = torch.gather(probs, 1, indices.unsqueeze(2).expand(-1, -1, 9))
    
    
    
    if mode == "average":
        
        flattened = result_torch.mean(0).flatten()
        arg_max = torch.argmax(flattened).numpy()
        return np.unravel_index(arg_max,(9,9,9))
    
    elif mode == "majority":
        
        votes = torch.argmax(result_torch.reshape(-1, 9 **3), -1).numpy()
        
        unique, counts = np.unique(votes, return_counts = True)
        
        vote = unique[np.argmax(counts)]
        
        return np.unravel_index(vote,(9,9,9))
        
        
class EnsembleModel(torch.nn.Module):
    
    def __init__(self, agent, n_different : int,  n_permuts_each : int, n_cpus : int = 1, mode = "average") -> None:
        
        super(EnsembleModel, self).__init__()
        
        self.agent = agent
        self.n_different = n_different 
        self.n_permuts_each = n_permuts_each 
        self.n_cpus = n_cpus 
        self.mode = mode
        
    def forward(self, obs, get_action = "unraveled"):
        
        assert get_action == "unraveled", "Nothing else implemented"
        
        return ensemble(obs, self.agent, self.n_different, self.n_permuts_each ,self.n_cpus, self.mode)