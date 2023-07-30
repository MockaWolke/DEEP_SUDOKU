import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

def string_to_array(string : str):
    
    string = string.replace(".","0")
    
    return np.array([int(c) for c in string], np.uint8).reshape(int(np.sqrt(len(string))), -1)

def visualize_sudoku(sudoko, return_fig = False, figsize = (6,6), dpi = 100 ):
    
    if isinstance(sudoko, str):
    
        field = string_to_array(sudoko)

    elif isinstance(sudoko, np.ndarray):
        
        if sudoko.shape != (9,9):
            raise ValueError("False Shape")
        
        field = sudoko
        
    white = np.ones_like(field).astype(np.float32) 

    cmap = colors.LinearSegmentedColormap.from_list("", ["green","white"])

    fig = plt.figure(figsize=figsize, dpi = dpi)
    plt.imshow(white, cmap=cmap, vmin = 0, vmax = 1)
    plt.axis("off")
    # Draw grid lines
    for i in range(10):
        if i % 3 == 0:
            plt.vlines(i-0.5, -0.5, 8.5, colors='black', linewidth=2)
            plt.hlines(i-0.5, -0.5, 8.5, colors='black', linewidth=2)
        else:
            plt.vlines(i-0.5, -0.5, 8.5, colors='black', linewidth=1)
            plt.hlines(i-0.5, -0.5, 8.5, colors='black', linewidth=1)

    # now plot the numbers of field 
    for i in range(9):
        for j in range(9):
            if field[i, j] != 0:
                plt.text(j, i, str(field[i, j]), ha='center', va='center', color='black', fontsize=16)

    if return_fig:
        return fig
    
    plt.show()
