import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange

def downsample1d(vec,dt_old,dt_new):
    #calculate length of new vector (total time stays the same)
    scaling_ratio=int(np.round(dt_new/dt_old))
    new_len=int(len(vec)/scaling_ratio)

    #new array
    new_vec=np.zeros((new_len,))
    for block in range(new_len):
        new_vec[block]=np.mean(vec[block*scaling_ratio:(block+1)*scaling_ratio])

    return new_vec

    





