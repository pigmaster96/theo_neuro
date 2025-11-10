import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

#expandbin to both upsample and downsample 2d arrays 
#this might not work if the new and old bin widths are not nice proportions of each other

def upsample2d(vec,dt_old,dt_new):

    #expect old dt to be higher
    new_len=int(np.round(vec.shape[1]*(dt_old/dt_new)))

    #init new array
    new_vec=np.zeros((vec.shape[0],new_len))
    
    for dt in range(vec.shape[1]):
       new_vec[:,dt*int(np.round(dt_old/dt_new)):(dt+1)*int(np.round(dt_old/dt_new))]=np.tile(vec[:,dt][:,np.newaxis],(1,int(np.round(dt_old/dt_new))))
    
    return new_vec






