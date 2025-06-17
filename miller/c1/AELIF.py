import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random
from LIFclass import LIFneuron


class AELIF(LIFneuron):
    '''ELIF/AELIF model'''
    
    def __init__(self,E_l=-70,C_m=100,G_l=10,
                 V_th=-50,V_r=-80,I_app=201,sigma_i=0.1):
        super().__init__(E_l,C_m,G_l,
                 V_th,V_r,I_app,sigma_i)
        


