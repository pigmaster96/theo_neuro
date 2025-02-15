import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random
from LIFclass import LIFneuron

class LIFneuronSRA(LIFneuron):
    '''separate class with spike rate adaptation incorporated'''
    
    def __init__(self,E_l=-70,C_m=100,G_l=10,
                 V_th=-50,V_r=-80,I_app=201,sigma_i=0.1):
        super().__init__(E_l,C_m,G_l,
                 V_th,V_r,I_app,sigma_i)
        
    def simulate(self,tmin=0,tmax=1000,dt=0.01,unstim_prop=0.2,noise=False,
                 SRA=True):
        '''with option for SRA'''
        t=np.arange(tmin,tmax,dt) #init time vector
        V=np.zeros(len(t)) #init results vector
        Iappvec=np.zeros(len(t)) #init vector for applied current
        spikeind=[] #init array for spike indices

        tau=self.C_m/self.G_l #time constant
        unstim_time=round(len(t)*unstim_prop) #number of timesteps before and after stim

        Iappvec[unstim_time:len(t)-unstim_time]=self.I_app #applied current magnitude
        V[0]=self.E_l #set initial potential to leak potential




    
test=LIFneuronSRA()
test1_1,test1_2,test1_3,test1_4=test.simulate()

plt.figure()
plt.plot(test1_1,test1_2)
plt.show()

