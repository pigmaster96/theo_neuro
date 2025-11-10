import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from AELIF import AELIFneuron


#set parameters for standard deviation
dt=1e-2
sigma=50e-12
std=sigma*np.sqrt(dt)

#calculate simulation duration
tmax=100
Iapp_len=int(np.round(tmax/dt))

#normal distribution to determine applied current
rng=np.random.default_rng()
Iapp=rng.normal(scale=std,size=Iapp_len)

print(Iapp.shape)
print(Iapp)


#test=AELIFneuron(E_l=-70e-3,G_l=10e-9,a=2e-9,b=0,tau_sra=150,dt=dt,tmax=tmax,



