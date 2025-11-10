import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from expandbin import downsampler
from time_varying_stimulus import Iapp_upsampled,dt,tmax
from AELIF import AELIFneuron

#simulate using old Iapp
t,v,_,_,spikes,spikebool=AELIFneuron(Iappvec=Iapp_upsampled,dt=dt,tmax=tmax)

#print(len(spikes))
#print(len(spikebool))
#print(len(Iapp_upsampled))

#downsample Iapp from 0.2ms to 1ms (by a factor of 50)
new_Iapp=downsampler(Iapp_upsampled,dt,1e-3)

#downsample spike bool vector and set all nonzero values to 1
new_spikebool=np.ceil(downsampler(spikebool,dt,1e-3))





#print(int(np.sum(new_spikebool))==len(spikes))
##check that the number of spikes in both upsampled and downsampled match 


