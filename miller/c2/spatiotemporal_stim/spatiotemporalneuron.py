import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange

from AELIF import AELIFneuron

#init stimulus
bins=40000
space=40
rng=np.random.default_rng()
stimulus=rng.uniform(low=-.5e-9,high=.5e-9,size=(space,bins))#(space,time)

#initialise weight vector
#the idea here is that given a stimulus, this `weight' simulates the final input to an AELIF favouring stimuli with higher values in the middle coordinate; for instance the timepoints in the stimulus matrix where the favoured coordinate happens to be higher will be amplified more by the weight vector, simulating a neuron that favours that spatial arrangement.
favoured_coordinate=20.5
weight=np.zeros((stimulus.shape[0],1))

x=np.array([i+1 for i in range(stimulus.shape[0])])
weight=np.cos(4*np.pi*(x-20.5)/space)*np.exp(-16*((x-20.5)/40)**2)   

##to plot weight vector
#plt.figure()
#plt.plot(weight)
#plt.show()

#now weight the stimulus to create weighted_stim(how the neuron perceives this stimulus)
#the timepoints with randomly initialised stimuli favouring the predetermined arrangement will have a higher magnitude summed over all the spatial coordinates, so we take the dot product
weighted_stim=weight@stimulus

#create time vector 
interval=5e-3
tmax=bins*interval #simulation time
dt=0.02e-3
tvec=np.arange(0,tmax,dt)

#upsample from weighted stimulus for input into AELIF 
Iapp=np.zeros(len(tvec))
for ind in range(len(weighted_stim)):
    Iapp[ind*int(np.round(interval/dt)):(ind+1)*int(np.round(interval/dt))]=weighted_stim[ind]

#simulate!
_,_,_,_,spikes,spikebool=AELIFneuron(Iappvec=Iapp,dt=dt,tmax=tmax)


