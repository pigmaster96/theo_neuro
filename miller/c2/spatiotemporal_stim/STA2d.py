import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from spatiotemporalneuron import stimulus_new,spikebool,dt

def STA2d(Iapp,spikes,dt,tminus=75e-3,tplus=25e-3):

    #timesteps for interval around spike
    nminus=int(np.round(tminus/dt))
    nplus=int(np.round(tplus/dt))

    #tcorr(x axis for plotting)
    tcorr=np.arange(-nminus*dt,nplus*dt,dt) 

    #init STA
    sta=np.zeros((Iapp.shape[0],len(tcorr)))

    #spike indices
    spikeinds=np.nonzero(spikes)[0]

    #loop and sum over window around each spike
    for ind in spikeinds:
        if ind-nminus<0 or ind+nplus>=Iapp.shape[1]:
            continue
        else:
            sta+=Iapp[:,ind-nminus:ind+nplus]
    sta=sta/len(spikeinds)

    return sta,tcorr
    
sta,tcorr=STA2d(stimulus_new,spikebool,dt=dt)

plt.figure()
plt.contourf(tcorr,np.array([i+1 for i in range(40)]),sta)
plt.show()

