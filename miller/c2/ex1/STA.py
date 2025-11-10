import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from expandbintest import new_Iapp,new_spikebool,dt

def STA(Iapp,spikes,dt,tminus=75e-3,tplus=25e-3):

    #timesteps for interval around spike
    nminus=int(np.round(tminus/dt))
    nplus=int(np.round(tplus/dt))

    #tcorr (x axis for plotting)
    tcorr=np.arange(-nminus*dt,nplus*dt,dt)

    #init STA
    sta=np.zeros(tcorr.shape)

    #spike indices
    spikeinds=np.nonzero(spikes)[0]

    #loop and calculate sum for window around each spike
    for ind in spikeinds:
        if ind-nminus<0 or ind+nplus>=len(Iapp):
            continue
        else:
            sta+=Iapp[ind-nminus:ind+nplus]
    sta=sta/len(spikeinds)

    return sta,tcorr

sta,tcorr=STA(Iapp=new_Iapp,spikes=new_spikebool,dt=dt)

plt.figure()
plt.plot(tcorr,sta)
plt.show()

    


