import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from LIFclass import LIFneuron

#default neuron with varying applied current---plotting firing rate


Iapps=np.linspace(150,400,101)
frates_sim=np.zeros(len(Iapps))
frates_theo=np.zeros(len(Iapps))
tmin,tmax=0,1000

#get all constant attributes
default=LIFneuron()
attr=default.get(['leak potential','membrane capacitance','leak conductance',
       'threshold potential','reset potential'])
tau=attr['membrane capacitance']/attr['leak conductance']
stim_time=(tmax-tmin)*0.6


for iappind in trange(len(Iapps)):

    #simulated firing rates
    neuron=LIFneuron(I_app=Iapps[iappind])
    t,v,i,spikeind=neuron.simulate()
    
    if len(spikeind)>0:
        frates_sim[iappind]=len(spikeind)/stim_time
    else:
        frates_sim[iappind]=0


    #theoretical firing rates

    #conditions where no solution exists:
    upper=(Iapps[iappind]/attr['leak conductance']+attr['leak potential']
         -attr['reset potential'])
    lower=(Iapps[iappind]/attr['leak conductance']
             +attr['leak potential']-attr['threshold potential'])

    #first condition: catch condition where log is negative---no solution
    condition1=(upper<0 and lower>0) or (upper>0 and lower<0)

    #second condition: catch condition where denominator in log is 0---indicates
    #steady state potential==threshold potential---no spiking
    condition2=(lower==0 )

    #check for condition and calculate solution if it exists
    if condition1 or condition2:
        frates_theo[iappind]=0
    else:
        log_this=upper/lower
        f=1/(tau*np.log(log_this))
        frates_theo[iappind]=f

comparison=plt.figure()
plt.plot(Iapps,frates_sim,label="simulated")
plt.plot(Iapps,frates_theo,label="theoretical")
plt.xlabel("Applied current($pA$)",fontsize='xx-large')
plt.ylabel("firing rate",fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.show()
