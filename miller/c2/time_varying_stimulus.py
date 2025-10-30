import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

from AELIF import AELIFneuron

#generate random array
Iapp=np.array([random.uniform(-.5e-9,.5e-9) for i in range(40000)])

#create time vector
bins=40000
interval=5e-3
tmax=bins*interval
dt=0.02e-3
tvec=np.arange(0,tmax,dt)


#upsample random array
Iapp_upsampled=np.zeros(len(tvec))
for ind in range(len(Iapp)):
    Iapp_upsampled[ind*int(np.round(interval/dt)):(ind+1)*int(np.round(interval/dt))]=Iapp[ind]


t,v,_,_,_=AELIFneuron(tmax=tmax,dt=dt,Iappvec=Iapp_upsampled)



plt.figure()
plt.plot(t,v)
plt.show()








