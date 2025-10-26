import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random


#AELIF Model

def AELIFneuron(E_l=-75,V_th=-50,V_r=-80,delta_th=2,G_l=10,C_m=100,a=2,b=0.02,tau_sra=200,dt=0.001,tmax=1.5,Iappval=500,Iapp_start=0.5):

    #init time vector
    timevec=np.linspace(0,tmax,int(tmax/dt))

    #init membrane potential vector V_m
    V_m=np.zeros(timevec.shape)
    V_m[0]=E_l

    #init spike rate adaptation current I_sra
    I_sra=np.zeros(timevec.shape)
    I_sra[0]=0

    #applied current vector
    Iapp=np.zeros(timevec.shape)
    Iapp[int(Iapp_start/dt):int(timevec.shape[0]-Iapp_start/dt)]=Iappval

    #simulate! (fwd euler)
    for t in range(timevec.shape[0]):
        dv_dt=(G_l*(E_l-V_m[t]+delta_th*np.exp((V_m[t]-V_th)/delta_th))-I_sra[t]+Iappvec[t])/C_m
        dI_sra_dt=




       






print(AELIFneuron())



