import numpy as np
import matplotlib.pyplot as plt

#Forward Euler: dxdt=0.3x w/ initial condition x(0)=exp(0.3)
dt=0.5

#time vector
t=np.arange(1,5+dt/2,dt)

#get number of steps
n=len(t)

#initialise array for x values
x=np.zeros(n)

#set initial condition
x[0]=np.exp(0.3*t[0])

#now do fwd euler
for i in range(n-1): #-1 because we already set initial condition
    x[i+1]=x[i]+0.3*x[i]

#plotting
fig=plt.figure(figsize=(6,4))
plt.plot(t,x)
plt.show()

