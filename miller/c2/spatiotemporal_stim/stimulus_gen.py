import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange

rng=np.random.default_rng()
stimulus=rng.uniform(low=-.5e-9,high=.5e-9,size=(40,40000))
weight=np.zeros((stimulus.shape[0],1))

#initialise weight vector
for x in range(stimulus.shape[0]):
    weight[x]=np.cos(4*np.pi*(x+1-20.5)/40)*np.exp(-16*((x+1-20.5)/40)**2)





    


    



plt.figure()
plt.plot(weight)
plt.show()


