import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

from LIFclass import LIFneuron

default=LIFneuron(sigma_i=0.5)
t,V,i,spikeind=default.simulate(noise=True)

plt.figure()
plt.plot(t,V)
plt.show()
