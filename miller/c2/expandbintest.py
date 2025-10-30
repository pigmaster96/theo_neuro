import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from expandbin import downsampler
from time_varying_stimulus import Iapp_upsampled, dt


new_Iapp=downsampler(Iapp_upsampled,dt,1e-3)



print(Iapp_upsampled[245:255])
print(new_Iapp[0:9])


