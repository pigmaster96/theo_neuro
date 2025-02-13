import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

from LIFclass import LIFneuronV2


test=LIFneuronV2()


#clamp
test1_1,test1_2,test1_3,test1_4=test.simulate(refractory_model='clamp')

plt.figure("clamp")

plt.subplot(2,1,1)
plt.title('Clamp model')
plt.plot(test1_1,test1_2)
plt.ylabel("$V_m(mV)$")


plt.subplot(2,1,2)#indicate when spike occurs
#output gives spike indices, create logic vector for each timepoint
logicarr=np.zeros(len(test1_1))
logicarr[test1_4]=1
plt.plot(test1_1,logicarr)
plt.ylabel("Spike")
plt.xlabel("Time(ms)")


#raised threshold
test2_1,test2_2,test2_3,test2_4,test2_5=test.simulate(refractory_model='threshold')

plt.figure("threshold")

plt.subplot(2,1,1)
plt.title('Threshold model')
plt.plot(test2_1,test2_2,label="membrane potential")
plt.plot(test2_1,test2_5,'--',label="threshold potential")
plt.legend(loc='upper right')
plt.ylabel("$V_m(mV)$")

plt.subplot(2,1,2)#indicate when spike occurs
#output gives spike indices, create logic vector for each timepoint
logicarr=np.zeros(len(test2_1))
logicarr[test2_4]=1
plt.plot(test2_1,logicarr)
plt.ylabel("Spike")
plt.xlabel("Time(ms)")

#conductance
test3_1,test3_2,test3_3,test3_4,test3_5=test.simulate(refractory_model='conductance',
                                                      tau_g=2)

plt.figure('conductance')

plt.subplot(2,1,1)
plt.title('Conductance model')
plt.plot(test3_1,test3_2)
plt.ylabel("$V_m(mV)$")

plt.subplot(2,1,2)
plt.plot(test3_1,test3_5)
plt.ylabel('K conductance(nS)')
plt.xlabel('Time(ms)')

#threshold and conductance
test4_1,test4_2,test4_3,test4_4,test4_5,test4_6=test.simulate(refractory_model='threshold and conductance',
                                                              tau_g=2)

plt.figure('threshold and conductance')

plt.subplot(2,1,1)
plt.title("Threshold and Conductance model")
plt.plot(test4_1,test4_2,label='membrane potential')
plt.plot(test4_1,test4_5,'--',label='threshold potential')
plt.legend(loc='upper right')
plt.ylabel('$V_m(mV)$')

plt.subplot(2,1,2)
plt.plot(test4_1,test4_6)
plt.ylabel('K conductance(nS)')
plt.xlabel('Time(ms)')

plt.show()