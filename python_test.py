import math as mt
import random as rd
from rete_neurale import ReteNeurale

dataset=[[9,7.0,0],
[2,5.0,1],
[3.2,4.94,1],
[9.1,7.46,0],
[1.6,4.83,1],
[8.4,7.46,0],
[8,7.28,0],
[3.1,4.58,1],
[6.3,9.14,0],
[3.4,5.36,1]]    

rn = ReteNeurale()
rn.test(dataset)
rn.train(dataset)
print ("after train", rn.w1, rn.w2, rn.bias)
rn.test(dataset)


print ("Previsione:", rn.prediction(2, 5.0))

