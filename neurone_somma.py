import math as mt
import random as rd


class NeuroneSomma:

    def __init__(self):

        self.w1 = 0
        self.w2 = 0
        self.lr = 0.0001
    
    def activationFunction(self, x1, x2):
        return x1*self.w1 + x2*self.w2

    
    def getOutput(self, x1, x2):
        return self.activationFunction(x1, x2)


    #as dataset we use random generated sums
    def train(self, trainingSet, learningRate):

        for i in range(trainingSet):

      
            x1 = rd.randint(0, 30)
            x2 = rd.randint(0, 30)

            target = x1 + x2 

            prediction = self.activationFunction(x1, x2)

            delta = target - prediction

            print(x1, x2, target, prediction, delta)

            self.w1 = self.w1 + (learningRate * delta * x1)
            self.w2 = self.w2 + (learningRate * delta * x2)


            print("epoc", i, "w1", self.w1, "w2", self.w2, "\t\t\t### 12 + 30 =", round(self.getOutput(12,30),3))

            if delta**2 <  0.0000000000001**2:
                break

