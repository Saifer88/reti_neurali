import random as rd
import math as mt

class ReteNeurale:

    def __init__(self):
        #rd.seed(1)
        self.w1 = 0
        self.w2 = 0
        self.bias = 0
        self.lr = 0.1

        print("Rete neurale creata, ", self.w1, self.w2, self.bias)

    def sig(self,t):
        return 1/(1+mt.exp(-t))

    def sig_p(self, t):
        return self.sig(t)*(1 - self.sig(t))

    def prediction(self, x1, x2):
        return self.sig(x1*self.w1 + x2*self.w2 + self.bias)

    def test(self, dataset):
        right = 0
        wrong = 0

        for element in dataset:
            result = self.prediction(element[0],element[1])
            if (result >= 0.5 and element[2] == 1) or (result < 0.5 and element[2] == 0):
                right += 1
            else :
                wrong += 1
        
        print ("right:", right, "wrong", wrong)


    def train(self, dataset):

        for i in range(100000):

            randomDatasetElementIndex = rd.randint(0, len(dataset)-1)
            randomDatasetElement = dataset[randomDatasetElementIndex]

            target = randomDatasetElement[2]

            prediction = self.prediction(randomDatasetElement[0], randomDatasetElement[1])

            cost = (prediction - target)**2
            cost_p = 2 * (prediction - target)

            t = randomDatasetElement[0]*self.w1 + randomDatasetElement[1]*self.w2 + self.bias

            pred_p = self.sig_p(t)

            dt_dw1 = randomDatasetElement[0]
            dt_dw2 = randomDatasetElement[1]
            dt_bias = 1

            cost_p_dz = cost_p * pred_p

            d_cost_dw1 = cost_p_dz * dt_dw1
            d_cost_dw2 = cost_p_dz * dt_dw2
            d_cost_db = cost_p_dz * dt_bias

            self.w1 = self.w1 - self.lr * d_cost_dw1
            self.w2 = self.w2 - self.lr * d_cost_dw2
            self.bias = self.bias - self.lr * d_cost_db










            









