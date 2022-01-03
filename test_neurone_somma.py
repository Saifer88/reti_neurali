from neurone_somma import NeuroneSomma



ns = NeuroneSomma()

print (ns.getOutput(10, 12))


ns.train(300,0.0005)

print (ns. getOutput(10, 12))