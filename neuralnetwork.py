import micrograd
import random

random.seed(1)

class Neuron:
    def __init__(self, no_of_neurons_last):
        self.weight = [micrograd.value(random.random()) for _ in range(no_of_neurons_last)]
        self.bias = micrograd.value(random.random())
        
    def predict(self, inputs):
        out = sum(( x * w for w,x in zip(self.weight, inputs)), self.bias)
        return out

class Layer:
    def __init__(self, no_of_neurons, no_of_neurons_last): 
        self.no = no_of_neurons
        self.neurons = [Neuron(no_of_neurons_last) for _ in range(no_of_neurons)]

class MLP:
    def __init__(self, *args):
        self.arg=args
        self.layers = [Layer(args[layer_current], args[layer_current-1]) for layer_current in range(1,len(args))]

    def predict(self, *inputs):
        l1 = []
        l2 = list(inputs)
        for _layer in self.layers:
            l1 = l2.copy()
            l2 = []
            for _neuron in _layer.neurons:
                l2.append(_neuron.predict(l1))
        return l2
