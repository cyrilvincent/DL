import math
from typing import List

constantFn = lambda x : 1
identityFn = lambda x: x
reluFn = lambda x: max(0, x)
hardSigmoidFn = lambda x: min(max(0, x + 2), 4)
leluFn = lambda x, alpha: x * alpha if x < 0 else x
tanhFn = lambda x: 0.5 + 0.5 * math.tanh(x / 2)
sigmoidFn = lambda x: 1 / (1 + math.exp(x * -1))
mseFn = lambda x, y : 0.5 * math.pow(x - y, 2)
maeFn = lambda x, y : abs(x - y)

class Perceptron:

    def __init__(self, id:str, weigths:List[float] = None, activationFn = reluFn):
        self.id = id
        self.weigths:List[float] = weigths
        self.activationFn = activationFn
        self.output:float = None
        self.inputs:List[float] = []
        if self.weigths != None:
            self.inputs=[0 for _ in self.weigths]

    @property
    def signal(self)->float:
        total = 0
        try:
            total = sum([input * weight for input, weight in zip(self.inputs, self.weigths)])
        except:
            raise ValueError(f"Bad number of inputs for signal {self.inputs} vs {self.weigths} on {self}")
        return total

    def propagation(self, inputs:List[float])->float:
        if len(inputs) == len(self.weigths) - 1:
            inputs.append(1)
        if len(inputs) != len(self.weigths):
            raise ValueError(f"Bad number of inputs {inputs} vs {self.weigths} on {self}")
        self.inputs = inputs
        self.output = self.activationFn(self.signal)
        return self.output

    def __repr__(self):
        s = f"P-{self.id} "
        for i,w in zip(self.inputs, self.weigths):
            s+=f"+{i}*{w}"
        s += f"={self.signal}=>{self.output}"
        return s

class Layer:

    def __init__(self, id:str, nbPerceptron:int = 0, activationFn = reluFn):
        self.id = id
        self.perceptrons: List[Perceptron] = []
        for i in range(nbPerceptron):
            p = Perceptron(f"{id}-{i}", None, activationFn)
            self.perceptrons.append(p)

    def __repr__(self):
        return f"L-{self.id}:{len(self.perceptrons)}"

class MLP:

    def __init__(self, learningRate:float = 0.0001, lossFn = mseFn):
        self.learningRate = learningRate
        self.lossFn = lossFn
        self.layers:List[Layer] = []
        self.loss:float = 0

    def propagation(self, inputs:List[float])->None:
        for perceptron, input in zip(self.layers[0].perceptrons, inputs):
            perceptron.propagation([input])
        self._propagation()

    def _propagation(self)->None:
        for i in range(len(self.layers) - 1):
            layer = self.layers[i + 1]
            prevLayer = self.layers[i]
            for perceptron in layer.perceptrons:
                inputs = [p.output for p in prevLayer.perceptrons]
                perceptron.propagation(inputs)

    def outputs(self)->List[float]:
        return [p.output for p in self.layers[-1].perceptrons]

    def computeLoss(self, target: List[float])->float:
        try:
            return sum([self.lossFn(o, t) for o, t in zip(self.outputs(), target)])
        except:
            raise ValueError(f"Bad number of outputs {target} vs {self.layers[-1].perceptrons}")

    def __repr__(self):
        return f"MLP:{[l for l in self.layers]}"

if __name__ == '__main__':
    # https://mattmazur.com/2015/03/17/a-step-by-step-backPropagation-example/
    hl = Layer("HL")
    hl.perceptrons.append(Perceptron("H1", [0.15, 0.2, 0.35], sigmoidFn))
    hl.perceptrons.append(Perceptron("H2", [0.25, 0.3, 0.35], sigmoidFn))

    # MLP
    il = Layer("IL")
    il.perceptrons.append(Perceptron("I1", [1]))
    il.perceptrons.append(Perceptron("I2", [1]))
    network = MLP()
    network.layers.append(il)
    network.layers.append(hl)
    ol = Layer("OL")
    ol.perceptrons.append(Perceptron("O1", [0.4, 0.45, 0.6], sigmoidFn))
    ol.perceptrons.append(Perceptron("O2", [0.5, 0.55, 0.6], sigmoidFn))
    network.layers.append(ol)
    print(network)
    network.propagation([0.05, 0.1])
    print(network.outputs())
    print(network.computeLoss([0.01, 0.09]))