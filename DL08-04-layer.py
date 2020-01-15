import math
from typing import List

constantFn = lambda x : 1
identityFn = lambda x: x
reluFn = lambda x: max(0, x)
hardSigmoidFn = lambda x: min(max(0, x + 2), 4)
leluFn = lambda x, alpha: x * alpha if x < 0 else x
tanhFn = lambda x: 0.5 + 0.5 * math.tanh(x / 2)
sigmoidFn = lambda x: 1 / (1 + math.exp(x * -1))

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

if __name__ == '__main__':

    hl = Layer("HL")
    hl.perceptrons.append(Perceptron("H1", [0.15,0.2,-0.35], sigmoidFn))
    hl.perceptrons.append(Perceptron("H2", [0.25,0.3,-0.35], sigmoidFn))
    hl.perceptrons[0].propagation([0.05, 0.1])
    hl.perceptrons[1].propagation([0.05,0.1])
    for p in hl.perceptrons:
        print(p)


