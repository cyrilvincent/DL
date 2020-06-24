import math
from typing import List

sigmoidFn = lambda x: 1 / (1 + math.exp(x * -1))
dsigmoidFn = lambda x: x * (1 - x)
tanhFn = lambda x: 0.5 + 0.5 * math.tanh(x / 2)
reluFn = lambda x: max(0, x)
toBoolFn = lambda x: x > 0
identityFn = lambda x: x
constantFn = lambda x : 1

class Perceptron:

    def __init__(self, weigths:List[float] = None, activationFn = reluFn):
        self.weigths:List[float] = weigths
        self.activationFn = activationFn
        self.output:float = None
        self.inputs:List[float] = []
        if self.weigths != None:
            self.inputs=[0 for _ in self.weigths]

    @property
    def signal(self)->float:
        try:
            total = sum([input * weight for input, weight in zip(self.inputs, self.weigths)])
        except:
            raise ValueError(f"Bad number of inputs for signal {self.inputs} vs {self.weigths} on {self}")
        return total

    def propagation(self, inputs:List[float])->float:
        if len(inputs) == len(self.weigths) - 1:
            inputs.append(1)
            # no bias <=> input with value 1
        if len(inputs) != len(self.weigths):
            raise ValueError(f"Bad number of inputs {inputs} vs {self.weigths} on {self}")
        self.inputs = inputs
        self.output = self.activationFn(self.signal)
        return self.output

    def __repr__(self):
        s = f"P "
        for i,w in zip(self.inputs, self.weigths):
            s+=f"+{i}*{w}"
        s += f"={self.signal}=>{self.output}"
        return s

if __name__ == '__main__':
    # Without bias
    p = Perceptron([0.4, 0.2], activationFn=sigmoidFn)
    p.propagation([0.1,0.6])
    print(p)

    # With bias 0.5
    p = Perceptron([0.4, 0.2, 0.5], activationFn=sigmoidFn)
    p.propagation([0.1, 0.6])
    print(p)
