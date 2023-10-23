import math
from typing import List

class Perceptron:

    def __init__(self, weigths:List[float]):
        self.weigths:List[float] = weigths
        self.output:float = None
        self.inputs:List[float] = []
        self.inputs=[0 for _ in self.weigths]

    @property
    def signal(self)->float:
        return sum([input * weight for input, weight in zip(self.inputs, self.weigths)])

    def propagation(self, inputs:List[float])->float:
        self.inputs = inputs
        self.output = self.signal
        return self.output

    def __repr__(self):
        s = f"P "
        for i,w in zip(self.inputs, self.weigths):
            s+=f"+{i}*{w}"
        s += f"={self.signal}=>{self.output}"
        return s

if __name__ == '__main__':
    p = Perceptron([0.4, 0.2])
    p.propagation([0.1,0.6])
    print(p)
