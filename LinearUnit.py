# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:07:48 2020

@author: rayts
"""

from __future__ import print_function
from functools import reduce
import matplotlib.pyplot as plt

class VectorOp(object):
    @staticmethod
    def dot(x, y):
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)
    
    @staticmethod
    def element_multiply(x, y):
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))
    
    @staticmethod
    def element_add(x, y):
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))
    
    @staticmethod
    def scala_multiply(v, s):
        return map(lambda e: e * s, v)
    
class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0] * input_num
        self.bias = 0.0
        
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
    
    def predict(self, input_vecs):
        return self.activator(VectorOp.dot(input_vecs, self.weights) + self.bias)
    
    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
    
    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for(input_vecs, label) in samples:
            output = self.predict(input_vecs)
            self._update_weights(input_vecs, output, label, rate)
            
    def _update_weights(self, input_vecs, output, label, rate):
        delta = label - output
        self.weights = VectorOp.element_add(self.weights, VectorOp.scala_multiply(input_vecs, rate * delta))
        self.bias += rate * delta


f = lambda x: x
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)
        
def get_training_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu

def plot(linear_unit):
    import matplotlib.pyplot as plt
    input_vecs, labels = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(map(lambda x: x[0], input_vecs), labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0,12,1)
    y = map(lambda x:weights[0] * x + bias, x)
    ax.plot(x, y)
    plt.show()
    
if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)
        
        
        
        
        
        
        
        