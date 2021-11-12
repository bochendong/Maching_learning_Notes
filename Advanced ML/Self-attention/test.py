import numpy as np

def softmax(x):
    return np.exp(x)/ (np.exp(16)+ np.exp(20)+np.exp(41)+np.exp(37))

print(softmax(16))
print(softmax(20))
print(softmax(41))
print(softmax(37))