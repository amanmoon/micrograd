import neuralnetwork
from micrograd import value
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
# %matplotlib inline
np.random.seed(137)
random.seed(137)
# make up a dataset

X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')



# initialize a model 
model = neuralnetwork.MLP(2,16, 16, 1) # 2-layer neural network


def loss(batch_size=None):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs=[list(map(value,xrow)) for xrow in Xb]
    scores=[]
    
    for x in inputs:
        scores.append(model.predict(x[0],x[1]))
    print(y)
    losses = [(1 + -(value(int(y[x])))*(scores[x][0])).relu() for x in range(len(scores))]
    data_loss = sum(losses) * (1.0 / len(losses))
    total_loss = data_loss 
    accuracy = [(yi > 0) == (scorei[0].data > 0) for yi, scorei in zip(yb, scores)]
    return   total_loss,sum(accuracy) / len(accuracy)


for k in range(100):
    
    # forward
    total_loss, acc = loss()
    
    # backward
    total_loss.runbackpropagation()
    
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.layers:
        for x in p.neurons:
            for y in x.weight:
                y.data -= learning_rate * y.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
