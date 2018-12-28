import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxsize

### Setup
# set random seed
rand.seed(42)

# 2 clusters
# not that both covariance matrices are diagonal
mu1 = [0, 5]
sig1 = [ [2, 0], [0, 3] ]

mu2 = [5, 0]
sig2 = [ [4, 0], [0, 1] ]

# generate samples
x1, y1 = np.random.multivariate_normal(mu1, sig1, 10).T
x2, y2 = np.random.multivariate_normal(mu2, sig2, 10).T

print("X1 points")
print(x1)
print("Y1 points")
print(y1)

xs = np.concatenate((x1, x2))
ys = np.concatenate((y1, y2))
labels = ([-1] * 10) + ([1] * 10)

data = {'x': xs, 'y': ys, 'label': labels}
df = pd.DataFrame(data=data)

print(df) 

# inspect the data
df.head()
df.tail()

fig = plt.figure()
plt.scatter(data['x'], data['y'], 24, c=data['label'])
fig.savefig("true-values.png")