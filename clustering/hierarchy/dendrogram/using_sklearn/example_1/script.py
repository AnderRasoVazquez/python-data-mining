from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [1,3],
    [1,4],
    [5,2],
    [5,1],
    [2,2],
    [7,2]
])

Z = hierarchy.linkage(X, method='single', metric='euclidean')
# method: single, complete, average... etc
# metric: str or function

dn = hierarchy.dendrogram(Z)

plt.savefig("result.png")
plt.show()
