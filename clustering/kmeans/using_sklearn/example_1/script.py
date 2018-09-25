from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'g',
                   }

a = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9,11]])


kmeans = KMeans(n_clusters=2)
kmeans.fit(a)

for index, centroid in enumerate(kmeans.cluster_centers_):
    plt.scatter(centroid[0], centroid[1], marker="x", color=LABEL_COLOR_MAP[index])

for instance in a:
    plt.scatter(instance[0], instance[1], marker="o", color=LABEL_COLOR_MAP[kmeans.predict(
        [[instance[0], instance[1]]]
    )[0]
    ])

nuevos = np.array([[2, 2],
              [1.9, 7],
              [5, 12],
              [3, 8],
              [5, 8],
              [7,12]])

for instance in nuevos:
    plt.scatter(instance[0], instance[1], marker="*", color=LABEL_COLOR_MAP[kmeans.predict(
        [[instance[0], instance[1]]]
    )[0]
    ])


plt.savefig("result.png")
plt.show()
