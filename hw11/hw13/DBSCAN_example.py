from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)


model = DBSCAN(eps=0.2, min_samples=5)
labels = model.fit_predict(X)


plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Paired')
plt.title("DBSCAN 分群結果")
plt.show()
