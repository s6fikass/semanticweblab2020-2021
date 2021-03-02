
from sklearn.cluster import Birch as BirchClustering
from MLalgorithms._Clustering import Clustering


class Birch(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
		self.n_clusters = n_clusters
		self.compute_labels = compute_labels
		self.branching_factor = branching_factor
		self.copy = copy
		self.threshold = threshold
		self.model = BirchClustering(threshold = self.threshold,
			branching_factor = self.branching_factor,
			compute_labels = self.compute_labels,
			copy = self.copy,
			n_clusters = self.n_clusters)

