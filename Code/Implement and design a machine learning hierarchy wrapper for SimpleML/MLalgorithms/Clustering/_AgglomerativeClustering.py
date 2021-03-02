
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.n_clusters = n_clusters
		self.memory = memory
		self.compute_distances = compute_distances
		self.affinity = affinity
		self.linkage = linkage
		self.distance_threshold = distance_threshold
		self.connectivity = connectivity
		self.compute_full_tree = compute_full_tree
		self.model = AC(compute_distances = self.compute_distances,
			distance_threshold = self.distance_threshold,
			affinity = self.affinity,
			connectivity = self.connectivity,
			linkage = self.linkage,
			n_clusters = self.n_clusters,
			memory = self.memory,
			compute_full_tree = self.compute_full_tree)

