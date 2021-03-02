
from sklearn.cluster import MeanShift as MSClustering
from MLalgorithms._Clustering import Clustering


class MeanShift(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300):
		self.bandwidth = bandwidth
		self.max_iter = max_iter
		self.n_jobs = n_jobs
		self.seeds = seeds
		self.min_bin_freq = min_bin_freq
		self.bin_seeding = bin_seeding
		self.cluster_all = cluster_all
		self.model = MSClustering(seeds = self.seeds,
			max_iter = self.max_iter,
			bin_seeding = self.bin_seeding,
			cluster_all = self.cluster_all,
			n_jobs = self.n_jobs,
			min_bin_freq = self.min_bin_freq,
			bandwidth = self.bandwidth)

