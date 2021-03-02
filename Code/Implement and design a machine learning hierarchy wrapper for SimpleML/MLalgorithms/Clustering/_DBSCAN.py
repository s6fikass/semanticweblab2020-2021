
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.p = p
		self.eps = eps
		self.metric = metric
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.min_samples = min_samples
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.model = DBSCANClustering(eps = self.eps,
			metric_params = self.metric_params,
			p = self.p,
			metric = self.metric,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			min_samples = self.min_samples,
			algorithm = self.algorithm)

