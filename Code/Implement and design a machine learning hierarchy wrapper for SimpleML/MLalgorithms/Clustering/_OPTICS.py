
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.predecessor_correction = predecessor_correction
		self.p = p
		self.max_eps = max_eps
		self.xi = xi
		self.eps = eps
		self.metric = metric
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.min_cluster_size = min_cluster_size
		self.min_samples = min_samples
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.cluster_method = cluster_method
		self.model = OPTICSClustering(predecessor_correction = self.predecessor_correction,
			max_eps = self.max_eps,
			cluster_method = self.cluster_method,
			eps = self.eps,
			leaf_size = self.leaf_size,
			metric_params = self.metric_params,
			p = self.p,
			xi = self.xi,
			metric = self.metric,
			min_cluster_size = self.min_cluster_size,
			n_jobs = self.n_jobs,
			min_samples = self.min_samples,
			algorithm = self.algorithm)

