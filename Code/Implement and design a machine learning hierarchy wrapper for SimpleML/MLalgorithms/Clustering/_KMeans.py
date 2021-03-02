
from sklearn.cluster import KMeans as KMeansClustering
from MLalgorithms._Clustering import Clustering


class KMeans(Clustering):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X, sample_weight=None):
		return self.model.predict(sample_weight=sample_weight,
			X=X)

	def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
		self.tol = tol
		self.max_iter = max_iter
		self.precompute_distances = precompute_distances
		self.random_state = random_state
		self.n_init = n_init
		self.n_clusters = n_clusters
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.copy_x = copy_x
		self.verbose = verbose
		self.model = KMeansClustering(max_iter = self.max_iter,
			copy_x = self.copy_x,
			n_jobs = self.n_jobs,
			n_init = self.n_init,
			n_clusters = self.n_clusters,
			verbose = self.verbose,
			tol = self.tol,
			precompute_distances = self.precompute_distances,
			random_state = self.random_state,
			algorithm = self.algorithm)

