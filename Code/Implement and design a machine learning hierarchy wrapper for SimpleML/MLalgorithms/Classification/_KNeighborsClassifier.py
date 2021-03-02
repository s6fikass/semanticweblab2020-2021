
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.p = p
		self.weights = weights
		self.metric = metric
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.n_neighbors = n_neighbors
		self.model = KNC(weights = self.weights,
			metric_params = self.metric_params,
			p = self.p,
			metric = self.metric,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			n_neighbors = self.n_neighbors,
			algorithm = self.algorithm)

