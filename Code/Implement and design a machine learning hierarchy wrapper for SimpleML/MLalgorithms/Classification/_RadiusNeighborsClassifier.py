
from sklearn.neighbors import RadiusNeighborsClassifier as RNC
from MLalgorithms._Classification import Classification


class RadiusNeighborsClassifier(Classification):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X, weights='uniform', p=2, metric='minkowski'):
		return self.model.predict(p=p,
			X=X,
			metric=metric,
			weights=weights)

	def __init__(self, radius=1.0, algorithm='auto', leaf_size=30, outlier_label=None, metric_params=None, n_jobs=None):
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.radius = radius
		self.metric_params = metric_params
		self.outlier_label = outlier_label
		self.leaf_size = leaf_size
		self.model = RNC(radius = self.radius,
			outlier_label = self.outlier_label,
			metric_params = self.metric_params,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			algorithm = self.algorithm)

