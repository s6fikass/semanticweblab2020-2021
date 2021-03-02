
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.random_state = random_state
		self.damping = damping
		self.affinity = affinity
		self.max_iter = max_iter
		self.copy = copy
		self.convergence_iter = convergence_iter
		self.verbose = verbose
		self.preference = preference
		self.model = APClustering(damping = self.damping,
			max_iter = self.max_iter,
			affinity = self.affinity,
			convergence_iter = self.convergence_iter,
			verbose = self.verbose,
			copy = self.copy,
			random_state = self.random_state,
			preference = self.preference)

