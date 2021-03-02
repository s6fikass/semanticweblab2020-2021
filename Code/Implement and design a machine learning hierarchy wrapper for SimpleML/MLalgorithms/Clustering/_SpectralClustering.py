
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None):
		self.assign_labels = assign_labels
		self.random_state = random_state
		self.n_init = n_init
		self.n_clusters = n_clusters
		self.coef0 = coef0
		self.n_jobs = n_jobs
		self.eigen_solver = eigen_solver
		self.affinity = affinity
		self.degree = degree
		self.n_neighbors = n_neighbors
		self.eigen_tol = eigen_tol
		self.gamma = gamma
		self.kernel_params = kernel_params
		self.n_components = n_components
		self.model = SC(coef0 = self.coef0,
			eigen_solver = self.eigen_solver,
			n_components = self.n_components,
			gamma = self.gamma,
			eigen_tol = self.eigen_tol,
			affinity = self.affinity,
			assign_labels = self.assign_labels,
			n_init = self.n_init,
			n_jobs = self.n_jobs,
			degree = self.degree,
			kernel_params = self.kernel_params,
			n_clusters = self.n_clusters,
			n_neighbors = self.n_neighbors,
			random_state = self.random_state)

