
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
		self.C = C
		self.nu = nu
		self.gamma = gamma
		self.max_iter = max_iter
		self.shrinking = shrinking
		self.degree = degree
		self.tol = tol
		self.cache_size = cache_size
		self.verbose = verbose
		self.kernel = kernel
		self.coef0 = coef0
		self.model = NSVR(coef0 = self.coef0,
			max_iter = self.max_iter,
			gamma = self.gamma,
			cache_size = self.cache_size,
			shrinking = self.shrinking,
			C = self.C,
			degree = self.degree,
			nu = self.nu,
			tol = self.tol,
			kernel = self.kernel,
			verbose = self.verbose)

