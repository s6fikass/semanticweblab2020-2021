
from sklearn.svm import SVR as SVRRegression
from MLalgorithms._Regression import Regression


class SVR(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
		self.C = C
		self.epsilon = epsilon
		self.gamma = gamma
		self.max_iter = max_iter
		self.shrinking = shrinking
		self.degree = degree
		self.tol = tol
		self.cache_size = cache_size
		self.verbose = verbose
		self.kernel = kernel
		self.coef0 = coef0
		self.model = SVRRegression(coef0 = self.coef0,
			max_iter = self.max_iter,
			gamma = self.gamma,
			cache_size = self.cache_size,
			shrinking = self.shrinking,
			epsilon = self.epsilon,
			C = self.C,
			degree = self.degree,
			tol = self.tol,
			kernel = self.kernel,
			verbose = self.verbose)

