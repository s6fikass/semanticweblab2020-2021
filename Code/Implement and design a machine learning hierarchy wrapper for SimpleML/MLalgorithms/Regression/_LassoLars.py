
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def fit(self, X, y, Xy=None):
		return self.model.fit(Xy=Xy,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.fit_path = fit_path
		self.random_state = random_state
		self.eps = eps
		self.alpha = alpha
		self.precompute = precompute
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.copy_X = copy_X
		self.verbose = verbose
		self.positive = positive
		self.jitter = jitter
		self.normalize = normalize
		self.model = LLR(fit_path = self.fit_path,
			max_iter = self.max_iter,
			alpha = self.alpha,
			eps = self.eps,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			jitter = self.jitter,
			positive = self.positive,
			normalize = self.normalize,
			verbose = self.verbose,
			copy_X = self.copy_X,
			random_state = self.random_state)

