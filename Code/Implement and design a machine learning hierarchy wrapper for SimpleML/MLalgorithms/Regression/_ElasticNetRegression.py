
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y,
			check_input=check_input)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.selection = selection
		self.random_state = random_state
		self.l1_ratio = l1_ratio
		self.fit_intercept = fit_intercept
		self.warm_start = warm_start
		self.precompute = precompute
		self.copy_X = copy_X
		self.alpha = alpha
		self.positive = positive
		self.normalize = normalize
		self.max_iter = max_iter
		self.model = ElasticNet(max_iter = self.max_iter,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			selection = self.selection,
			positive = self.positive,
			normalize = self.normalize,
			l1_ratio = self.l1_ratio,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			tol = self.tol,
			random_state = self.random_state)

