
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.solver = solver
		self.random_state = random_state
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.copy_X = copy_X
		self.max_iter = max_iter
		self.alpha = alpha
		self.normalize = normalize
		self.model = Ridge(max_iter = self.max_iter,
			solver = self.solver,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			copy_X = self.copy_X,
			tol = self.tol,
			random_state = self.random_state)

