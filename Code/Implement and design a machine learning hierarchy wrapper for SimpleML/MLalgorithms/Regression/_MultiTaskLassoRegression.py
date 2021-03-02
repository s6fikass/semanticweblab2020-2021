
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.selection = selection
		self.random_state = random_state
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.warm_start = warm_start
		self.copy_X = copy_X
		self.max_iter = max_iter
		self.alpha = alpha
		self.normalize = normalize
		self.model = MLTR(max_iter = self.max_iter,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			selection = self.selection,
			normalize = self.normalize,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			tol = self.tol,
			random_state = self.random_state)

