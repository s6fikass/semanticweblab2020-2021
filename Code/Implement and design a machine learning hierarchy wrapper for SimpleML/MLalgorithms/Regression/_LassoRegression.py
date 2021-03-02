
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.selection = selection
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.warm_start = warm_start
		self.precompute = precompute
		self.copy_X = copy_X
		self.alpha = alpha
		self.positive = positive
		self.normalize = normalize
		self.max_iter = max_iter
		self.model = Lasso(max_iter = self.max_iter,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			selection = self.selection,
			positive = self.positive,
			normalize = self.normalize,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			tol = self.tol,
			random_state = self.random_state)

