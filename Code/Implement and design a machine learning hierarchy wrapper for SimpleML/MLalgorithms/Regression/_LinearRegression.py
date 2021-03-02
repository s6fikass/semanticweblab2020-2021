
from sklearn.linear_model import LinearRegression as LR
from MLalgorithms._Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, positive=False):
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.copy_X = copy_X
		self.positive = positive
		self.normalize = normalize
		self.model = LR(fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			copy_X = self.copy_X,
			n_jobs = self.n_jobs,
			positive = self.positive)

