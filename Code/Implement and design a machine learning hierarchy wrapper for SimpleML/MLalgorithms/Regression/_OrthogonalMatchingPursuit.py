
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from MLalgorithms._Regression import Regression


class OrthogonalMatchingPursuit(Regression):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=False, precompute='auto'):
		self.n_nonzero_coefs = n_nonzero_coefs
		self.precompute = precompute
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.normalize = normalize
		self.model = OMP(fit_intercept = self.fit_intercept,
			n_nonzero_coefs = self.n_nonzero_coefs,
			precompute = self.precompute,
			normalize = self.normalize,
			tol = self.tol)

