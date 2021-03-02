
from sklearn.svm import LinearSVR as LSVR
from MLalgorithms._Regression import Regression


class LinearSVR(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=None, max_iter=1000):
		self.loss = loss
		self.tol = tol
		self.epsilon = epsilon
		self.random_state = random_state
		self.dual = dual
		self.fit_intercept = fit_intercept
		self.intercept_scaling = intercept_scaling
		self.verbose = verbose
		self.max_iter = max_iter
		self.model = LSVR(dual = self.dual,
			intercept_scaling = self.intercept_scaling,
			max_iter = self.max_iter,
			loss = self.loss,
			fit_intercept = self.fit_intercept,
			epsilon = self.epsilon,
			verbose = self.verbose,
			tol = self.tol,
			random_state = self.random_state)

