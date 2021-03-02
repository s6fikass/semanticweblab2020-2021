
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.class_weight = class_weight
		self.solver = solver
		self.random_state = random_state
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.copy_x = copy_x
		self.max_iter = max_iter
		self.alpha = alpha
		self.normalize = normalize
		self.model = RC(max_iter = self.max_iter,
			copy_x = self.copy_x,
			solver = self.solver,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			tol = self.tol,
			class_weight = self.class_weight,
			random_state = self.random_state)

