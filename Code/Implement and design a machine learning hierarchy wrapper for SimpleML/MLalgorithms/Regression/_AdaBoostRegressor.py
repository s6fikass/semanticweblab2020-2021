
from sklearn.ensemble import AdaBoostRegressor as ABR
from MLalgorithms._Regression import Regression


class AdaBoostRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
		self.base_estimator = base_estimator
		self.learning_rate = learning_rate
		self.random_state = random_state
		self.n_estimators = n_estimators
		self.model = ABR(n_estimators = self.n_estimators,
			learning_rate = self.learning_rate,
			base_estimator = self.base_estimator,
			random_state = self.random_state)

