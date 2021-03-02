
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.C = C
		self.class_weight = class_weight
		self.break_ties = break_ties
		self.random_state = random_state
		self.gamma = gamma
		self.probability = probability
		self.max_iter = max_iter
		self.shrinking = shrinking
		self.degree = degree
		self.tol = tol
		self.cache_size = cache_size
		self.verbose = verbose
		self.decision_function_shape = decision_function_shape
		self.kernel = kernel
		self.coef0 = coef0
		self.model = SVCClassification(coef0 = self.coef0,
			decision_function_shape = self.decision_function_shape,
			gamma = self.gamma,
			max_iter = self.max_iter,
			cache_size = self.cache_size,
			shrinking = self.shrinking,
			C = self.C,
			degree = self.degree,
			break_ties = self.break_ties,
			verbose = self.verbose,
			probability = self.probability,
			tol = self.tol,
			kernel = self.kernel,
			class_weight = self.class_weight,
			random_state = self.random_state)

