
from sklearn.linear_model import LogisticRegression as LR
from MLalgorithms._Regression import Regression


class LogisticRegression(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
		self.C = C
		self.class_weight = class_weight
		self.tol = tol
		self.dual = dual
		self.multi_class = multi_class
		self.random_state = random_state
		self.l1_ratio = l1_ratio
		self.penalty = penalty
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.warm_start = warm_start
		self.solver = solver
		self.intercept_scaling = intercept_scaling
		self.verbose = verbose
		self.model = LR(dual = self.dual,
			intercept_scaling = self.intercept_scaling,
			max_iter = self.max_iter,
			solver = self.solver,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			C = self.C,
			n_jobs = self.n_jobs,
			penalty = self.penalty,
			l1_ratio = self.l1_ratio,
			verbose = self.verbose,
			warm_start = self.warm_start,
			multi_class = self.multi_class,
			class_weight = self.class_weight,
			random_state = self.random_state)

