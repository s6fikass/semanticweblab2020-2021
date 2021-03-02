
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.class_weight = class_weight
		self.tol = tol
		self.dual = dual
		self.multi_class = multi_class
		self.Cs = Cs
		self.random_state = random_state
		self.penalty = penalty
		self.max_iter = max_iter
		self.cv = cv
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.refit = refit
		self.l1_ratios = l1_ratios
		self.solver = solver
		self.intercept_scaling = intercept_scaling
		self.verbose = verbose
		self.scoring = scoring
		self.model = LRCV(dual = self.dual,
			intercept_scaling = self.intercept_scaling,
			max_iter = self.max_iter,
			cv = self.cv,
			solver = self.solver,
			fit_intercept = self.fit_intercept,
			l1_ratios = self.l1_ratios,
			refit = self.refit,
			tol = self.tol,
			n_jobs = self.n_jobs,
			penalty = self.penalty,
			verbose = self.verbose,
			Cs = self.Cs,
			scoring = self.scoring,
			multi_class = self.multi_class,
			class_weight = self.class_weight,
			random_state = self.random_state)

