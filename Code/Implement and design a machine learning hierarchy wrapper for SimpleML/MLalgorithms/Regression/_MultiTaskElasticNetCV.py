
from sklearn.linear_model import MultiTaskElasticNetCV as MTENCV
from MLalgorithms._Regression import Regression


class MultiTaskElasticNetCV(Regression):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=None, alphas=None, fit_intercept=True, normalize=False, max_iter=100, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
		self.tol = tol
		self.selection = selection
		self.alphas = alphas
		self.eps = eps
		self.random_state = random_state
		self.l1_ratio = l1_ratio
		self.max_iter = max_iter
		self.cv = cv
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.copy_X = copy_X
		self.n_alphas = n_alphas
		self.verbose = verbose
		self.normalize = normalize
		self.model = MTENCV(max_iter = self.max_iter,
			cv = self.cv,
			eps = self.eps,
			fit_intercept = self.fit_intercept,
			n_jobs = self.n_jobs,
			selection = self.selection,
			normalize = self.normalize,
			alphas = self.alphas,
			verbose = self.verbose,
			l1_ratio = self.l1_ratio,
			n_alphas = self.n_alphas,
			copy_X = self.copy_X,
			tol = self.tol,
			random_state = self.random_state)

