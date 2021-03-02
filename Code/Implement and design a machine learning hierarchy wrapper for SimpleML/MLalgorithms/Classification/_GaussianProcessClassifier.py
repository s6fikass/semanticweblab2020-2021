
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from MLalgorithms._Classification import Classification


class GaussianProcessClassifier(Classification):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
		self.kernel = kernel
		self.n_restarts_optimizer = n_restarts_optimizer
		self.random_state = random_state
		self.multi_class = multi_class
		self.n_jobs = n_jobs
		self.max_iter_predict = max_iter_predict
		self.warm_start = warm_start
		self.copy_X_train = copy_X_train
		self.optimizer = optimizer
		self.model = GPC(copy_X_train = self.copy_X_train,
			n_jobs = self.n_jobs,
			max_iter_predict = self.max_iter_predict,
			n_restarts_optimizer = self.n_restarts_optimizer,
			optimizer = self.optimizer,
			warm_start = self.warm_start,
			multi_class = self.multi_class,
			kernel = self.kernel,
			random_state = self.random_state)

