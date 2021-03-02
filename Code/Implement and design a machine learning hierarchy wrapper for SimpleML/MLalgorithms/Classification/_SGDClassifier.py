
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			intercept_init=intercept_init,
			coef_init=coef_init,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.penalty = penalty
		self.fit_intercept = fit_intercept
		self.n_iter_no_change = n_iter_no_change
		self.average = average
		self.class_weight = class_weight
		self.alpha = alpha
		self.learning_rate = learning_rate
		self.tol = tol
		self.max_iter = max_iter
		self.early_stopping = early_stopping
		self.warm_start = warm_start
		self.l1_ratio = l1_ratio
		self.epsilon = epsilon
		self.shuffle = shuffle
		self.loss = loss
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.validation_fraction = validation_fraction
		self.power_t = power_t
		self.verbose = verbose
		self.eta0 = eta0
		self.model = SGDC(learning_rate = self.learning_rate,
			validation_fraction = self.validation_fraction,
			alpha = self.alpha,
			penalty = self.penalty,
			epsilon = self.epsilon,
			shuffle = self.shuffle,
			n_iter_no_change = self.n_iter_no_change,
			class_weight = self.class_weight,
			power_t = self.power_t,
			early_stopping = self.early_stopping,
			loss = self.loss,
			average = self.average,
			warm_start = self.warm_start,
			n_jobs = self.n_jobs,
			tol = self.tol,
			verbose = self.verbose,
			max_iter = self.max_iter,
			eta0 = self.eta0,
			fit_intercept = self.fit_intercept,
			l1_ratio = self.l1_ratio,
			random_state = self.random_state)

