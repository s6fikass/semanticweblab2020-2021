
from sklearn.ensemble import GradientBoostingRegressor as GBR
from MLalgorithms._Regression import Regression


class GradientBoostingRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None, monitor=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			monitor=monitor,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
		self.max_depth = max_depth
		self.tol = tol
		self.max_leaf_nodes = max_leaf_nodes
		self.n_iter_no_change = n_iter_no_change
		self.min_impurity_decrease = min_impurity_decrease
		self.alpha = alpha
		self.ccp_alpha = ccp_alpha
		self.n_estimators = n_estimators
		self.criterion = criterion
		self.subsample = subsample
		self.min_samples_leaf = min_samples_leaf
		self.loss = loss
		self.warm_start = warm_start
		self.learning_rate = learning_rate
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.min_impurity_split = min_impurity_split
		self.min_samples_split = min_samples_split
		self.random_state = random_state
		self.max_features = max_features
		self.validation_fraction = validation_fraction
		self.verbose = verbose
		self.model = GBR(learning_rate = self.learning_rate,
			min_impurity_decrease = self.min_impurity_decrease,
			ccp_alpha = self.ccp_alpha,
			subsample = self.subsample,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			validation_fraction = self.validation_fraction,
			alpha = self.alpha,
			max_leaf_nodes = self.max_leaf_nodes,
			max_depth = self.max_depth,
			min_samples_split = self.min_samples_split,
			n_iter_no_change = self.n_iter_no_change,
			n_estimators = self.n_estimators,
			loss = self.loss,
			warm_start = self.warm_start,
			tol = self.tol,
			verbose = self.verbose,
			min_impurity_split = self.min_impurity_split,
			min_samples_leaf = self.min_samples_leaf,
			max_features = self.max_features,
			criterion = self.criterion,
			random_state = self.random_state)

