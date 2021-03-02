
from sklearn.ensemble import RandomForestRegressor as RFR
from MLalgorithms._Regression import Regression


class RandomForestRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
		self.max_samples = max_samples
		self.max_leaf_nodes = max_leaf_nodes
		self.max_features = max_features
		self.bootstrap = bootstrap
		self.min_samples_split = min_samples_split
		self.random_state = random_state
		self.min_samples_leaf = min_samples_leaf
		self.ccp_alpha = ccp_alpha
		self.min_impurity_decrease = min_impurity_decrease
		self.criterion = criterion
		self.n_jobs = n_jobs
		self.max_depth = max_depth
		self.warm_start = warm_start
		self.oob_score = oob_score
		self.verbose = verbose
		self.n_estimators = n_estimators
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.min_impurity_split = min_impurity_split
		self.model = RFR(ccp_alpha = self.ccp_alpha,
			bootstrap = self.bootstrap,
			min_impurity_decrease = self.min_impurity_decrease,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			min_impurity_split = self.min_impurity_split,
			max_depth = self.max_depth,
			min_samples_split = self.min_samples_split,
			max_leaf_nodes = self.max_leaf_nodes,
			n_estimators = self.n_estimators,
			min_samples_leaf = self.min_samples_leaf,
			max_features = self.max_features,
			oob_score = self.oob_score,
			max_samples = self.max_samples,
			verbose = self.verbose,
			warm_start = self.warm_start,
			n_jobs = self.n_jobs,
			criterion = self.criterion,
			random_state = self.random_state)

