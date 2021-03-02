
from sklearn.tree import DecisionTreeClassifier as DTC
from MLalgorithms._Classification import Classification


class DecisionTreeClassifier(Classification):
	
	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y,
			check_input=check_input)

	def predict(self, X, check_input=True):
		return self.model.predict(X=X,
			check_input=check_input)

	def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0):
		self.class_weight = class_weight
		self.max_leaf_nodes = max_leaf_nodes
		self.min_samples_split = min_samples_split
		self.random_state = random_state
		self.min_samples_leaf = min_samples_leaf
		self.ccp_alpha = ccp_alpha
		self.min_impurity_decrease = min_impurity_decrease
		self.max_features = max_features
		self.splitter = splitter
		self.max_depth = max_depth
		self.criterion = criterion
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.min_impurity_split = min_impurity_split
		self.model = DTC(ccp_alpha = self.ccp_alpha,
			min_impurity_decrease = self.min_impurity_decrease,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			min_impurity_split = self.min_impurity_split,
			splitter = self.splitter,
			min_samples_split = self.min_samples_split,
			max_leaf_nodes = self.max_leaf_nodes,
			max_depth = self.max_depth,
			min_samples_leaf = self.min_samples_leaf,
			max_features = self.max_features,
			criterion = self.criterion,
			class_weight = self.class_weight,
			random_state = self.random_state)

