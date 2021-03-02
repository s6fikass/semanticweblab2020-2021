
from MLalgorithms._MLalgorithms import MLalgorithms


class Clustering(MLalgorithms):
	
	def predict(self, X, sample_weight=None):
		return self.model.predict(sample_weight=sample_weight,
			X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

