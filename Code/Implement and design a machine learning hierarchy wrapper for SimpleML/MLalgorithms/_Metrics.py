
from MLalgorithms._MLalgorithms import MLalgorithms


class Metrics(MLalgorithms):
	
	def __init__(self, y_true, labels=None, sample_weight=None):
		self.sample_weight = sample_weight
		self.y_true = y_true
		self.labels = labels

