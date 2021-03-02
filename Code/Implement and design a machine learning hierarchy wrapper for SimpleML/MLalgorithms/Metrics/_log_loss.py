
from sklearn.metrics import log_loss as LL
from MLalgorithms._Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, y_true, y_pred, eps=1e-15, labels=None, normalize=True, sample_weight=None):
		self.y_pred = y_pred
		self.eps = eps
		self.normalize = normalize
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = LL(sample_weight = self.sample_weight,
			eps = self.eps,
			labels = self.labels,
			y_true = self.y_true,
			normalize = self.normalize,
			y_pred = self.y_pred)

