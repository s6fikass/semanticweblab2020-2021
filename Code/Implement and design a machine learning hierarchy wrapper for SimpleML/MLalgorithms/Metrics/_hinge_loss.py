
from sklearn.metrics import hinge_loss as HL
from MLalgorithms._Metrics import Metrics


class hinge_loss(Metrics):
	
	def __init__(self, y_true, pred_decision, labels=None, sample_weight=None):
		self.pred_decision = pred_decision
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = HL(pred_decision = self.pred_decision,
			sample_weight = self.sample_weight,
			labels = self.labels,
			y_true = self.y_true)

