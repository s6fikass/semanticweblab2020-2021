
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.zero_division = zero_division
		self.output_dict = output_dict
		self.y_pred = y_pred
		self.target_names = target_names
		self.digits = digits
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = CR(sample_weight = self.sample_weight,
			zero_division = self.zero_division,
			digits = self.digits,
			y_true = self.y_true,
			labels = self.labels,
			output_dict = self.output_dict,
			y_pred = self.y_pred,
			target_names = self.target_names)

