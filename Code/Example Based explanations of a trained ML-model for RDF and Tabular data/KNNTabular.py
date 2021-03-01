from sklearn.neighbors import LocalOutlierFactor
from decimal import Decimal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


class KNNTabular():
    """
    This calss return the k neighbors of a sample from a tabular dataset.

    ...

    Methods
    -------
    str_column_to_int()
        Cahnge all the columns with object data type to integer
    outliers()
        Detect the outliers in the dataset
    get_neighbors()
        Return the k neighbors of the sample
    """

    # Convert string column to integer
    def str_column_to_int(self, data):
        encs = dict()
        for c in data.columns:
            if data[c].dtype == "object":
                encs[c] = LabelEncoder()
                data[c] = encs[c].fit_transform(data[c])
        return data

    def str_sample_to_int(self, sample):
        for c in sample:
            if type(c) == "object":
                c = LabelEncoder()

    # identify outliers in the training data
    def outliers(self, data):
        iForest=IsolationForest(n_estimators=100, max_samples='auto',
                      contamination=float(0.01), max_features=1.0)
        iForest.fit(data)
        # data['scores'] = iForest.decision_function(data)
        data['anomaly'] = iForest.predict(data)
        anomaly=data.loc[data['anomaly'] == -1]
        anomaly_index=list(anomaly.index)
        data = data.drop(anomaly_index)
        data = data.drop(columns=['anomaly'])
        return data


    # Calculate distances (loos function)
    @staticmethod
    def p_root(value, root):
        root_value = 1 / float(root)
        return round(Decimal(value) **
                     Decimal(root_value), 3)
    @staticmethod
    def minkowski_distance(x, y, p_value):
        return (KNNTabular.p_root(sum(pow(abs(a-b), p_value)
                           for a, b in zip(x, y)), p_value))

    # Locate the most similar neighbors
    def get_neighbors(self, data, k, sample):
        data = data.values
        distances = list()
        for row in data:
            dist = KNNTabular.minkowski_distance(row, sample, 3)
            distances.append((row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors
