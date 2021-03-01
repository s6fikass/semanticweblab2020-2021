from KNNTabular import KNNTabular
from KNN_RDF import KnnAndClusters
from ProtoDashRDFImage import ProtoDash
from pandas import read_csv


class ExampleBasedExplanations:

    """
    This class helps the user to understand and explain the behaviour of a machine learning model,
    by returning instances of the dataset that affect the prediction of the ML
    ...
    Attributes
    ----------
    dataset : str
        Path and name of the dataset we want to read
    k : int, default k = 3
        The number neighburs or prototyps
    sample: list or str
        The data that we want to test
    dataTyple: str
        The type of the dataset you want to test with ('tabular', 'rdf', 'image')
    algorithm:
        The algorithm you want to use ('knn','protodash'), if the data type is 'image' this should be None
    outlier: boolean, default False
        Enable or disable outlier detection on the dataset before getting the neighbours. Applied on tabular data

    Methods
    -------
    """

    def KNNTabularOutlier(self, file_name):
        data = read_csv(file_name, header=None)
        knn = KNNTabular()
        data = knn.str_column_to_int(data)
        outliers = knn.outliers(data)
        return outliers

    def KNNTabularNeighbors (self, data, k, sample):
        if isinstance(data, str) is True:
            data = read_csv(data, header=None)
        knn = KNNTabular()
        data = knn.str_column_to_int(data)
        neighbors = knn.get_neighbors(data, k, sample)
        return neighbors

         
    def knnRDF(self, file_name, input_node,number_of_neighbors):
        knnrdf = KnnAndClusters()
        results, nx_graph = knnrdf.K_Nearest_neighbours(file_name, input_node,number_of_neighbors)
        return results

    def knnRDFCluster(self, file_name):
        knnrdf = KnnAndClusters()
        result2 = knnrdf.create_cluster(file_name)
        return result2


    def protoDashRDF(self, file_name, num_proto, sample_triple):
        ProtoDash.ProtoDashOnRDF(file_name, num_proto, sample_triple)

    def protoDashImage(self, digit, num_proto):
        ProtoDash.ProtoDashOnImage(digit, num_proto)











