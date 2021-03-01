import ExampleBasedExplanations as ebe


test = ebe.ExampleBasedExplanations()

# knn with tabular
newData = test.KNNTabularOutlier(fileName)
neighbors = test.KNNTabularNeighbors(fileName, 5, [5.4,3.9,1.7,0.4])

# knn with protodash

test.protoDashRDF("WikiMovieActors.rdf", "4", "http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#jimmy_sangster,http://www.w3.org/1999/02/22-rdf-syntax-ns#type,http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#Film_writer")
test.protoDashImage("0", "3")

# knn with rdf

file_name='WikiMovie.rdf'
input_node='http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#naomi_watts'
result0 = test.knnRDF(file_name, input_node, 4)
result = test.knnRDFCluster(file_name)
