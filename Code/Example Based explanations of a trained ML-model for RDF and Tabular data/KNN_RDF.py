import numpy as np
import pandas as pd
import rdflib
from rdflib.extras.external_graph_libs import *
from rdflib import Graph, URIRef, Literal
from rdflib.util import guess_format as gf
from sklearn.cluster import KMeans as kmeans
import networkx as nx
from collections import Counter, defaultdict
import operator
import itertools
import pprint
class KnnAndClusters:

    def K_Nearest_neighbours(self,Filepath,input_node,number_of_neighbors):
        """
        This method creates a graph representation of rdf dataset and 
        calculates nearest neighbors to the given input node by using Jaccard coefficient as metrics.

        Args:
            Filepath(String):It specifies path where the rdf file is located. Example:"F:/semanticLab/dataset/WikiMovie.rdf"
            input_node(String):It is the input node for which the nearest neighbors are calculated.
                                Example:'http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#naomi_watts'

        Returns :
                result_dict(dict): A dictionaries containing nearest neighbors as the keys along with corresponding similarity as values.
                nx_graph: A graph created from the rdf data given.

        """
        graph = Graph()
        graph.parse(Filepath, format="xml")
        nx_graph = rdflib_to_networkx_graph(graph)
        output_dict={}
        neighbour_dict = {}
        for node2 in nx_graph.nodes():
            if(rdflib.term.URIRef(input_node)==node2):
                continue
            preds = nx.jaccard_coefficient(nx_graph, [(rdflib.term.URIRef(input_node), node2)])
            for u,v,p in preds:
                neighbour_dict[v]= p
        output_dict[u] = neighbour_dict
        output_dict[u]=self.sorting_dict(output_dict,input_node)
        result_dict = dict(itertools.islice(output_dict.get(rdflib.term.URIRef(input_node)).items(),number_of_neighbors))
        return result_dict, nx_graph

    def sorting_dict(self,output_dict,input_node):
        """
        It sorts the dictionary obtained from the K_nearest neighbours
        Args:
            output_dict(dict): dictionary containing the similarity value and the other node with which the input node is compared.
            input_node(String): A node which is to be compared with other nodes to get the simiarity value and sort the doctionary based on that.

        Returns:
                sortedDic(dict): It is a dictionary with similarity values sorted 
        """
        sortedDic={}
        sortedDic=dict(sorted(output_dict[rdflib.term.URIRef(input_node)].items(), key=operator.itemgetter(1), reverse=True))
        return sortedDic


    def create_cluster(self,Filepath):
        """
        Thsi function is used to create clusters 
        Args:
            nx_graph:It is a networkX generated graph for the given rdf dataset
        Returns:
                nodeSet(set): it contains unique set of nodes involved in a particular cluster
        """
        graph = Graph()
        graph.parse(Filepath, format="xml")
        nx_graph = rdflib_to_networkx_graph(graph)
        similarity_array=[]
        similarity_array_nodes = []
        for node1 in nx_graph.nodes():
            for node2 in nx_graph.nodes():
                if(node1==node2):
                    continue
                preds = nx.jaccard_coefficient(nx_graph, [(node1, node2)])
            #similarity_array_nodes.append(preds)
                pred_list = []
                for u,v,p in preds:
                    similarity_array.append(p)
                    pred_list.append(u)
                    pred_list.append(v)
                    pred_list.append(p)
                similarity_array_nodes.append(pred_list)
        data=pd.DataFrame(similarity_array_nodes,columns=['node1','node2','similarity_value'])
        K_meaned_values=kmeans(5,init='k-means++',n_init=200).fit(data['similarity_value'].values.reshape(-1,1))
        print(Counter(K_meaned_values.labels_))
        cluster_index_value=int(input("Please enter the number of the cluster you are interested:"))
        nodeSet=self.print_cluster_elements(data,cluster_index_value,K_meaned_values)
        return  nodeSet

    def print_cluster_elements(self,data,cluster_index_value,K_meaned_values):
        """
        Prints the cluster formed.
        Args:
            data(pandas Dataframe): Its a dataframe contains details of the nodes and their corresponding similarity values.
            cluster_index_value(int): cluster index the user is interested in.
            K_meaned_values:Output of K means clustering algorithm.
        Returns:
                nodeSet(set): Set containing all the  unique set of nodes involved in a particular cluster
        """
        clusters_indices = defaultdict(list)
        data_list=[]
        
        for index, c  in enumerate(K_meaned_values.labels_):
            clusters_indices[c].append(index)
        
        for i in clusters_indices[cluster_index_value]:
            data_list.append(data.iloc[i].values)
        nodeSet=set()
        for i in data_list:
            nodeSet.add(i[0])
            nodeSet.add(i[1])
        #print(nodeSet)
        #print(len(nodeSet))
        return nodeSet
    

   
"""
if __name__ == "__main__":
        path="F:/semanticLab/Shobith/WikiMovie.rdf"
        input_node='http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#naomi_watts'
        result, nx_graph =KnnAndClusters().K_Nearest_neighbours(path,input_node)
        pprint.pprint('KNN Result:')
        print(result)
        cluster_data = KnnAndClusters().create_cluster(nx_graph)
        print("-----------------------------------------------------------------------------------")
        print('cluster data:')

        pprint.pprint(cluster_data)
"""