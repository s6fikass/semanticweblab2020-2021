# relevant imports
import numpy as np
from rdflib import Graph, URIRef
from pyspark.mllib.linalg import DenseVector
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, LongType
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import mnist
from textwrap import wrap
import os.path

import warnings
warnings.filterwarnings("ignore")

from pyspark import SparkContext
from pyspark.sql import SparkSession


class ProtoDash():

    def create_spark_session():
        global sc
        sc = SparkContext.getOrCreate()
        sc.setLogLevel("ERROR")
        return SparkSession(sc)
    
    def create_vec_rdd(X, part=4):
        """
        Function returning a DenseVector RDD from a dataset X.    
        Args:
        -X: a dataset with rows corresponding to observations and columns corresponding to features.
        -part: n of partitions.    
        Returns: the RDD for X
        """
        # creating a Spark session
        spark_session = ProtoDash.create_spark_session()
        
        X_rdd = (sc.parallelize(X, part)
                 .map(lambda x: DenseVector(x))
                 .zipWithIndex())
        return X_rdd
    
    def mean_inner_product(inp,sigma,n):
        """
        Function computing the gaussian kernel inner product of a vector in Y vs.
        a vector in X, divided by n the total number of observations in X.
        """
        index_1 = inp[0][1]
        inner_product = float(np.exp(inp[0][0].squared_distance(inp[1][0])/(-2*sigma**2))/n)
        return (index_1, inner_product)
    
    
    def inner_product(inp,sigma):
        """
        Function computing the gaussian kernel inner product of a vector vs.
        another.
        """
        index_1 = inp[0][1]
        index_2 = inp[1][1]
        inner_product = float(np.exp(inp[0][0].squared_distance(inp[1][0])/(-2*sigma**2)))
        return (index_1, [(index_2, inner_product)])
    
    
    def weighted_sum(inp,w_arr):
        """
        compute the weighted sum of matrix values for a set of indices and weights.
        Note it is fine using a list comprehension here since the number of prototypes m << |X^{(1)}|.
        """
        return float(np.sum(np.array([x[1] for x in inp])*w_arr))
    
    
    def udf_weighted_sum(w_arr):
        """
        UDF instance of the weighted_sum function.
        """
        return F.udf(lambda l: ProtoDash.weighted_sum(l,w_arr))
    
    
    def merge_lists(x, y):
        """
        merge lists.
        """
        return sorted(x+y, key=lambda tup: tup[0])
    
    # Create UDF corresponding to merge_lists function.
    DType = ArrayType(StructType([StructField("_1", LongType()), StructField("_2", FloatType())]))
    udf_merge_lists = F.udf(merge_lists, DType)


    def optimize(K, u, opt_w0, init_val, max_w=10000):
        """
        Function solving quadratic optimization problem.        
        Args:
        -K: inner product matrix
        -u: mean inner product of each prototype
        -opt_w0: initial weights vector
        -init_val: starting run
        -max_w: an upper bound on weight value    
        Returns:
        -weights and the objective value
        """
        dim = u.shape[0]
        low_b = np.zeros((dim, 1))
        upper_b = max_w * np.ones((dim, 1))
        x_init = np.append( opt_w0, init_val / K[dim-1, dim-1] )
        G = np.vstack((np.identity(dim), -1*np.identity(dim)))
        h = np.vstack((upper_b, -1*low_b))
    
        # solve constrained quadratic problem
        soltn = solve_qp(K, -u, G, h, A = None, b = None , solver = 'cvxopt' , initvals = x_init )
        
        # calculate the objective function value for optimal solution
        x_sol = soltn.reshape( soltn.shape[0], 1 )
        q = - u.reshape( u.shape[0], 1 )
        obj_value = 1/2 * np.matmul(np.matmul(x_sol.T, K), x_sol) + np.matmul(q.T, x_sol)
        
        return ( soltn, obj_value[0,0] )
    
    
    def ProtoDashAlgoritm(X,Y,m,sigma,partitions=20,verbose=True):
        """
        Implementation of the ProtoDash algorithm
        
        Args:
        -X (RDD of indexed DenseVector rows): Target dataset/ the dataset to be represented.
        -Y (RDD of indexed DenseVector rows): Source dataset/ the dataset to select prototypes from.
        -m (integer): total number of prototypes to select.
        -sigma (strictly positive float): gaussian kernel parameter.
        -partitions (integer): number of RDD partitions to compute inner product RDDs with.
        -verbose (boolean): whether or not to print the cumulative number of prototypes selected at each iteration.
        
        Returns:
        -L (integer list): the set of indices corresponding to selected prototypes.
        -w (float list): the optimal set of weights corresponding to each selected prototype.
        """
        
           
        # get count of observations in X
        n_X = X.count()
        
        # build mu DataFrame
        mu_df = (Y.cartesian(X)
                  .map(lambda x: ProtoDash.mean_inner_product(x,sigma,n_X))
                  .reduceByKey(lambda x,y: x+y)
                  .toDF(["obs","mu"]))
                
        # initialise key variables
        L = np.zeros(m, dtype=int)        # set of prototype indices L
        w = np.zeros(m, dtype=float)      # set of optimal prototype weights
        f_eval = np.zeros(m, dtype=float) # set of the f(w) eval. at prototype selection
        n_L = 0                           # count of prototypes selected so far
            
        # find the index corresponding to the maximum mu value
        max_grad_0 = mu_df.orderBy(F.desc("mu")).limit(1).collect()[0]
            
        # collect values 
        L[n_L] = max_grad_0.obs 
        w[n_L] = max_grad_0.mu 
        f_eval[n_L] = 1/2 * max_grad_0.mu ** 2
        n_L += 1
                
        # select the row of Y corresponding to the first chosen index
        Y_row_j0 = Y.filter(lambda x: x[1]==L[:n_L]).collect()[0]
        
        # take its inner product with all rows of Y to build the starting K dataframe
        K_init_df = (Y.map(lambda x: ProtoDash.inner_product((x,Y_row_j0),sigma))
                      .toDF(["obs","K"]))
        
        # join mu and K dataframes
        join_df = (mu_df.join(K_init_df, "obs")
                        .repartition(partitions))
        
        # cache join_df as it is reused often
        join_df.cache()
            
        # compute the new gradient vector
        grad_df = (join_df.withColumn("K_weighted", ProtoDash.udf_weighted_sum(w[:n_L])(F.col("K")))
                          .withColumn("grad", F.col("mu") - F.col("K_weighted"))
                          .select("obs","grad"))
        
                
        # begin while loop
        while n_L < m:        
                    
            # remove the rows that have an index already included in L
            grad_df = grad_df.filter(~grad_df.obs.isin([int(x) for x in L[:n_L]]))
    
            # find the row that has the maximum value in the filtered gradient vector
            argmax_grad = grad_df.orderBy(F.desc("grad")).limit(1).collect()[0]
    
            # update L
            L[n_L] = argmax_grad.obs                
    
            # select the row of Y corresponding to the chosen index
            Y_row_j = Y.filter(lambda x: x[1]==L[n_L]).collect()[0]
    
            # take its inner product with all rows of Y to build new K
            K_int_df = (Y.map(lambda x: ProtoDash.inner_product((x,Y_row_j),sigma))
                         .toDF(["obs","new_K_col"]))
    
            # add new K col to previous K col
            join_df = (join_df.join(K_int_df, "obs")
                              .withColumn("K_merged", ProtoDash.udf_merge_lists(F.col("K"), F.col("new_K_col")))
                              .select("obs", "mu", "K_merged")
                              .withColumnRenamed("K_merged", "K"))
    
            # cache new joined_df
            join_df.cache()
    
            # increment n_L
            n_L +=1 
            
            # sort L
            L[:n_L] = sorted(L[:n_L])
    
            if verbose is True and n_L % 5==0:
                    print( "Prototypes selected - "+str(n_L) )
    
            # take max gradient val.
            max_grad = argmax_grad.grad
    
            # filter join dataframe for given indices in L
            filt_df = (join_df.filter(join_df.obs.isin([int(x) for x in L[:n_L]]))
                              .orderBy(F.col("obs").asc()))
    
            # take mu vector
            mu_arr = np.array(filt_df.select("mu").collect(), dtype=float)
    
            # take K matrix
            K_mat = np.array(filt_df.rdd
                                     .map(lambda x: [y[1] for y in x[2]])
                                     .collect(), dtype=float )
            
            # find optimal weights for the index set L
            opt_res = ProtoDash.optimize(K_mat, mu_arr, w[:n_L-1], max_grad)
            (w[:n_L], f_eval[n_L-1]) = opt_res[0], -opt_res[1]
    
            # compute gradient vector with new optimal weights
            grad_df = (join_df.withColumn("K_weighted", ProtoDash.udf_weighted_sum(w[:n_L])(F.col("K")))
                              .withColumn("grad", F.col("mu") - F.col("K_weighted"))
                              .select("obs","grad"))        
            
        # tuple of indices and their corresponding weight, sorted by weight in descending order.
        res = sorted([(w[i],L[i]) for i in range(m)], key=lambda tup: -tup[0])
               
        # return tuple of index set L and optimal weight set w, set of f_eval
        return res, f_eval
    
    #######################################################
    #########     RDF IMPLEMENTATION             ###########
    #######################################################

    rdf_dataset = None
    numeric_dataset = None
    subject_index = {}
    predicate_index = {}
    object_index = {}
    
    def infer_index(token, token_indices):
        """
        Enumerate the distinct tokens. If the token is found in the token_indices, then return it,
        else assign the next integer number (after the last assigned index) which is also the size of the token_indices.
        """
        if token in token_indices:
            return token_indices[token]
        else:
            token_index = len(token_indices)
            token_indices[token] = token_index
            return token_index
    
    def convert_rdf_to_ntriples(dataset):
        """
        Loads rdf data and converts into n-triples, treating each triple as a datapoint in the dataset.
        """
        g = Graph()
        g.load(dataset)
        
        rem_object = URIRef("http://www.w3.org/2002/07/owl#NamedIndividual") # deleting the triples that have object value as 'owl#NamedIndividual'
        for s, p, o in g:
            g.remove((s, p, rem_object))
            
        global rdf_dataset
        global numeric_dataset
    
        # create n-triples of strings
        rdf_dataset = [(str(s), str(p), str(o)) for s, p, o in g]
        
    
        # preprocess and create a numeric dataset in order to input to ProtoDash
        numeric_dataset = list(map(lambda e: 
            (ProtoDash.infer_index(e[0], ProtoDash.subject_index),
            ProtoDash.infer_index(e[1], ProtoDash.predicate_index),
            ProtoDash.infer_index(e[2], ProtoDash.object_index)),
            rdf_dataset))
        #print(rdf_dataset)
        #print('************************************')
        #print('Size of dataset:', len(rdf_dataset))
        #print('Subjects cardinality:', len(subject_index))
        #print('Predicates cardinality:', len(predicate_index))
        #print('Objects cardinality:', len(object_index))
        print('************************************')
    
        return numeric_dataset
    
    
    def strip_rdf_prefix(triple):
        """
        Strips the common URL-like prefixes from the RDF data and takes the suffix after '#'.
    
        Example:
    
        Input triple:  ('http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#naomi_watts',
                        'http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#acted_in',
                        'http://www.semanticweb.org/vinu/ontologies/2014/6/untitled-ontology-91#rabbits')
    
        Output: naomi_watts acted_in rabbits
        """
        return ' '.join(tuple(map(lambda e: e[e.find('#') + 1:], triple)))
    
    def get_sample_index(dataset, sample):
    
        # Function returns the index of the triple in the dataset
        global dataset_rdd
        dataset_rdd = ProtoDash.convert_rdf_to_ntriples(dataset)
        index_list = [x for x, y in enumerate(rdf_dataset)]
        for i in range(len(rdf_dataset)):
            if rdf_dataset[i] == sample:
                return index_list[i]
        
        
    def get_rdf_prototypes(dataset, sample_triple, num_proto):  
        
        # Index of the sample from the dataset to be given to the ProtoDash as dataset to be explained
        # These prototypes that come out of ProtoDash can be thought as the cluster that this sample belongs to.
        # Or vice versa, the sampled datapoint can be thought as cluster centroid, and the explaining prototypes
        # as the data that belong to that cluster.        
        sample_index = ProtoDash.get_sample_index(dataset, sample_triple)
        
        if sample_index is not None:
            # Create a target dataset comprising of the selected sample
            target = [numeric_dataset[sample_index]]
            # Create a source dataset comprising of all triples but the selected sample
            source = numeric_dataset[: sample_index] + numeric_dataset[sample_index + 1 :]
        
            # Convert the datasets to PySpark RDDs
            target_rdd = ProtoDash.create_vec_rdd(target)
            source_rdd = ProtoDash.create_vec_rdd(source)
        
            print ('Starting ProtoDash on RDF')
            res, f = ProtoDash.ProtoDashAlgoritm(target_rdd, source_rdd, num_proto, 50, partitions=4, verbose=True)[:2]
            print ('Finished ProtoDash on RDF')
        
            print ('The chosen sample_index:', sample_index)
        
            # Raw RDF triples has a long common prefixes, for the sake presentation (to keep it short),
            # I strip the common long URL-like prefixes and take the suffix after '#' - the data that matters.
            stripped_target = ProtoDash.strip_rdf_prefix(rdf_dataset[sample_index])
        
            # Print the target datapoint 
            print ('Target (sampled) datapoint: ', stripped_target)
        
            # create the Y and X axis of the plot
            # The result (res) that comes from the ProtoDash is a list of pairs of weight and index
            # I use the index find the triples from the raw dataset to be used X-axis
            # and the weights are used as Y-coordinates
            values = list(map(lambda  e: e[0], res))    # e[0] is weight
            names = list(map(lambda e: rdf_dataset[e[1]], res)) # e[1] is index
            # strip the names to fit into the plot
            names = list(map(ProtoDash.strip_rdf_prefix, names))
        
            plt.barh(names, values)
            plt.title(stripped_target)
            plt.show()
        else:
            print("Please enter a valid triple")
    
    def ProtoDashOnRDF(dataset, num_proto, sample_triple):
    
        # dataset: path to the file
        # num_proto: number of prototypes for ProtoDash to select
        # sample_triple: the sample_triple is string which refer to the triple
          
        if os.path.isfile(dataset):
            if num_proto.isdigit():
                sample_triple = tuple(sample_triple.split(','))
                ProtoDash.get_rdf_prototypes(dataset, sample_triple, int(num_proto))
            else:
                print("Number of prototypes can be only integer")
        else:
            print("File do not exists")
      

    #######################################################
    #########     Image IMPLEMENTATION             ###########
    #######################################################
    
    # collect MNIST train/test sets
    train_images = np.array(mnist.train_images(), dtype='float')
    train_labels = mnist.train_labels()
    
    test_images = np.array(mnist.test_images(), dtype='float')
    test_labels = mnist.test_labels()
    
    def create_target_set(labels, images, digit, target_n, percentage):
        """
        This function creates a MNIST image dataset in which a specified percentage of the total observations
        correspond to a specific digit, while the remaining observations correspond to other randomly
        chosen digits.
        
        Args:
        -labels: the digit label for each MNIST image.
        -images: the MNIST image.
        -digit: a digit between 0 and 9.
        -target_n: the number of total observations required in the target dataset.
        -percentage: the percentage of images in the target dataset that correspond to the specified digit.
        
        Returns:
        -the target images.
        
        """
        
        # take integer number of obs. corresponding to digit
        n_dig = int( np.floor( percentage * target_n ) )
        
        # get indices corresponding to digit
        idx = np.where(labels == digit)[0]
        
        # reduce indices to specific %
        idx_red = idx[:n_dig]
        
        # slice images with index and reshape
        target_set_dig = images[idx_red,:]
        target_set_dig = np.reshape( target_set_dig, (target_set_dig.shape[0], 28*28))
        
        # get remaining indices
        rem = target_n - n_dig
        rem_ind = np.setdiff1d( np.arange(len(labels)), idx_red )[:rem]
        
        # fill the remaining observations with images corresponding to other digits
        target_set_non_dig = images[rem_ind]
        target_set_non_dig = np.reshape( target_set_non_dig, (target_set_non_dig.shape[0], 28*28))
        
        # create the dataset
        target_set = np.vstack((target_set_non_dig,target_set_dig))
        
        # shuffle it
        arr = np.arange(target_n)
        np.random.shuffle(arr)
        
        return target_set
    
    def get_image_prototypes(num_proto, digit):    
        
        part = 6         # number of Pyspark RDD partitions to use
        sigma = 50       # gaussian kernel parameter
        n_1 = 5420       # the number of observations in X_1
        n_2 = 1500       # the number of observations in X_2
        #percentages = [.3, .5, .7, .9, 1.] 
        percentages = [1.] # the percentage of X_1 that will correspond to the chosen digit
    
        # list of experiment results
        exp_1_res_list = []
    
        # list of f_eval sequences
        exp_1_f_eval_list = []
    
        # set source dataset and labels
        source_set = np.reshape( ProtoDash.test_images[:n_2], (n_2, 28 * 28) )
    
        # select the target datasets
        target_set = ProtoDash.create_target_set(ProtoDash.train_labels, ProtoDash.train_images, digit, n_1, 1)
    
        # convert target and source datasets to RDDs
        target_rdd = ProtoDash.create_vec_rdd(target_set, part)
        source_rdd = ProtoDash.create_vec_rdd(source_set, part)
    
        # collect the indices of m prototypes along with their ascribed weight
        res, f = ProtoDash.ProtoDashAlgoritm(target_rdd, source_rdd, num_proto, sigma, partitions=part,verbose=True)[:2]
        
        # collect the results
        exp_1_res_list.append(res)
        exp_1_f_eval_list.append(f)
        
        
        fig, axes= plt.subplots(num_proto, 1, figsize=(12,10), squeeze=False)
    
        for i in range (num_proto):
            for  j in range (len(percentages)):
                axes[i][j].imshow( np.reshape( source_set[exp_1_res_list[j][i][1],:], (28, 28)) )
                axes[i][j].get_xaxis().set_ticks([])
                axes[i][j].get_yaxis().set_ticks([])
         
        fig.suptitle("\n".join(wrap("Top %d prototypes selected by ProtoDash corresponding to the digit %d" 
                                    % (num_proto,digit), 60)), fontsize=20)
    
        plt.show()
        spark.stop()

    def ProtoDashOnImage(digit, num_proto):
        
        # digit: the digit to be represented in the target dataset X_1
        # num_proto: number of prototypes for ProtoDash to select
        
        if digit.isdigit() and 0 <= int(digit) <= 9:
            if num_proto.isdigit():
                ProtoDash.get_image_prototypes(int(num_proto), int(digit))
            else:
                print("Please enter an integer value for number of prototypes")
        else:
            print("Please enter a digit between 0-9")