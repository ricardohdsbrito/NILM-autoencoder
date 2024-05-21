import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle

class KMC:
    def __init__(self, k, n, iters):
        self.k = k
        self.n = n
        self.iters = iters
    
        
    def k_means_centroid(self):

        #with tf.device('/device:GPU:0'):
        
	    "Randomly initialize K centroids"

	    centroids = tf.Variable(tf.random_uniform([self.k, self.n]))

	    centroids_expanded = tf.expand_dims(centroids, 1)

	    X = tf.placeholder(tf.float32, [None, 275])
	    X_expanded = tf.expand_dims(X, 0)

	    distances = tf.reduce_sum(tf.square(tf.subtract(X_expanded, centroids_expanded)), 2)
	    indexes = tf.argmin(distances, axis=0)

	    means = []

	    for c in xrange(self.k):
                means.append(tf.reduce_mean(
			tf.gather(X,
			 tf.reshape(
				tf.where(
					tf.equal(indexes, c)), [1,-1])
			), reduction_indices=[1]))

	    new_centroids = tf.concat(means, 0)


	    #which rows in means are not nans?
	    nan_test = tf.where(~tf.is_nan(means))

	    update_indexes, _ = tf.unique(tf.gather(tf.transpose(nan_test, [1,0]), 0))
	    update_values = tf.gather(new_centroids, update_indexes)


	    update_centroids = tf.scatter_update(centroids, update_indexes, update_values)
	    assign_centroids = tf.assign(centroids, update_centroids)


	    #return X, centroids, new_centroids, indexes
        
	    return X, indexes, assign_centroids

    def run_kmc(self, data):
        
        X, indexes, assign_centroids = self.k_means_centroid()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        labels = []

        for iteration in tqdm(range(self.iters)):
        
            sess.run(assign_centroids, feed_dict={X:data})

            if iteration == self.iters - 1:
                labels = sess.run(indexes, feed_dict={X:data}).tolist()

        self.labels = labels
        
    def save(self, house):
        path = 'REDD/' + house + '/labels.npy'
        with open(path, 'wb') as f:
            pickle.dump(self.labels, f)
            f.close()

    def load(self, house, base = './NILM_datasets/', save_path=''):
        
        path = base + 'REDD/' + house + save_path + '/labels.npy'
        with open(path, 'rb') as f:
            self.labels = pickle.load(f)
            f.close()
