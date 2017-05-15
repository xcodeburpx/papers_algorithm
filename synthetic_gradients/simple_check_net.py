import tensorflow as tf
import numpy as np
from dataset import generate_dataset

class VanillaNetwork:

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(shape=[None,24], dtype=tf.float32)
            self.y_ = tf.placeholder(shape=[None,12], dtype=tf.float32)

            # Normal Network Layer weights

            self.layer_1_weights = {"weights": tf.Variable(tf.truncated_normal([input_dim, hidden_dim_1]), name="l_1_weights"),
                                    "biases": tf.Variable(tf.zeros([hidden_dim_1]), name="l_1_biases")}
            self.layer_2_weights = {"weights": tf.Variable(tf.truncated_normal([hidden_dim_1, hidden_dim_2]), name="l_2_weights"),
                                    "biases": tf.Variable(tf.zeros([hidden_dim_2]), name="l_2_biases")}
            self.output_weights = {"weights": tf.Variable(tf.truncated_normal([hidden_dim_2, output_dim]), name="output_weights"),
                                   "biases": tf.Variable(tf.zeros([output_dim]), name="output_biases")}

            #Synthetic Gradients Layer weights

            self.layer_1_synth_weights = {"weights": tf.Variable(tf.truncated_normal([hidden_dim_1, hidden_dim_1]), name="l_1_synth_weights"),
                                          "biases": tf.Variable(tf.zeros([hidden_dim_1]), name="l_1_synth_biases")}
            self.layer_2_synth_weights = {"weights": tf.Variable(tf.truncated_normal([hidden_dim_2, hidden_dim_2]), name="l_2_synth_weights"),
                                          "biases": tf.Variable(tf.zeros([hidden_dim_2]), name="l_2_synth_biases")}

            #Normal Operations
            self.layer_1_add = tf.add(tf.matmul(self.x, self.layer_1_weights['weights']), self.layer_1_weights['biases'])
            self.layer_1 = tf.nn.sigmoid(self.layer_1_add)

            self.layer_2_add = tf.add(tf.matmul(self.layer_1, self.layer_2_weights['weights']), self.layer_2_weights['biases'])
            self.layer_2 = tf.nn.sigmoid(self.layer_2_add)

            self.output = tf.matmul(self.layer_2, self.output_weights['weights']) + self.output_weights['biases']

            self.cost = tf.reduce_mean(tf.abs(tf.nn.sigmoid(self.output) - self.y_))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

            self.init = tf.global_variables_initializer()



    def train(self, sess, data, batch_size=10, iterations=1000):
        x_train, y_train = data

        for iter in range(iterations):
            loss = 0
            error = 0
            for batch_i in range(int(len(x_train)/batch_size)):
                batch_x = x_train[(batch_i * batch_size):(batch_i+1)*batch_size]
                batch_y = y_train[(batch_i * batch_size):(batch_i+1)*batch_size]

                _,loss = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y_: batch_y})

                error += loss
            if (iter+1) % 100 == 0:
                print("Iter:" +str(iter)+" Loss:"+str(error))


if __name__ == "__main__":
    tf.reset_default_graph()
    data = generate_dataset(12, 1000)
    network = VanillaNetwork(24, 128, 64, 12)

    with tf.Session(graph=network.graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(network.init)
        network.train(sess, data, batch_size=10, iterations=1000)