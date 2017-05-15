import tensorflow as tf
import numpy as np
from dataset import generate_dataset

class DNI:

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, alpha=0.001):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.alpha = alpha

            def update(weights, up_data, alph):
                weights -= up_data * alph
                return weights

            def bi_aver(biases, update, alph):
                biases -= tf.reduce_mean(update, axis=0)*alph
                return biases


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

            #Synthetic Grads Operations
            with tf.device("/cpu:0"):
                self.synth_grad_1 = tf.add(tf.matmul(self.layer_1, self.layer_1_synth_weights['weights']), self.layer_1_synth_weights['biases'])
                self.synth_grad_2 = tf.add(tf.matmul(self.layer_2, self.layer_2_synth_weights['weights']), self.layer_2_synth_weights['biases'])

                #Sythetic Gradient Calculation
                self.weight_synth_grad_1 = self.synth_grad_1 * tf.gradients(self.layer_1, self.layer_1_add)[0]
                self.weight_synth_grad_2 = self.synth_grad_2 * tf.gradients(self.layer_2, self.layer_2_add)[0]
                #Update weights
                self.up_1 = tf.matmul(tf.transpose(self.x),self.weight_synth_grad_1)
                self.up_2 = tf.matmul(tf.transpose(self.layer_1),self.weight_synth_grad_2)

                #Full Update
                self.update_synth_1 = update(self.layer_1_weights['weights'], self.up_1, self.alpha)
                self.update_synth_2 = update(self.layer_2_weights['weights'], self.up_2, self.alpha)
                self.bi_up_1 = bi_aver(self.layer_1_weights['biases'], self.weight_synth_grad_1, self.alpha)
                self.bi_up_2 = bi_aver(self.layer_2_weights['biases'], self.weight_synth_grad_2, self.alpha)

                self.grad_1 = tf.gradients(self.output, self.layer_1)[0]
                self.grad_2 = tf.gradients(self.output, self.layer_2)[0]
                self.synth_true_1 = self.synth_grad_1 - self.grad_1
                self.synth_true_2 = self.synth_grad_2 - self.grad_2

                self.synth_up_1 = tf.matmul(tf.transpose(self.synth_true_1),self.layer_1)
                self.synth_up_2 = tf.matmul(tf.transpose(self.synth_true_2),self.layer_2)

                self.s_up_1 = update(self.layer_1_synth_weights['weights'], self.synth_up_1, self.alpha)
                self.s_up_2 = update(self.layer_2_synth_weights['weights'], self.synth_up_2, self.alpha)
                self.s_bi_1 = bi_aver(self.layer_1_synth_weights['biases'], self.synth_true_1, self.alpha)
                self.s_bi_2 = bi_aver(self.layer_2_synth_weights['biases'], self.synth_true_2, self.alpha)

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

                _,_,_,_ = sess.run([self.update_synth_1, self.update_synth_2, self.bi_up_1, self.bi_up_2],
                                   feed_dict={self.x: batch_x, self.y_: batch_y})

                _,loss = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y_: batch_y})

                _,_,_,_ = sess.run([self.s_up_1, self.s_up_2, self.s_bi_1, self.s_bi_2],
                                   feed_dict={self.x: batch_x, self.y_: batch_y})

                error += loss
            if (iter+1) % 100 == 0:
                print("Iter:" +str(iter)+" Loss:"+str(error))


if __name__ == "__main__":
    tf.reset_default_graph()
    data = generate_dataset(12, 1000)
    network = DNI(24, 128, 64, 12)

    with tf.Session(graph=network.graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(network.init)
        network.train(sess, data, batch_size=10, iterations=1000)