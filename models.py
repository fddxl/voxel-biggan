import logging
import nets
import numpy as np
import ops
import os
import tensorflow as tf
import utils


class Base(object):

    def run_train(self, dataset, params_path, out_path, batch_size, n_iters, log_interval, save_interval):
        total_batch = int(dataset.train.n_examples / batch_size)
        test_data = dataset.test.next_batch(batch_size)

        for epoch in range(1, n_iters+1):
            for batch in range(total_batch):
                train_data = dataset.train.next_batch(batch_size)
                self.optimize(*train_data)

                if batch % log_interval == 0:
                    self.save_log(train_data, test_data, epoch, batch, out_path)

            if epoch % save_interval == 0:
                filename = os.path.join(params_path, 'epoch_{0:03d}'.format(epoch))
                self.saver.save(self.sess, filename)


class Stage0(Base):
    """
    This stage predicts camera position of input sketch
    """
    def __init__(self, sess, batch_size, npx, learning_rate, beta1):
        self.sess = sess
        self.netR = nets.Regressor()

        self.build_network(batch_size, npx)
        self.opt = tf.train.AdamOptimizer(learning_rate, beta1).minimize(self.loss, var_list=self.vars)

        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(self.variables_to_save)

    def build_network(self, batch_size, npx):
        x = tf.placeholder(tf.float32, [batch_size, npx, npx, 1], 'x')
        y = tf.placeholder(tf.float32, [batch_size, 2], 'y')

        # Networks
        y_ = self.netR.stage0(x, 8, 2)
        self.y_ = y_

        # Variables
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if var.name.startswith('R_stage0')]

        self.variables_to_save = vars
        self.vars = vars

        # Loss function
        self.loss = tf.reduce_mean(tf.squared_difference(y_, y))

    def optimize(self, x, y):
        fd = {'x:0': x, 'y:0': y}
        self.sess.run(self.opt, feed_dict=fd)

    def get_errors(self, x, y):
        fd = {'x:0': x, 'y:0': y}
        loss = self.sess.run(self.loss, feed_dict=fd)
        return loss

    def predict(self, x):
        fd = {'x:0': x}
        return self.sess.run(self.y_, feed_dict=fd)

    def save_log(self, train_data, test_data, epoch, batch, out_path):
        train_loss = self.get_errors(*train_data)
        test_loss = self.get_errors(*test_data)
        logging.info('{0:>3}, {1:>5}, {2:.8f}, {3:.8f}'.format(epoch, batch, train_loss, test_loss))
