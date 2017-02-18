import tensorflow as tf
import numpy as np
import sys
import time

class S2S(object):

    def __init__(self, x_len, y_len, x_vocab_size, y_vocab_size, emb_dim, num_layers, ckpt_path, lr=0.001, epochs=10000, model_name='s2s_model'):

        self.x_len = x_len
        self.y_len = y_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name

        def __graph__():
            tf.reset_default_graph()
            #ulaz na enkoder - maksimalan broj reci koji je dozvoljen x_len
            self.encoder_input = [tf.placeholder(shape=[None,], dtype=tf.int64,
            name='ei_{}'.format(t)) for t in range(x_len)]
            #
            self.labels = [ tf.placeholder(shape=[None,], dtype=tf.int64, 
            name='ei_{}'.format(t)) for t in range(y_len)]
            #
            self.decoder_input = [ tf.zeros_like(self.encoder_input[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]
            #

            self.keep_prob = tf.placeholder(tf.float32)
            #RNN Celija - uzimamo GRU zbog brzine
            cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
                output_keep_prob = self.keep_prob)

            #Povezujemo vise RNN Celija
            stacked_cells = tf.nn.rnn_cell.MultiRNNCell(
                [cell]*num_layers, 
                state_is_tuple=True
            )

            with tf.variable_scope('decoder') as scope:
                self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(
                    self.encoder_input,
                    self.decoder_input,
                    stacked_cells,
                    x_vocab_size,
                    y_vocab_size,
                    emb_dim
                )

                scope.reuse_variables()

                self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                    self.encoder_input,
                    self.decoder_input,
                    stacked_cells,
                    x_vocab_size,
                    y_vocab_size,
                    emb_dim,
                    feed_previous=True
                )

            #loss funkcija
            loss_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.labels]
            self.loss = tf.nn.seq2seq.sequence_loss(
                self.decode_outputs,
                self.labels,
                loss_weights,
                y_vocab_size
            )

            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        sys.stdout.write('Building graph...\n')
        __graph__()
        sys.stdout.write('Done...\n')

    """
    training
    """
    
    def get_feed(self, x, y, keep_prob):
        feed_dict = {self.encoder_input[t]: x[t] for t in range(self.x_len)}
        feed_dict.update({self.labels[t]: y[t] for t in range(self.y_len)})
        feed_dict[self.keep_prob] = keep_prob
        return feed_dict

    def train_batch(self, sess, train_batch_gen):
        batchX, batchY = train_batch_gen.__next__()

        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def evaluate_steps(self, sess, eval_batch_gen):
        batchX, batchY = eval_batch_gen.__next__()

        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)

        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
        return loss_v, dec_op_v, batchX, batchY

    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.evaluate_steps(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    def train(self, train_set, valid_set, sess=None):
        saver = tf.train.Saver()

        if not sess:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            
        sys.stdout.write('>>>Training started...\n')

        for i in range(self.epochs):
            try:
                start = time.time()
                loss = self.train_batch(sess, train_set)
                end = time.time()

                print('Loss', loss, 'at iteration', i, '/', self.epochs, 'Time -', end - start)

                save_every = 1000

                if i and i%500 == 0:
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    val_loss = self.eval_batches(sess, valid_set, 16)

                    print('Model saved after', i, 'iterations.')
                    print('Validate loss:', val_loss)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print('Interrupted by user at iteration', i)
                self.session = sess
                return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess

    def predict(self, sess, x):
        feed_dict = {self.encoder_input[t]: x[t] for t in range(self.x_len)}
        feed_dict[self.keep_prob] = 1.0
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)

        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])

        return np.argmax(dec_op_v, axis=2)

    def advance_predict(self, sess, x, axis=3):
        feed_dict = {self.encoder_input[t]: x[t] for t in range(self.x_len)}
        feed_dict[self.keep_prob] = 1.0
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)

        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])

        if axis == 3:
            return np.argmax(dec_op_v)
        else:
            return np.argmax(dec_op_v, axis=axis)
