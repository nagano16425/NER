"""
Title:Basic4
Detail:Model for Basic4(BLSTM+LSTM)
Design:Naonori Nagano
Date:2018/02/22
"""

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

class model():
    def __init__(self, cfg):
        self.cfg = cfg

        self.Wembed = tf.Variable(tf.random_uniform([cfg.vocab_size, cfg.embed_size], -1.0, 1.0), name="WordEmbedding")
        self.Hembed = tf.Variable(tf.random_uniform([cfg.tag_size, cfg.hrch_embed_size], -1.0, 1.0), name="HierarchyEmbedding")

        self.init_hrch_lstm_h_W = self.init_weight(cfg.bi_hidden_size, cfg.hrch_lstm_size, name='init_hrch_lstm_h_W')
        self.init_hrch_lstm_h_b = self.init_bias(cfg.hrch_lstm_size, name='init_hrch_lstm_h_b')
        self.init_hrch_lstm_c_W = self.init_weight(cfg.bi_hidden_size, cfg.hrch_lstm_size, name='init_hrch_lstm_c_W')
        self.init_hrch_lstm_c_b = self.init_bias(cfg.hrch_lstm_size, name='init_hrch_lstm_c_b')

        self.hrch_lstm_U = self.init_weight(cfg.hrch_lstm_size, cfg.hrch_lstm_size*4, name='hrch_lstm_U')
        self.hrch_lstm_W = self.init_weight(cfg.hrch_embed_size, cfg.hrch_lstm_size*4, name='hrch_lstm_W')
        self.hrch_lstm_b = self.init_bias(cfg.hrch_lstm_size*4, name='hrch_lstm_b')
        self.context_encode_W = self.init_weight(cfg.bi_hidden_size, cfg.hrch_lstm_size*4, name='context_encode_W')

        self.decode_hrch_lstm_W = self.init_weight(cfg.hrch_lstm_size, cfg.tag_size, name='decode_hrch_lstm_W')
        self.decode_hrch_lstm_b = self.init_bias(cfg.tag_size, name='decode_hrch_lstm_b')

    def _bidirectionalLSTM(self, word_vec):
        with tf.variable_scope('forward1',reuse=True):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cfg.hidden_size, forget_bias = 0.0, state_is_tuple = True)
        with tf.variable_scope('backward1',reuse=True):
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cfg.hidden_size, forget_bias = 0.0, state_is_tuple = True)
        inputs = [word_vec[:,time_step,:] for time_step in range(self.cfg.input_size)]
        l = tf.placeholder(tf.int32, [self.cfg.batch_size])
        outputs, final_state, _  = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32, sequence_length=l)
        output = tf.reshape(tf.pack(outputs),[-1, self.cfg.bi_hidden_size])
        return output, l, final_state

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0, scale=1.0):
        return tf.Variable(scale * tf.truncated_normal([dim_in, dim_out], stddev=stddev/np.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def init_lstm(self, context, w_h, b_h, w_c, b_c):
        h = tf.nn.tanh(tf.matmul(context, w_h) + b_h)
        c = tf.nn.tanh(tf.matmul(context, w_c) + b_c)
        return h, c

    def init_hrch_lstm(self, context):
        w_h, b_h = self.init_hrch_lstm_h_W, self.init_hrch_lstm_h_b
        w_c, b_c = self.init_hrch_lstm_c_W, self.init_hrch_lstm_c_b
        return self.init_lstm(context, w_h, b_h, w_c, b_c)

    def lstm(self, h, c, x_t, context, wu, ww):
        preactive = tf.matmul(h, wu) + x_t + tf.matmul(context, ww)
        i, f, o, u = tf.split(1, 4, preactive)

        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.nn.tanh(u)

        c = f * c + i * u
        h = o * tf.nn.tanh(c)
        return h, c

    def hrch_lstm(self, h, c, x_t, context):
        u, w = self.hrch_lstm_U, self.context_encode_W
        return self.lstm(h, c, x_t, context, u, w)

    def build_model(self, mode):
        x_word = tf.placeholder(tf.int32, [self.cfg.input_size], name="WordInputs")
        x_hrch = tf.placeholder(tf.int32, [self.cfg.input_size, self.cfg.hierarchy_size], name="HierarchyInputs")
        mask_tag = tf.placeholder(tf.float32, [self.cfg.input_size, self.cfg.hierarchy_size], name="HierarchyMasks")
        mask_sent = tf.placeholder(tf.float32, [self.cfg.input_size], name="SentenceMasks")

        word_vec = tf.nn.embedding_lookup(self.Wembed, x_word)
        if self.cfg.dropout_wv:
            word_vec = tf.nn.dropout(word_vec, self.cfg.keep_prob_wv)
        word_vec_reshape = tf.reshape(word_vec, [self.cfg.batch_size, self.cfg.input_size, self.cfg.embed_size])

        BLSTM, sqlength, BLSTM_state = self._bidirectionalLSTM(word_vec_reshape)
        
        if self.cfg.dropout_w_lstm_out:
            BLSTM_dropout = tf.nn.dropout(BLSTM, self.cfg.keep_w_lstm_out)

        h, c = self.init_hrch_lstm(BLSTM_dropout)

        cross_entropy_list = []
        sampled_tag = []
        loss = 0
        for t in range(self.cfg.hierarchy_size):
            if t == 0:
                hrch_vec = tf.zeros([self.cfg.input_size, self.cfg.embed_size])
            else:
                if mode == 'training':
                    st = x_hrch[:,t-1]
                elif mode == 'inference':
                    st = next_tag
                with tf.device("/cpu:0"):
                    hrch_vec = tf.nn.embedding_lookup(self.Hembed, st)

            x_t = tf.matmul(hrch_vec, self.hrch_lstm_W) + self.hrch_lstm_b

            h, c = self.hrch_lstm(h, c, x_t, BLSTM)
            
            logits = tf.matmul(h, self.decode_hrch_lstm_W) + self.decode_hrch_lstm_b

            if mode == 'training':
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, x_hrch[:,t])
                cross_entropy = cross_entropy * mask_tag[:,t] * mask_sent
                cross_entropy_list.append(cross_entropy)
                current_loss = tf.reduce_sum(cross_entropy)
                loss += current_loss

            if mode == 'inference':
                next_tag = tf.argmax(logits, 1)
                sampled_tag.append(next_tag)

        if mode == 'training':
            loss /= tf.reduce_sum(mask_tag)
            cross_entropy_list = tf.transpose(cross_entropy_list)
            sampled_tag = tf.transpose(sampled_tag)

        if mode == 'inference':
            sampled_tag = tf.transpose(sampled_tag)
            # Accuracy
            mask_xy = mask_tag * tf.reshape(mask_sent, [self.cfg.input_size, 1])
            eq = tf.equal(tf.cast(sampled_tag, tf.int32), x_hrch)
            eq = tf.mul(tf.cast(eq, tf.int32), tf.cast(mask_xy, tf.int32))
            mask_xy_sum = tf.reduce_sum(mask_xy)
            accuracy = tf.cast(tf.reduce_sum(eq), tf.float32) / mask_xy_sum
            # accuracy = tf.contrib.metrics.accuracy(sampled_tag, tf.cast(x_hrch, tf.int64))

        if mode == 'training':
            self.sqlength = sqlength
            self.x_word = x_word
            self.x_hrch = x_hrch
            self.mask_tag = mask_tag
            self.mask_sent = mask_sent

            self.cross_entropy_list = cross_entropy_list
            self.loss = loss
            self.sampled_tag = sampled_tag
            
        if mode == 'inference':
            self.sqlength_inf = sqlength
            self.x_word_inf = x_word
            self.x_hrch_inf = x_hrch
            self.mask_tag_inf = mask_tag
            self.mask_sent_inf = mask_sent

            self.sampled_tag_inf = sampled_tag
            self.accuracy = accuracy
            self.eq = eq
            self.mask_xy_sum = mask_xy_sum
            
