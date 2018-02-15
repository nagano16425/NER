"""
Title:Proposed2
Detail:Model for Proposed2(BLSTM+GRU)
Design:Naonori Nagano
Date:2018/1/15
"""

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

class model():
    def __init__(self, cfg):
        self.cfg = cfg

        self.Wembed = tf.Variable(tf.random_uniform([cfg.vocab_size, cfg.embed_size], -1.0, 1.0), name="WordEmbedding")
        self.Hembed = tf.Variable(tf.random_uniform([cfg.tag_size, cfg.hrch_embed_size], -1.0, 1.0), name="HierarchyEmbedding")

        self.init_hrch_gru_h_W = self.init_weight(cfg.bi_hidden_size, cfg.hrch_gru_size, name='init_hrch_gru_h_W')
        self.init_hrch_gru_h_b = self.init_bias(cfg.hrch_gru_size, name='init_hrch_gru_h_b')
        
        self.hrch_gru_U = self.init_weight(cfg.hrch_gru_size, cfg.hrch_gru_size, name='hrch_gru_U')
        self.hrch_gru_Wx = self.init_weight(cfg.hrch_embed_size, cfg.hrch_gru_size, name='hrch_gru_Wx')
        self.hrch_gru_Wr = self.init_weight(cfg.hrch_embed_size, cfg.hrch_gru_size, name='hrch_gru_Wr')
        self.hrch_gru_Wz = self.init_weight(cfg.hrch_embed_size, cfg.hrch_gru_size, name='hrch_gru_Wz')
        self.hrch_gru_br = self.init_bias(cfg.hrch_gru_size, name='hrch_gru_br')
        self.hrch_gru_bz = self.init_bias(cfg.hrch_gru_size, name='hrch_gru_br')
        self.context_encode_gru_W = self.init_weight(cfg.bi_hidden_size, cfg.hrch_gru_size, name='context_encode_gru_W')
        
        self.decode_hrch_gru_W = self.init_weight(cfg.hrch_gru_size, cfg.tag_size, name='decode_hrch_gru_W')
        self.decode_hrch_gru_b = self.init_bias(cfg.tag_size, name='decode_hrch_gru_b')

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

    def init_gru(self, context, w_h, b_h):
        h = tf.nn.tanh(tf.matmul(context, w_h) + b_h)
        return h
        
    def init_hrch_gru(self, context):
        w_h, b_h = self.init_hrch_gru_h_W, self.init_hrch_gru_h_b
        return self.init_gru(context, w_h, b_h)
        
    def gru(self, h, x, context, wu, ww):
        z = tf.nn.sigmoid(tf.matmul(x, self.hrch_gru_Wz) + self.hrch_gru_bz + tf.matmul(context, ww))
        r = tf.nn.sigmoid(tf.matmul(x, self.hrch_gru_Wr) + self.hrch_gru_br + tf.matmul(context, ww))
        h_ = tf.nn.tanh(tf.matmul(x, self.hrch_gru_Wx) + tf.matmul(h, wu) * r)
        h = tf.mul((1-z), h_) + tf.mul(h, z)
        return h
        
    def hrch_gru(self, h, x, context):
        u, w = self.hrch_gru_U, self.context_encode_gru_W
        return self.gru(h, x, context, u, w)

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

        h = self.init_hrch_gru(BLSTM_dropout)

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

            h = self.hrch_gru(h, hrch_vec, BLSTM)
            
            logits = tf.matmul(h, self.decode_hrch_gru_W) + self.decode_hrch_gru_b

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
            
