"""
Title:Proposed2
Detail:Configuration for Proposed2
Design:Naonori Nagano
Date:2018/2/22
"""

# To define data setting
import sys
import tensorflow as tf

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])
iterations = int(sys.argv[3])

# To define the size
input_size = 200
batch_size = 1
hidden_size = 150
bi_hidden_size = 2 * hidden_size
hrch_lstm_size = 150
hrch_gru_size = 150
embed_size = 100
hierarchy_size = 8
hrch_embed_size = 100
tree_size = 2

# Set up Dropout
keep_prob_wv = 0.8
dropout_wv = True
keep_w_lstm_out = 0.5
dropout_w_lstm_out = True
