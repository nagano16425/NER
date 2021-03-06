"""
Title:Proposed2
Detail:Configuration for Proposed2
Design:Naonori Nagano
Date:2018/02/22
"""

import tensorflow as tf
import numpy as np

# Model Initialize & Session Ranning
def make_session(gpu_id='0', allow_growth=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    config.gpu_options.visible_device_list = gpu_id
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    return sess
