"""
Title:Proposed2
Detail:Training for Proposed2
Config:config_model.py
       util_tf.py
       model.py
       Adam(Optimizer)
Input:data.npy
      train_data**.npy
Design:Naonori Nagano
Date:2018/02/22
"""

import time
import tensorflow as tf
import numpy as np

from prepare import *
import config_model as cfg
from model import *
from util_tf import *

def _chomp(t):
    if "<EOL>" in t:
        idx = t.index("<EOL>")
    else:
        idx = len(t)
    return " ".join(t[:idx])

def _id_to_tag(data, arr):
    return [[data["id_to_tag"][t] for t in ts] for ts in arr]

def _convert_obvious_ans(gts_t, smpls_t):
    for i in range(len(gts_t)):
        if gts_t[i][0] == "<EOL>":
            smpls_t[i] = ["<EOL>"] * len(gts_t[i])
    return smpls_t
      
def save_model(saver, sess, epoch, eval_path=cfg.test_path):
    save_path = saver.save(sess, "./model/"+eval_path+"/my-model"+eval_path, global_step=epoch_one)
    print("Model saved in file:", save_path)

def load():
    np.random.seed(0)
    tf.set_random_seed(0)

    data = prepare(cfg.input_size, "train")

    cfg.vocab_size = max(data["train"]["vocab_size"],data["test"]["vocab_size"])
    cfg.tag_size = data["tag_size"]

    m = model(cfg)
    m.build_model(mode="training")
    tf.get_variable_scope().reuse_variables()
    m.build_model(mode="inference")
    
    return data, cfg, m

if __name__ == "__main__":
    start_time = time.time()
    data, cfg, m = load()
    
    # Gradient descent
    train_step = tf.train.AdamOptimizer().minimize(m.loss)

    # Model Initialize & Session Ranning
    sess = make_session(gpu_id='0')

    # Model Save
    saver = tf.train.Saver(max_to_keep=50)

    # Generate log for TensorBoard
    writer = tf.train.SummaryWriter("tensorboard", graph=sess.graph)

    # Train Session
    max_epoch = cfg.iterations
    loss_hist, accuracy_hist = [], []
    for epoch in range(max_epoch):
        for i in range(len(data["train"]["word_i"])):
            feed_dict = {m.x_word:data["train"]["word_i"][i],
                         m.x_hrch:data["train"]["hierarchy_i"][i],
                         m.mask_tag:data["train"]["mask_tag"][i],
                         m.mask_sent:data["train"]["mask_sent"][i],
                         m.sqlength:[200]}
            _, loss = sess.run([train_step, m.loss], feed_dict = feed_dict)
            print("%sepoch" % epoch, i, loss)
            if i % 100 == 0:
                loss_hist.append(loss)
        epoch_one = epoch + 1
        if epoch == 0 or (epoch_one % 1 == 0 and epoch_one <= 10) or (epoch_one % 10 == 0 and epoch_one <= 100) or (epoch_one % 50 == 0 and epoch_one <= 1000):
            save_model(saver, sess, epoch_one)
    end_time = time.time()
    print ("Train time: " + str(end_time - start_time))
    
