"""
Title:Proposed2
Detail:Test for Proposed2
Config:config_model.py
       util_tf.py
       model.py
       Adam(Optimizer)
Input:data.npy
      train_data**.npy
      my-model*
Design:Naonori Nagano
Date:2018/02/22
"""

import time
import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

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

def validation(k="test"):
    loss_list, samples_list, accuracy_list, mask_xy_sum_list = [], [], [], []
    for i in range(len(data[k]["word_i"])):
        feed_dict = {m.x_word:data[k]["word_i"][i], m.x_hrch:data[k]["hierarchy_i"][i], m.mask_tag:data[k]["mask_tag"][i], m.mask_sent:data[k]["mask_sent"][i], m.sqlength:[200], 
        m.x_word_inf:data[k]["word_i"][i], m.x_hrch_inf:data[k]["hierarchy_i"][i], m.mask_tag_inf:data[k]["mask_tag"][i], m.mask_sent_inf:data[k]["mask_sent"][i], m.sqlength_inf:[200]}        
        loss, samples, accuracy, mask_xy_sum = sess.run([m.loss, m.sampled_tag_inf, m.accuracy, m.mask_xy_sum], feed_dict = feed_dict)
        loss_list.append(loss)
        samples_list.append(samples)
        accuracy_list.append(accuracy)
        mask_xy_sum_list.append(mask_xy_sum)
    return loss_list, samples_list, accuracy_list, mask_xy_sum_list

def f_measure(data, true, pred):
    pred = np.array(pred)
    target_names = [None] * cfg.tag_size
    for k, v in data["tag_to_id"].items():
        target_names[v] = k
    print(classification_report(true.flatten(), pred.flatten(), target_names=target_names))
    precision, recall, f_score, s = precision_recall_fscore_support(true.flatten(), pred.flatten())
    return precision, recall, f_score
    
def output(word, tag):
    idtotag = data["id_to_tag"]
    pre_tag = [[[idtotag[t] for t in ts] for ts in tss] for tss in tag]
    with open("Evi"+str(cfg.test_path)+"_"+str(cfg.iterations)+".iob2", 'w', encoding="utf-8") as iob1:
        with open("EviHRNN"+str(cfg.test_path)+"_"+str(cfg.iterations)+".iob2", 'w', encoding="utf-8") as iob2:
            for i in range(len(word)):
                for j in range(len(word[i])):
                    iob1.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\n")
                    if pre_tag[i][j][0] == "<EOL>":
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\n")
                    elif pre_tag[i][j][1] == "<EOL>":
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\t" + pre_tag[i][j][1] + "\n")
                    elif pre_tag[i][j][2] == "<EOL>":
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\t" + pre_tag[i][j][1] + "\t" + pre_tag[i][j][2] + "\n")
                    elif pre_tag[i][j][3] == "<EOL>":
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\t" + pre_tag[i][j][1] + "\t" + pre_tag[i][j][2] + "\t" + pre_tag[i][j][3] + "\n")
                    elif pre_tag[i][j][4] == "<EOL>":
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\t" + pre_tag[i][j][1] + "\t" + pre_tag[i][j][2] + "\t" + pre_tag[i][j][3] + "\t" + pre_tag[i][j][4] + "\n")
                    elif pre_tag[i][j][5] == "<EOL>":
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\t" + pre_tag[i][j][1] + "\t" + pre_tag[i][j][2] + "\t" + pre_tag[i][j][3] + "\t" + pre_tag[i][j][4] + "\t" + pre_tag[i][j][5] + "\n")
                    elif pre_tag[i][j][6] == "<EOL>":
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\t" + pre_tag[i][j][1] + "\t" + pre_tag[i][j][2] + "\t" + pre_tag[i][j][3] + "\t" + pre_tag[i][j][4] + "\t" + pre_tag[i][j][5] + "\t" + pre_tag[i][j][6] + "\n")
                    else:
                        iob2.write(word[i][j] + "\t" + pre_tag[i][j][0] + "\t" + pre_tag[i][j][1] + "\t" + pre_tag[i][j][2] + "\t" + pre_tag[i][j][3] + "\t" + pre_tag[i][j][4] + "\t" + pre_tag[i][j][5] + "\t" + pre_tag[i][j][6] + "\t" + pre_tag[i][j][7] + "\n")
                iob1.write(" " + "\n")
                iob2.write(" " + "\n")
    return
      
def load_model(saver, sess, iterations, eval_path=cfg.test_path):
    load_path = ("./model/"+eval_path+"/my-model"+eval_path+"-"+str(iterations))
    print("Model load in file: %s" % load_path)
    saver.restore(sess, load_path)

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

    # Load Tensorflow Model
    if tf.train.get_checkpoint_state("./model/"+cfg.test_path+"/"):
        load_model(saver, sess, cfg.iterations)
        
    # Test Session
    loss_list, samples_list, accuracy_list, mask_xy_sum_list = validation()
    output(data["test"]["word"], samples_list)
    
    # If you want to use Scikit, enable the following code
    # precision, recall, f_score = f_measure(data, data["test"]["hierarchy_i"], samples_list)
    
    # If you want to get accuragy, enable the following code
    
    # loss = np.sum(loss_list)
    # correct_num = np.multiply(accuracy_list,mask_xy_sum_list).sum()
    # accuracy = correct_num / np.sum(mask_xy_sum_list)
    
    end_time = time.time()
    print ("Output time: " + str(end_time - start_time))
    
