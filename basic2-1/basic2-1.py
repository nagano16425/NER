"""
Title:Basic2
Detail:InputSize(200)
       Embedding(100)
       Batch(10)
       BiDirectional-LSTM(1Layer,150)
       Forget-bias(0.0)
       Adam(Optimizer)
       Dropout(0.8,0.5)
Input:Traffic*.iob2
Design:Naonori Nagano
Date:2017/07/31
"""

# Import Python Library
import collections,time,sys,os,random
# Import TensorFlow
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# Read Words
def _read_words(filename):
    f = tf.gfile.GFile(filename, "r")
    word = []
    chunk = []
    chunk_tag = []
    for line in f.readlines():
        if len(line)>3:
            pair = line.strip().split("\t")
            word.append(pair[0])
            chunk.append(pair[1])
            pair_extend = pair[1].split("-")
            if pair_extend[0] == "B" or pair_extend[0] == "I":
                chunk_tag.append(pair_extend[1])
            else:
                chunk_tag.append("<eos>")
        else:
            word.append("<eos>")
            chunk.append("<eos>")
            chunk_tag.append("<eos>")
    return word, chunk, chunk_tag

# Build Vocabulary(wordlist & chunklist)
def _build_vocab(wordlist):
    counter = collections.Counter(wordlist)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

# Making sentence list for word embedding
def _sentence_list(word_list, tag_list, word_to_id):
    sentence_list = []
    sent_tag = []
    sentence = []
    tag = []
    sent_len = []
    for i in range(len(word_list)):
        if word_list[i] != "<eos>":
            if word_list[i] in word_to_id:
                sentence.append(word_to_id[word_list[i]])
                if tag_list[i] in tag_to_id:
                    tag.append(tag_to_id[tag_list[i]])
                else:
                    tag.append(tag_size)
            else:
                sentence.append(len(word_to_id)+1)
                tag.append(tag_size)
        else:
            sent_len.append(len(sentence))
            for k in range(input_size):
                if len(sentence) < input_size:
                    sentence.append(word_to_id["<eos>"])
                    tag.append(tag_to_id["<eos>"])
                else:
                    break
            sentence_list.append(sentence)
            sent_tag.append(tag)
            sentence = []
            tag = []
    sent_len = np.array(sent_len)
    return sentence_list, sent_tag, sent_len

# Input array
def _array_list(sent_list, sent_tag):
    data = np.array(sent_list, dtype=np.int32) 
    tag = np.array(sent_tag, dtype=np.float32)
    return data, tag

# Vocablary
def _vocablary(train, test):
    train = max(map(max,train))
    test = max(map(max,test))
    vocab = max(train, test) + 1
    return vocab

# Triangle
def _triangle(input_size):
    triangle = []
    for i in range(input_size+1):
        triangle.append([1 for x in range(i)] + [0 for x in range(input_size-i)])
    return tf.constant(np.array(triangle, dtype=np.float32))

# Random number generation
def _batch_random(sentence_list, sentence_tag, sent_len, batch_size = 1):
    random_list = []
    for i in range(len(sentence_list)):
        random_list.append(i)
    random.shuffle(random_list)
    return (np.array([sentence_list[ random_list[x] ] for x in range(batch_size)], dtype=np.int32), 
            np.array([sentence_tag[ random_list[x] ] for x in range(batch_size)]), 
            np.array([sent_len[ random_list[x] ] for x in range(batch_size)], dtype=np.int32))

# Output(Result for Prediction)
def _output(iterations):
    ylist = []
    ylist_rest = []
    rest = int(len(test_data) % batch_size)
    for i in range(0,len(test_data),batch_size):
        if len(test_data[i:i+batch_size]) == batch_size:
            ylist.extend(sess.run(tf.argmax(yreshape,2),
                                  feed_dict={x:test_data[i:i+batch_size],
                                             y_:test_tag[i:i+batch_size],
                                             l:test_sent_len[i:i+batch_size],
                                             keep_prob_wv:1.0,
                                             keep_prob_out:1.0}))
    ylist_rest.extend(sess.run(tf.argmax(yreshape,2),
                               feed_dict={x:test_data[-batch_size:],
                                          y_:test_tag[-batch_size:],
                                          l:test_sent_len[-batch_size:],
                                          keep_prob_wv:1.0,
                                          keep_prob_out:1.0}))
    ylist.extend(ylist_rest[-rest:])
    del ylist_rest[:]
    # Output
    tag_prediction = []
    for i in range(len(ylist)):
        for j in range(test_sent_len[i]):
            for key, value in tag_to_id.items():
                if ylist[i][j] == value:
                    tag_prediction.append(key)
        tag_prediction.append("")
    del ylist[:]
    # Output(Writing iob2)
    with open("Evi"+eval_type+"_"+str(iterations)+".iob2", 'w', encoding="utf-8") as iob2:
        for i in range(len(tag_prediction)):
            if test_word_list[i] == "<eos>":
                iob2.write(" " + "\n")
            else:
                iob2.write(test_word_list[i] + "\t" + tag_prediction[i] + "\n")
    del tag_prediction[:]
    return
   
if __name__ == "__main__":
    random.seed(0)
    tf.set_random_seed(0)
    start_time = time.time()

    # To define the size
    input_size = 200
    batch_size = 10
    hidden_size = 150
    bi_hidden_size = 2 * hidden_size
    embed_size = 100
    eval_type = str(sys.argv[2])
    iterations = int(sys.argv[3])

    print("---read start---")

    # Input dataset
    data_path = "/home/coin/nagano.16425/tensorflow/datasets"
    train_path = os.path.join(data_path, "Traffic"+sys.argv[1]+".iob2")
    test_path = os.path.join(data_path, "Traffic"+sys.argv[2]+".iob2")
    train_word_list, train_chunk_list, train_tag_list = _read_words(train_path)
    test_word_list, test_chunk_list, test_tag_list = _read_words(test_path)

    # Make Dictionaly
    word_to_id = _build_vocab(train_word_list)
    tag_to_id = _build_vocab(train_chunk_list)
    tag_size = len(tag_to_id)+1

    # Making sentence list
    train_sent_list, train_sent_tag, train_sent_len = _sentence_list(train_word_list, train_chunk_list, word_to_id)
    test_sent_list, test_sent_tag, test_sent_len = _sentence_list(test_word_list, test_chunk_list, word_to_id)

    # Input array
    train_data, train_tag = _array_list(train_sent_list, train_sent_tag)
    test_data, test_tag = _array_list(test_sent_list, test_sent_tag)

    # Vocabulary
    vocab = _vocablary(train_data, test_data)

    # Triangle
    l_look = _triangle(input_size)

    print("---read finish---")
   
    # Word embedding(Vocabulary)
    x = tf.placeholder(tf.int32, [batch_size, input_size])
    Wembed = tf.Variable(tf.random_uniform([vocab, embed_size], -1.0, 1.0))

    # Dropout(Word Vector)
    word_vec = tf.nn.embedding_lookup(Wembed, x)
    keep_prob_wv = tf.placeholder(tf.float32)
    word_vec_drop = tf.nn.dropout(word_vec, keep_prob_wv)
    # Reshape word vector
    word_vec_reshape = tf.reshape(word_vec, [batch_size, input_size, embed_size])

    # LSTM(BiDirectional)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
    inputs = [word_vec_reshape[:,time_step,:] for time_step in range(input_size)]
    l = tf.placeholder(tf.int32, [batch_size])
    try:
        outputs, final_state, _  = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, 
                                                           dtype=tf.float32, sequence_length=l)
    except Exception:
        outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32, sequence_length=l)
    output = tf.reshape(tf.concat(1, outputs), [-1, bi_hidden_size])
  
    # Dropout(Outputs)
    keep_prob_out = tf.placeholder(tf.float32)
    output_drop = tf.nn.dropout(output, keep_prob_out)
   
    # Weight & bias
    W = tf.Variable(tf.random_uniform([bi_hidden_size, tag_size], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([tag_size], -1.0, 1.0))
    W_hist = tf.histogram_summary("weights", W)
    b_hist = tf.histogram_summary("biases", b)
   
    # Softmax
    y = tf.nn.softmax(tf.matmul(output_drop, W) + b)
    yreshape = tf.reshape(y, [batch_size, input_size, tag_size])

    # True Data
    y_ = tf.placeholder(tf.int32, [batch_size, input_size])

    # One-hot vector
    tag_one = tf.one_hot(y_, depth=tag_size)
   
    # Closs Entropy
    cross_entropy = tf.reduce_sum(-tf.reduce_sum(tag_one*tf.log(tf.clip_by_value(yreshape,1e-10,1.0)),
                                                reduction_indices=[2])
                                * tf.nn.embedding_lookup(l_look, l)
                                / tf.cast(tf.reduce_sum(l), dtype=tf.float32))
   
    # Gradient descent
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    cross_summary = tf.scalar_summary("cross_entropy", cross_entropy)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(yreshape,2), tf.argmax(y_,0))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float") 
                             * tf.nn.embedding_lookup(l_look,l)) / tf.cast(tf.reduce_sum(l), "float")
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

    # Model Initialize & Session Ranning
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # Model Save
    saver = tf.train.Saver(max_to_keep=50)

    # Generate log for TensorBoard
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("tensorboard", graph=sess.graph)
   
    # exist model
    if tf.train.get_checkpoint_state("./model/"+eval_type+"/"):
        load_path = ("./model/"+eval_type+"/my-model"+eval_type+"-"+str(iterations))
        print("Model load in file: %s" % load_path)
        saver.restore(sess, load_path)
        print ("---output start---")
        out_start_time = time.time()
        # Output(Result for Prediction)
        iob2 = _output(iterations)
        out_end_time = time.time()
        print("output time: " + str(out_end_time - out_start_time))
        print ("---output finish---")
    else:
        print ("---train start---")
        # Training
        for i in range(iterations):
            batch_xs, batch_ys, batch_len = _batch_random(train_data, train_tag, train_sent_len, batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys, l:batch_len, keep_prob_wv:0.8, keep_prob_out:0.5})
            result = sess.run(merged, feed_dict={x:train_data[0:batch_size],
                                                y_:train_tag[0:batch_size],
                                                l:train_sent_len[0:batch_size],
                                                keep_prob_wv:1.0,
                                                keep_prob_out:1.0})
            writer.add_summary(result, i)
            if (i+1)%10 == 0:
                print("count "+str(i+1), end = " ")
                sys.stdout.flush()
                if (i+1)%100 == 0:
                    print()
                    print(sess.run((cross_entropy, accuracy),
                                    feed_dict={x:train_data[0:batch_size],
                                               y_:train_sent_tag[0:batch_size],
                                               l:train_sent_len[0:batch_size],
                                               keep_prob_wv:1.0,
                                               keep_prob_out:1.0}))
            j = i+1
            if i == 0 or (j % 100 == 0 and j <= 1000) or (j % 1000 == 0 and j <= 10000) or (j % 5000 == 0 and j <= 100000):
                save_path = saver.save(sess, "./model/"+eval_type+"/my-model"+eval_type, global_step=i+1)
                print("Model saved in file: %s" % save_path)
        print ("---train finish---")

    # Calculate Time
    end_time = time.time()
    print ("calc time: " + str(end_time - start_time))
