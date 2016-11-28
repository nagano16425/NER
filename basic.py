"""
Title:Basic(Interim Report)
Detail:InputSize(200)
       Embedding(100)
       Batch(10)
       LSTM(1Layer,150cell)
       Adam(Optimizer)
       Dropout(0.8)
Input:crfnew**_chunk.iob2
Design:Naonori Nagano
Date:2016/11/22
"""

# Import Python Library
import collections,time,sys,os,random
# Import TensorFlow
import tensorflow as tf
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

# Chunk change to array
def _tag_change(tag):
   tag_data = [0 for i in range(tag_size)]
   if not tag == "<eos>" and tag in tag_to_id:
      tag_data[tag_to_id[tag]] = 1
   elif tag == "<eos>":
      tag_data[tag_to_id["<eos>"]] = 1
   else:
      tag_data[tag_size-1] = 1
   return tag_data
      
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
            tag.append(_tag_change(tag_list[i]))
         else:
            sentence.append(len(word_to_id)+1)
            tag.append(_tag_change(tag_list[i]))
      else:
         sent_len.append(len(sentence))
         for k in range(input_size):
            if len(sentence) < input_size:
               sentence.append(word_to_id["<eos>"])
               tag.append(_tag_change("<eos>"))
            else:
               break
         sentence_list.append(sentence)
         sent_tag.append(tag)
         sentence = []
         tag = []
   sent_len = np.array(sent_len)
   return sentence_list, sent_tag, sent_len

# Random number generation
def _batch_random(sentence_list, sentence_tag, sent_len, batch_size = 1):
   random_list = []
   for i in range(len(sentence_list)):
      random_list.append(i)
   random.shuffle(random_list)
   return (np.array([sentence_list[ random_list[x] ] for x in range(batch_size)], dtype=np.int32),
           np.array([sentence_tag[ random_list[x] ] for x in range(batch_size)]),
           np.array([sent_len[ random_list[x] ] for x in range(batch_size)], dtype=np.int32))

if __name__ == "__main__":
   random.seed(0)
   tf.set_random_seed(0)
   start_time = time.time()

   # To define the size
   input_size = 200
   batch_size = 10
   hidden_size = 150
   embed_size = 100
   iterations = int(sys.argv[3])
   
   print("---read start---")

   # Input dataset
   data_path = "/home/nagano.16425/tensorflow/datasets/"
   train_path = os.path.join(data_path, sys.argv[1])
   test_path = os.path.join(data_path, sys.argv[2])
   train_word_list, train_chunk_list, train_tag_list = _read_words(train_path)
   test_word_list, test_chunk_list, test_tag_list = _read_words(test_path)
   
   # Make Dictionaly
   word_to_id = _build_vocab(train_word_list)
   tag_to_id = _build_vocab(train_chunk_list)
   tag_size = len(tag_to_id)+1

   # Making sentence list
   train_sent_list, train_sent_tag, train_sent_len = _sentence_list(train_word_list, train_chunk_list, word_to_id)
   test_sent_list, test_sent_tag, test_sent_len = _sentence_list(test_word_list, test_chunk_list, word_to_id)
   
   # Input array(Tensor)
   train_data = np.array(train_sent_list, dtype=np.int32)
   test_data = np.array(test_sent_list, dtype=np.int32)
   train_tag = np.array(train_sent_tag, dtype=np.float32)
   test_tag = np.array(test_sent_tag, dtype=np.float32)
   
   # Training(Vocabulary)
   vocab_train = max(map(max,train_data))
   vocab_test = max(map(max,test_data))
   vocab = max(vocab_train, vocab_test) + 1

   # Triangle
   triangle = []
   for i in range(input_size+1):
      triangle.append([1 for x in range(i)] + [0 for x in range(input_size-i)])
   l_look = tf.constant(np.array(triangle, dtype=np.float32))
   
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

   # LSTM
   lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
   inputs = [word_vec_reshape[:,time_step,:] for time_step in range(input_size)]
   l = tf.placeholder(tf.int32, [batch_size])
   outputs, final_state = tf.nn.rnn(lstm_cell, inputs, dtype=tf.float32, sequence_length=l)
   output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

   # Dropout(Outputs)
   keep_prob_out = tf.placeholder(tf.float32)
   output_drop = tf.nn.dropout(output, keep_prob_out)
   
   # Weight & bias
   W = tf.Variable(tf.random_uniform([hidden_size, tag_size], -1.0, 1.0))
   b = tf.Variable(tf.random_uniform([tag_size], -1.0, 1.0))
   
   # Softmax
   y = tf.nn.softmax(tf.matmul(output_drop, W) + b)
   yreshape = tf.reshape(y, [batch_size, input_size, tag_size])
   
   # True Data
   y_ = tf.placeholder(tf.float32, [batch_size, input_size, tag_size])
   y_reshape = tf.reshape(y_, [-1, tag_size])
   
   # Closs Entropy
   # cross_entropy = -tf.reduce_sum(y_reshape*tf.log(y))
   cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(yreshape),
                                                reduction_indices=[2])
                                 * tf.nn.embedding_lookup(l_look, l)
                                 / tf.cast(tf.reduce_sum(l), dtype=tf.float32))

   # Gradient descent
#   train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
   train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

   # Model Initialize & Session Ranning
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   
   print ("---train start---")

   # Training
   for i in range(iterations):
      batch_xs, batch_ys, batch_len = _batch_random(train_data, train_tag, train_sent_len, batch_size)
      sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys, l:batch_len, keep_prob_wv:0.8, keep_prob_out:1.0})
      if (i+1)/10 == int((i+1)/10):
         print("count "+str(i+1), end = " ")
         sys.stdout.flush()
         if (i+1)/100 == int((i+1)/100):
            print()
            print(sess.run(cross_entropy, feed_dict={x:train_data[0:batch_size],
                                                     y_:train_tag[0:batch_size],
                                                     l:train_sent_len[0:batch_size],
                                                     keep_prob_wv:1.0,
                                                     keep_prob_out:1.0}))
   print ("---train finish---")

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

   # Output(Result for Prediction)
   tag_prediction = []
   for i in range(len(ylist)):
      for j in range(test_sent_len[i]):
         for key, value in tag_to_id.items():
            if ylist[i][j] == value:
               tag_prediction.append(key)
      tag_prediction.append("")

   # Output(Result for Prediction)
   iob2 = open(sys.argv[4], "w", encoding="utf-8")
   for i in range(len(tag_prediction)):
      if test_word_list[i] == "<eos>":
         iob2.write(" " + "\n")
      else:
         iob2.write(test_word_list[i] + "\t" + tag_prediction[i] + "\n")
      
   # Calculate Time
   end_time = time.time()
   print ("calc time: " + str(end_time - start_time))
