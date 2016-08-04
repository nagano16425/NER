"""
Title:Tensorflow
Detail:F measure
Design:Naonori Nagano
Date:2016/08/04
"""

# Import Python Library
import collections,time,sys,os,random
# Import TensorFlow
import tensorflow as tf
import numpy as np

# Read Words
def  _read_words(filename):
   f = tf.gfile.GFile(filename, "r")
   word = []
   chunk = []
   NER = []
   for line in f.readlines():
      if len(line)>3:
         pair = line.strip().split("\t")
         pair_extend = pair[1].split("-")
         word.append(pair[0])
         chunk.append(pair_extend[0])
         if pair_extend[0] == "B" or pair_extend[0] == "I":
            NER.append(pair_extend[1])
         else:
            NER.append("<eos>")
      else:
         word.append("<eos>")
         chunk.append("<eos>")
         NER.append("<eos>")
   return word, chunk, NER

# Build Vocabulary
def _build_vocab(wordlist):
   counter = collections.Counter(wordlist)
   count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
   words, _ = list(zip(*count_pairs))
   word_to_id = dict(zip(words, range(len(words))))
   return word_to_id

# Chunk change to array from BIO
def _chunk_change(chunk):
   chunk_data = [0 for i in range(4)]
   if chunk == "B":
      chunk_data[0] = 1
   elif chunk == "I":
      chunk_data[1] = 1
   elif chunk == "O":
      chunk_data[2] = 1
   else:
      chunk_data[3] = 1
   return chunk_data

# Making sentence list for word embedding
def _sentence_list(word_list, chunk_list, word_to_id):
   sentence_list = []
   sent_chunk = []
   sentence = []
   chunk = []
   sent_len = []
   for i in range(len(word_list)):
      if word_list[i] != "<eos>":
         if word_list[i] in word_to_id:
            sentence.append([word_to_id[word_list[i]]])
            chunk.append(_chunk_change(chunk_list[i]))
         else:
            sentence.append([len(word_to_id)+1])
            chunk.append(_chunk_change(chunk_list[i]))
      else:
         sent_len.append(len(sentence))
         for k in range(input_size):
            if len(sentence) < input_size:
               sentence.append([word_to_id["<eos>"]])
               chunk.append(_chunk_change("<eos>"))
            else:
               break   
         sentence_list.append(sentence)
         sent_chunk.append(chunk)
         sentence = []
         chunk = []
   sent_len = np.array(sent_len)
   return sentence_list, sent_chunk, sent_len

# Random number generation
def _batch_random(sentence_list, sentence_chunk,sent_len, batch_size = 1):
   random_list = []
   for i in range(len(sentence_list)):
      random_list.append(i)
   random.shuffle(random_list)
   return (np.array([sentence_list[ random_list[x] ] for x in range(batch_size)], dtype=np.int32),
           np.array([sentence_chunk[ random_list[x] ] for x in range(batch_size)]),
           np.array([sent_len[ random_list[x] ] for x in range(batch_size)], dtype=np.int32))

if __name__ == "__main__":
   random.seed(0)
   tf.set_random_seed(0)
   start_time = time.time()

   # To define the size
   input_size = 183
   batch_size = 10
   hidden_size = 150
   embed_size = 10
   iterations = int(sys.argv[3])
   
   print("---read start---")
   data_path = "/home/nagano.16425/tensorflow/datasets/"
   train_path = os.path.join(data_path, sys.argv[1])
   test_path = os.path.join(data_path, sys.argv[2])
   train_word_list, train_chunk_list, train_NER_list = _read_words(train_path)
   test_word_list, test_chunk_list, test_NER_list = _read_words(test_path)
   word_to_id = _build_vocab(train_word_list)
   
   train_sent_list, train_sent_chunk, train_sent_len = _sentence_list(train_word_list, train_chunk_list, word_to_id)
   test_sent_list, test_sent_chunk, test_sent_len = _sentence_list(test_word_list, test_chunk_list, word_to_id)
   
   # Input array(Tensor)
   train_data = np.array(train_sent_list, dtype=np.int32)
   test_data = np.array(test_sent_list, dtype=np.int32)
   train_chunk = np.array(train_sent_chunk, dtype=np.float32)
   test_chunk = np.array(test_sent_chunk, dtype=np.float32)
   
   vocab_train = max(map(max,train_data))[0]
   vocab_test = max(map(max,test_data))[0]
   vocab = max(vocab_train, vocab_test) + 1

   triangle = []
   for i in range(input_size+1):
      triangle.append([1 for x in range(i)] + [0 for x in range(input_size-i)])
   l_look = tf.constant(np.array(triangle, dtype=np.float32))
   
   print("---read finish---")

   # Word_embedding
   x = tf.placeholder(tf.int32, [batch_size, input_size, 1])
   Wembed = tf.Variable(tf.random_uniform([vocab, embed_size], -1.0, 1.0))

   # Reshape word vector
   word_vec = tf.nn.embedding_lookup(Wembed, x)
   word_vec_reshape = tf.reshape(word_vec, [batch_size, input_size, embed_size])

   # LSTM
   lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0)
   inputs = [word_vec_reshape[:,time_step,:] for time_step in range(input_size)]
   l = tf.placeholder(tf.int32, [batch_size])
   outputs, final_state = tf.nn.rnn(lstm_cell, inputs, dtype=tf.float32, sequence_length=l)
   output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

   # Output Layer
   W = tf.Variable(tf.random_uniform([hidden_size, 4], -1.0, 1.0))
   b = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
   y = tf.nn.softmax(tf.matmul(output, W) + b)
   yreshape = tf.reshape(y, [batch_size, input_size, 4])
   
   # Closs Entropy
   y_ = tf.placeholder(tf.float32, [batch_size, input_size, 4])
   y_reshape = tf.reshape(y_, [-1, 4])
   # cross_entropy = -tf.reduce_sum(y_reshape*tf.log(y))
   cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(yreshape),
                                                reduction_indices=[2])
                                 * tf.nn.embedding_lookup(l_look, l)
                                 / tf.cast(tf.reduce_sum(l), dtype=tf.float32))
   
   # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
   train_step = tf.train.MomentumOptimizer(0.01, 0.95).minimize(cross_entropy)
   
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   
   print ("---start train---")
   for i in range(iterations):
      batch_xs, batch_ys, batch_len = _batch_random(train_data, train_chunk, train_sent_len, batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys, l:batch_len})
      if (i+1)/10 == int((i+1)/10):
         print("count "+str(i+1), end = " ")
         sys.stdout.flush()
         if (i+1)/100 == int((i+1)/100):
            print()
            print(sess.run(cross_entropy, feed_dict={x:train_data[0:batch_size],
                                                     y_:train_chunk[0:batch_size],
                                                     l:train_sent_len[0:batch_size]}))
            
   print ("---train finish---")
   exc = tf.placeholder(tf.int32, [batch_size])
   correct_prediction = tf.equal(tf.argmax(yreshape,2), tf.argmax(y_,2))
   accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float")
                            * tf.nn.embedding_lookup(l_look,l)) / tf.cast(tf.reduce_sum(l), "float")
   exc_acc = tf.reduce_sum(tf.cast(correct_prediction, "float")
                           * tf.nn.embedding_lookup(l_look,l)
                           * tf.nn.embedding_lookup(l_look,exc))/ tf.cast((tf.reduce_sum(l*exc)
                                                                          / input_size), "float")
   
   sum_acc = 0
   ylist = []
   y_list = []
   for i in range(0,len(test_data),batch_size):
      if len(test_data[i:i+batch_size]) == batch_size:
         sum_acc += sess.run(accuracy, feed_dict={x: test_data[i:i+batch_size],
                                                  y_: test_chunk[i:i+batch_size],
                                                  l: test_sent_len[i:i+batch_size]})
         ylist.extend(sess.run(tf.argmax(yreshape,2),
                               feed_dict={x: test_data[i:i+batch_size],
                                          y_: test_chunk[i:i+batch_size],
                                          l: test_sent_len[i:i+batch_size]}))
         y_list.extend(sess.run(tf.argmax(y_,2),
                                feed_dict={x: test_data[i:i+batch_size],
                                           y_: test_chunk[i:i+batch_size],
                                           l: test_sent_len[i:i+batch_size]}))
      else:
         amari = (len(test_data) % batch_size)
         sum_acc += sess.run(exc_acc, feed_dict={x: test_data[-batch_size:],
                                                 y_: test_chunk[-batch_size:],
                                                 l: test_sent_len[-batch_size:],
                                                 exc: np.array([0 for x in range(batch_size - amari)]
                                                               + [input_size for x in range(amari)],
                                                               dtype=np.int32)})
   # Accuracy
   print("accuracy: ",end="")
   print(sum_acc/(int(len(test_data)/batch_size)+amari))

   # Calculate Time
   end_time = time.time()
   print ("calc time: " + str(end_time - start_time))

   print ("---F-Measure start---")

   # Counting(True Positive + False Positive)
   b_think = 0
   for i in range(len(ylist)):
      for k in range(test_sent_len[i]):
         if ylist[i][k] == 0:
            b_think += 1

   # Counting(True Positive + False Negative)
   b_truth = 0
   for i in range(len(y_list)):
      for k in range(test_sent_len[i]):
         if y_list[i][k] == 0:
            b_truth += 1

   # Counting(correct)
   correct = 0
   for i in range(len(ylist)):
      for k in range(test_sent_len[i]):
         if ylist[i][k] == y_list[i][k] == 0 and (test_sent_len[i]-1) > k:
            for l in range(test_sent_len[i]-k-1):
               if ylist[i][k+l+1] != y_list[i][k+l+1]:
                  break
               elif ylist[i][k+l+1] == y_list[i][k+l+1] == 0:
                  correct += 1
                  break
               elif ylist[i][k+l+1] == y_list[i][k+l+1] == 2:
                  correct += 1
                  break
         elif ylist[i][k] == y_list[i][k] == 0 and k == (test_sent_len[i]-1):
            correct += 1

   # Calculating F-measure
   precision = correct / b_think
   recall = correct / b_truth
   f_measure = 2 * precision * recall / (precision+recall)

   # Output F-measure
   print("think " +str(b_think), "truth " +str(b_truth), "correct " +str(correct))
   print("Precision " +str(precision), "Recall " +str(recall))
   print("F-measure " +str(f_measure))
   print ("---F-Measure finish---")
