import pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf


start = time.time()

doc_train_data = None
user_train_data = None

# load test data for now
user_test_data = pickle.load(open('data/test_user.pkl', 'rb')).tocsr()
doc_test_data = pickle.load(open('data/test_doc.pkl', 'rb')).tocsr()
test_lines = doc_test_data.shape[0]
def load_train_data():
    global doc_train_data, user_train_data
    doc_train_data = None
    user_train_data = None
    start = time.time()
    user_train_data = pickle.load(open('data/train_user.pkl', 'rb')).tocsr()
    doc_train_data = pickle.load(open('data/train_doc.pkl', 'rb')).tocsr()
    end = time.time()
    print ("\nTrain data is loaded in %.2fs" % ( end - start))
    return pickle.load(open('data/train_user.pkl', 'rb')).tocsr().shape[0]


TRIGRAM_DOC = pickle.load(open('data/train_doc.pkl', 'rb')).tocsr().shape[1]
TRIGRAM_USER = pickle.load(open('data/train_user.pkl', 'rb')).tocsr().shape[1]

NEG = 3
BS = 1000
training_epochs = 10
learning_rate = 0.1
L1_N = 512
L2_N = 256
L3_N = 128

with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    user_batch = tf.sparse_placeholder(tf.float32, name='userBatch')
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, name='DocBatch')

with tf.name_scope('L1'):
    l1_par_range_doc = np.sqrt(6.0 / (TRIGRAM_DOC + L1_N))
    l1_par_range_user = np.sqrt(6.0 / (TRIGRAM_USER + L1_N))
    weight1_doc = tf.Variable(tf.random_uniform([TRIGRAM_DOC, L1_N], -l1_par_range_doc, l1_par_range_doc))
    weight1_user = tf.Variable(tf.random_uniform([TRIGRAM_USER, L1_N], -l1_par_range_user, l1_par_range_user))
    bias1_doc = tf.Variable(tf.random_uniform([L1_N], -l1_par_range_doc, l1_par_range_doc))
    bias1_user = tf.Variable(tf.random_uniform([L1_N], -l1_par_range_user, l1_par_range_user))

    user_l1 = tf.sparse_tensor_dense_matmul(user_batch, weight1_user) + bias1_user
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1_doc) + bias1_doc

    user_l1_out = tf.nn.relu(user_l1)
    doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('L2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2_user = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    weight2_doc = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2_user = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    bias2_doc = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))

    user_l2 = tf.matmul(user_l1_out, weight2_user) + bias2_user
    doc_l2 = tf.matmul(doc_l1_out, weight2_doc) + bias2_doc
    user_l2_out = tf.nn.relu(user_l2)
    doc_l2_out = tf.nn.relu(doc_l2)

with tf.name_scope('L3'):
    l3_par_range = np.sqrt(6.0 / (L2_N + L3_N))

    weight3_user = tf.Variable(tf.random_uniform([L2_N, L3_N], -l3_par_range, l3_par_range))
    weight3_doc = tf.Variable(tf.random_uniform([L2_N, L3_N], -l3_par_range, l3_par_range))
    bias3_user = tf.Variable(tf.random_uniform([L3_N], -l3_par_range, l3_par_range))
    bias3_doc = tf.Variable(tf.random_uniform([L3_N], -l3_par_range, l3_par_range))

    user_l3 = tf.matmul(user_l2_out, weight3_user) + bias3_user
    doc_l3 = tf.matmul(doc_l2_out, weight3_doc) + bias3_doc
    user_y = tf.nn.relu(user_l3)
    doc_y = tf.nn.relu(doc_l3)

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce NEG FD-
    temp = tf.tile(doc_y, [1, 1])
    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat([doc_y,
                           tf.slice(temp, [rand, 0], [BS - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])], 0)
with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    user_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(user_y), 1, True)), [NEG + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))
    prod = tf.reduce_sum(tf.multiply(tf.tile(user_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(user_norm, doc_norm)
    cos_sim_raw = tf.truediv(prod, norm_prod)
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    prob = tf.nn.softmax((cos_sim))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
    tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)


def pull_batch(user_data, doc_data, batch_idx):
    begin = batch_idx * BS
    end = begin + BS
    #BS_NOW = end-begin
    user_in = user_data[begin:end, :]
    doc_in = doc_data[begin:end, :]
    user_in = user_in.tocoo()
    doc_in = doc_in.tocoo()

    user_in = tf.SparseTensorValue(
        np.transpose([np.array(user_in.row, dtype=np.int64), np.array(user_in.col, dtype=np.int64)]),
        np.array(user_in.data, dtype=np.float),
        np.array(user_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array(doc_in.shape, dtype=np.int64))

    return user_in, doc_in


def feed_dict(Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        user_in, doc_in = pull_batch(user_train_data, doc_train_data, batch_idx)
    else:
        user_in, doc_in = pull_batch(user_test_data, doc_test_data, batch_idx)
    return {user_batch: user_in, doc_batch: doc_in}


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
saver = tf.train.Saver()
checkpoint_steps = 2
checkpoint_dir = ''
is_train = True
is_test = True
test_res = []
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if is_train:
        train_lines = load_train_data()
        for epoch in range(training_epochs):
            start = time.time()
            epoch_loss = 0
            for batch_idx in range(train_lines/BS):
                _, loss_batch = sess.run([train_step,loss], feed_dict=feed_dict(True, batch_idx))
                epoch_loss += loss_batch
            epoch_loss /= train_lines/BS
            end = time.time()
            print ("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                    (epoch, epoch_loss, end - start))
            if (epoch + 1) % checkpoint_steps == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch+1)
    if is_test:
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass
        test_loss = 0
        test_steps = test_lines/BS
        for batch_idx in range(test_steps):
            sim, loss_v = sess.run([hit_prob, loss], feed_dict=feed_dict(False, batch_idx))
            test_loss += loss_v
            test_res.append(sim.tolist())
            #print sim
        with open('data/test_res','wb') as fw:
            for i in range(len(test_res)):
                for j in range(len(test_res[i])):
                    fw.write(str(test_res[i][j][0]))
                    fw.write('\n')
        test_loss /= test_steps
        print ("\nTest Loss: %-4.3f" % test_loss)
