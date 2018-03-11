import numpy as np
import pickle
from numpy import *
from tensorflow.contrib.data import Iterator, Dataset
import threading
from PIL import Image as pil_image
import random


train_filename = 'train.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()


test_filename = 'test.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Test data: {}/{}'.format(i, len(test_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(test_addrs[i])
    label = test_labels[i]
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()


def load_image(addr):
    image = pil_image.open(addr)
    image = np.asarray(image.resize((227, 227), pil_image.ANTIALIAS), dtype=np.float32)
    return image


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_tfrecords_file(input_queue, feature, type):
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(input_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features[type + 'image'], tf.float32)
    label = tf.cast(features[type + 'label'], tf.int32)
    label = tf.one_hot(label, label_count)
    image = tf.reshape(image, [227, 227, 3])
    return image, label


def run_model(train_path, test_path, iter):
    nb_epoch = 40
    graph = tf.Graph()
    with graph.as_default():
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(train_path, num_epochs=nb_epoch)
        images, labels = tf.train.shuffle_batch(read_tfrecords_file(filename_queue, feature, 'train/'), 
            batch_size=batch_size, 
            capacity=5000,
            min_after_dequeue=1000,
            allow_smaller_final_batch=True)

        queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32, tf.float32], shapes=[[227, 227, 3], [label_count]])
        enqueue_op = queue.enqueue_many([images, labels])
        dequeue_op = queue.dequeue()
        data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=500)

        feature_test = {'test/image': tf.FixedLenFeature([], tf.string),
                'test/label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue_test = tf.train.string_input_producer(test_path, num_epochs=1)
        data_batch_test, label_batch_test = tf.train.batch(read_tfrecords_file(filename_queue_test, feature_test, 'test/'), batch_size=10)

        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc8W = weight_variable([4096, label_count], 'W_fc8')
        fc8b = bias_variable([label_count], 'b_fc8')
        keep_prob = tf.placeholder('float')

        def enqueue(session):
            while True:
                session.run([images, labels, enqueue_op])

        def model(x):
            conv1 = tf.nn.relu(conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1))
            lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=0.00002, beta=0.75, bias=1.0)
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))
            lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=0.00002, beta=0.75, bias=1.0)
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            conv3 = tf.nn.relu(conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
            conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
            conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
            fc6_drop = tf.nn.dropout(fc6, keep_prob)
            fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b)
            fc7_drop = tf.nn.dropout(fc7, keep_prob)
            fc8 = tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b)
            return fc8

        logits = model(data_batch)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch))
        regularizers = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = tf.reduce_mean(cross_entropy + WEIGHT_DECAY * regularizers)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.01, global_step, 1000, 0.65, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits),1), tf.argmax(label_batch,1))
        count_correct_prediction = tf.reduce_sum(tf.cast(correct_prediction, 'float'))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        
        logits_test = model(data_batch_test)
        test_labels = tf.argmax(label_batch_test, 1)
        prediction_vector_score = tf.nn.softmax(logits_test)
        count_correct_prediction_test = tf.reduce_sum(tf.cast(
            tf.equal(tf.argmax(prediction_vector_score,1), test_labels), 'float'))
        count_top_5_correct_prediction = tf.reduce_sum(tf.cast(
            tf.nn.in_top_k(prediction_vector_score, test_labels, k=5), 'float'))

        saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        session.run(tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer()))

        enqueue_thread = threading.Thread(target=enqueue, args=[session])
        enqueue_thread.isDaemon()
        enqueue_thread.start()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            step = 0
            while True:
                run_options = tf.RunOptions(timeout_in_ms=40000)
                batch_res = session.run([ train_step, loss, accuracy ],
                    feed_dict = { keep_prob: 0.5 })

                print step, batch_res[1:]
                step += 1
        except tf.errors.OutOfRangeError:
              print('Done training for %d epochs, %d steps.' % (nb_epoch, step))
                
        test_ret = 0
        test_ret_top5 = 0
        test_count = 0
        try:
            while True:
                ret = session.run([count_correct_prediction_test, count_top_5_correct_prediction, prediction_vector_score, test_labels], 
                    feed_dict = { keep_prob: 1.0})
                test_ret = test_ret + ret[0]
                test_ret_top5 = test_ret_top5 + ret[1]
                test_count = test_count + 10 
        except tf.errors.OutOfRangeError:
            print('Done testing')
        
        print test_ret, test_count, float(test_ret) / test_count, float(test_ret_top5) / test_count
    
        session.run(queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        session.close()