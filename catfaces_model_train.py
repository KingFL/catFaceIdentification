#coding = utf-8
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets


def read_and_decode_tfrecord(filename):
    filename_deque = tf.train.string_input_producer(filename)#转化为张量数据加入队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_deque)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)})
    #tf.cast转换数据类型
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    #print(label)
    #print(img.shape.as_list())

    img = tf.reshape(img, [ 224, 224, 3])
    img = tf.cast(img, tf.float32)
    img = tf.cast(img, tf.float32) / 255.0        #将矩阵归一化0-1之间
    #print(img.shape.as_list())
    return img, label


save_dir = r"./DataSet/models/catfaces_train.models"
# 批处理数
batch_size_ = 2
lr = tf.Variable(0.0001, dtype=tf.float32)#定义一个初始值为0.0001的变量
x = tf.placeholder(tf.float32, [None,224, 224, 3])#定义一个空间
y_ = tf.placeholder(tf.float32, [None])
#print(y)
#print(x)


# train_list = ['traindata_63.tfrecords-000', 'traindata_63.tfrecords-001', 'traindata_63.tfrecords-002',
#               'traindata_63.tfrecords-003', 'traindata_63.tfrecords-004', 'traindata_63.tfrecords-005',
#               'traindata_63.tfrecords-006', 'traindata_63.tfrecords-007', 'traindata_63.tfrecords-008',
#               'traindata_63.tfrecords-009', 'traindata_63.tfrecords-010', 'traindata_63.tfrecords-011',
#               'traindata_63.tfrecords-012', 'traindata_63.tfrecords-013', 'traindata_63.tfrecords-014',
#               'traindata_63.tfrecords-015', 'traindata_63.tfrecords-016', 'traindata_63.tfrecords-017',
#               'traindata_63.tfrecords-018', 'traindata_63.tfrecords-019', 'traindata_63.tfrecords-020',
#               'traindata_63.tfrecords-021']
# 随机打乱顺序
# train_list = ['traindata_63.tfrecords-000','traindata_63.tfrecords-001']

train_list = ['./DataSet/models/catfaces_train.tfrecords']
img, label = read_and_decode_tfrecord(train_list)
# img, label = read_and_decode_tfrecord(train_list)
img_batch, label_batch = tf.train.shuffle_batch([img, label], num_threads=2, batch_size=batch_size_, capacity=10000, min_after_dequeue=9900)
# capacity队列容量

# 将label值进行onehot编码
one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=10)
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
#pred, end_points = nets.resnet_v2.resnet_v2_101(x, num_classes=10, is_training=True)
pred = tf.reshape(pred, shape=[-1, 10])
#print('pred:%s'%pred)
# 定义损失函数和优化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=one_hot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 准确度
a = tf.argmax(pred, 1)
b = tf.argmax(one_hot_labels, 1)
correct_pred = tf.equal(a, b)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))) as sess:
    sess.run(tf.global_variables_initializer())
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner,此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    while True:
        i += 1
        b_image, b_label = sess.run([img_batch, label_batch])
        #print('b_label:%s'%b_label)
        #print('b_image:%s'%b_image)
        #bb=[None,b_image]
        #print('bbbbbbbbbbbbbbbbbbbb:%s'%bb)

        _, loss_, y_t, y_p, a_, b_ = sess.run([optimizer, loss, one_hot_labels, pred, a, b], feed_dict={x: b_image,
                                                                                                        y_: b_label})
        print('step: {}, train_loss: {}'.format(i, loss_))
        if i % 100 == 0:
            _loss, acc_train = sess.run([loss, accuracy], feed_dict={x: b_image, y_: b_label})
            print('--------------------------------------------------------')
            print('step: {}  train_acc: {}  loss: {}'.format(i, acc_train, _loss))
            print('--------------------------------------------------------')
            if i % 1000 == 0:
                saver.save(sess, save_dir, global_step=i)
                print("saved a model when trained {} steps".format(i))
                print("--------------------------------------------------------")
            # saver.save(sess, save_dir, global_step=i)
        if i == 6000:
            print("finished")
            break

    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
