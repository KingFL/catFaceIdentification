
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from PIL import Image
from flask import Flask,render_template,request,redirect,url_for,make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2


# 制作训练图片列表
# class_names_to_ids = {'Caicai':0, 'Kaki':1, 'Hulu':2, 'Samsara':3, 'Dunge':4,
#                         'Xiaohe':5, 'Douzi':6, 'Lingdang':7}
# data_dir = './train_images/'
# output_path = 'list.txt'
# fd = open(output_path, 'w')
# for class_name in class_names_to_ids.keys():
#     images_list = os.listdir(data_dir + class_name)
#     for image_name in images_list:
#         fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
# fd.close()


# 生成随机测试集
# #
# import random
# _NUM_VALIDATION = 350
# _RANDOM_SEED = 0
# list_path = 'list.txt'
# train_list_path = 'list_train.txt'
# val_list_path = 'list_val.txt'
# fd = open(list_path)
# lines = fd.readlines()
# fd.close()
# random.seed(_RANDOM_SEED)
# random.shuffle(lines)
# # fd = open(train_list_path, 'w')
# # for line in lines[_NUM_VALIDATION:]:
# #     fd.write(line)
# # fd.close()
# fd = open(val_list_path, 'w')
# for line in lines[:_NUM_VALIDATION]:
#     fd.write(line)
# fd.close()


#测试识别率
list_path='list_val.txt'
fd = open(list_path)
lines = [line.split() for line in fd]
fd.close()

num_true=0
num_flase=0

model_dir = r'../models/train_flower.models-12000'  # 模型地址
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
#pred, end_points = nets.resnet_v2.resnet_v2_101(x,num_classes=10,is_training=False)

pred = tf.reshape(pred, shape=[-1, 10])
a = tf.argmax(pred, 1)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_dir)
    for li in lines:
        ima_path='./train_images/'+li[0]
        print(ima_path)
        img = Image.open(ima_path)
        img = img.resize((224, 224))
        img = tf.reshape(img, [1, 224, 224, 3])
        img1 = tf.reshape(img, [1, 224, 224, 3])
        img = tf.cast(img, tf.float32) / 255.0
        #img = tf.cast(img, tf.float32)
        b_image, b_image_raw = sess.run([img, img1])
        t_label = sess.run(a, feed_dict={x: b_image})
        print(t_label[0])
        if str(t_label[0])==str(li[1]):
            num_true=num_true+1
        else:
            num_flase=num_flase+1
        print(num_flase)
        print(num_true)

result=num_true/(num_flase+num_true)
print("识别率："+str(result))