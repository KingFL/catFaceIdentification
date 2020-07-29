
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from PIL import Image
from flask import Flask,render_template,request,redirect,url_for,make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import random
import time

# 制作训练图片列表
class_names_to_ids = {'Caicai':0, 'Hulu':1, 'Dunge':2, 'Kaki':3, 'Samsara':4,
                        'Xiaohe':5, 'Lingdang':6, 'Douzi':7}
data_dir = './faceData/'
output_path = './DataSet/face_test/list.txt'
fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close()


# 生成随机测试集
# #
_NUM_VALIDATION = 100
_RANDOM_SEED = 0
list_path = './DataSet/face_test/list.txt'
train_list_path = './DataSet/face_test/list_train.txt'
val_list_path = './DataSet/face_test/list_val.txt'
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)
# fd = open(train_list_path, 'w')
# for line in lines[_NUM_VALIDATION:]:
#     fd.write(line)
# fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()
