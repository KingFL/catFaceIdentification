# coding=utf8

import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
import pymysql
from datetime import timedelta


# 允许设置的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

basepath = os.path.dirname(__file__)
# cascade_path = os.path.join(basepath, '/haarcascades/cashaarcascade_frontalcatface_extended.xml')
cascade_path = os.path.join(basepath, 'haarcascades/haarcascade_frontalcatface.xml')
save_dir = 'D://images//face'
model_dir = r'./DataSet/models/catfaces_train.models-6000'  # 模型地址

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
classes = ['Caicai', 'Hulu', 'Dunge', 'Kaki', 'Samsara', 'Xiaohe', 'Lingdang', 'Douz']  # 标签顺序
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
pred = tf.reshape(pred, shape=[-1, 10])
a = tf.argmax(pred, 1)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, model_dir)


@app.route('/id', methods=['POST', 'GET'])
def upload():
    if request.method == "GET":
        return render_template('upload.html')

    f = request.files['file']
    print(type(f))
    if not (f and allowed_file(f.filename)):
        return jsonify({"err": 1001, "msg": "清检查上传的图片类型，仅限于png、jpg、PNG、JPG、bmp, jpeg"})

    print("basepath: " + basepath)
    # upload_path = os.path.join(basepath, 'result', secure_filename(f.filename))
    fname = secure_filename(f.filename)
    ext = fname.rsplit('.', 1)[1]  # 获取文件后缀
    unix_time = int(time.time())
    # new_filename = str(unix_time) + '.' + ext  # 修改了上传的文件名
    new_filename = 'upload' + '.' + ext  # 修改了上传的文件名
    upload_path = os.path.join(save_dir, new_filename)  # 保存文件到upload目录
    f.save(upload_path)

    # 使用OpenCV自带的猫脸检测模型
    cascade = cv2.CascadeClassifier(cascade_path)
    # cascade.load()

    # read the pic of cat
    catimg = cv2.imread(upload_path)
    # transfer RGB to GRAY
    catimg_gray = cv2.cvtColor(catimg, cv2.COLOR_BGR2GRAY)
    # use the models to detect cat face
    catfaces = cascade.detectMultiScale(catimg_gray, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))

    if len(catfaces) > 0:
        for (i, (x, y, w, h)) in enumerate(catfaces):
            # mark the face at resource picture
            # cv2.rectangle(catimg, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
            # cut the part of cat face
            roiImg = catimg[y:y + h, x:x + w]
            # save the picture of cat face
            cur_file_cut = new_filename.split('.')
            cur_file_name = cur_file_cut[0]
            cur_file_ext = cur_file_cut[1]
            print(cur_file_cut)
            cv2.imwrite(save_dir + "/" + cur_file_name + "_" + str(i) + "." + cur_file_ext, roiImg)
            # cv2.imwrite(save_dir + "/" + cur_file + "_" + str(i), roiImg)
            print("save a face pic at " + save_dir + "/" + cur_file_name + "_" + str(i) + "." + cur_file_ext)
            # print the cat face
            # cv2.imshow('facedected', roiImg)
            # cv2.waitKey(0)
        # cv2.imwrite('cat.jpg', catimg)
    else:
        print('not found cat face, check the picture')

    image_path = os.path.join(save_dir, f'upload_0.{ext}')
    print(image_path)

    # 使用openCV转换一下图片的格式和名称
    image = cv2.imread(image_path)
    model_dir = r'./DataSet/models/catfaces_train.models-6000'  # 模型地址
    print("\n model_dir")
    print(model_dir)
    #  model_dir= "D:\\Development\\workspace\\catFaceIdentification\\DataSet\\6000 - 10000\\catfaces_train.models-6000"
    # x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # classes = ['Caicai', 'Hulu', 'Dunge', 'Kaki', 'Samsara', 'Xiaohe', 'Lingdang', 'Douz']  # 标签顺序
    # # pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
    # pred = tf.reshape(pred, shape=[-1, 10])
    # a = tf.argmax(pred, 1)
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver.restore(sess, model_dir)
    #     img = Image.open(upload_path)
    #     img = img.resize((224, 224))
    #     img = tf.reshape(img, [1, 224, 224, 3])
    #     img1 = tf.reshape(img, [1, 224, 224, 3])
    #     img = tf.cast(img, tf.float32) / 255.0
    #     b_image, b_image_raw = sess.run([img, img1])
    #     t_label = sess.run(a, feed_dict={x: b_image})
    #     print(t_label[0])
    #     predict = classes[t_label[0]]
    #     print(predict)
    # sess.run(tf.global_variables_initializer())
    img = Image.open(upload_path)
    img = img.resize((224, 224))
    img = tf.reshape(img, [1, 224, 224, 3])
    img1 = tf.reshape(img, [1, 224, 224, 3])
    img = tf.cast(img, tf.float32) / 255.0
    b_image, b_image_raw = Sess.run([img, img1])
    t_label = Sess.run(a, feed_dict={x: b_image})
    print(t_label[0])
    predict = classes[t_label[0]]
    print(predict)
    try:
        # 获取数据库连接
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd='qwer842655', db='facecat', port=3306, charset='utf8')
        # 打印数据库连接对象
        print('数据库连接对象为：{}'.format(conn))
        # 获取游标
        cur = conn.cursor()
        # 打印游标
        # print("游标为：{}".format(cur))
        # print("游标是："+cur) #
        print("游标是：" + str(cur))
        # 查询sql语句
        img_id = t_label[0]
        # str_list=['select* from connect.Plant where typeID=',img_id]
        # sql=''.join(str_list)

        sql = 'select* from facecat.cats where id='+str(img_id)
        # sql = 'select* from connect.Plant where typeID=0'
        print(sql)
        cur.execute(sql)
        conn.commit()
        # 使用 fetchall() 方法获取数据对象
        # data = cur.fetchall()
        # 使用 fetchone() 方法获取一条数据
        data = cur.fetchone()
        # for item in data:
        #     print(item)
        print(data[2])
        cur.close()
        conn.close()
    except Exception as e:
        print(e)
    name = data[1]
    join_date = data[2]
    # birthday = data[3]
    # if birthday == 'unknown':
    #     birthday = '暂时还不知道哦'

    # cv2.imwrite(os.path.join(basepath, 'static/Example_img/images', 'test.jpg'), image)
    return jsonify({"data": data})
    # return render_template('upload_ok.html', catname=name, joindate=join_date, picname=name)#val1=time.time())


if __name__ == '__main__':

    # model_dir = r'./DataSet/models/catfaces_train.models-6000'  # 模型地址
    # print("\n model_dir")
    # print(model_dir)
    # #  model_dir= "D:\\Development\\workspace\\catFaceIdentification\\DataSet\\6000 - 10000\\catfaces_train.models-6000"
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    classes = ['Caicai', 'Hulu', 'Dunge', 'Kaki', 'Samsara', 'Xiaohe', 'Lingdang', 'Douz']  # 标签顺序
    pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
    pred = tf.reshape(pred, shape=[-1, 10])
    a = tf.argmax(pred, 1)
    Sess=tf.Session()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_dir)
        Sess=sess
    app.run(host='127.0.0.1', port=20000, debug=True)