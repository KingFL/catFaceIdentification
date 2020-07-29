# coding=utf8
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from PIL import Image
from flask import Flask,render_template,request,redirect,url_for,make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
import pymysql

from datetime import timedelta

#允许设置的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
#设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

#每一个页面对应一个网址，每一个网址对应一个函数
@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f=request.files['file']
        print(type(f))
        if not (f and allowed_file(f.filename)):
            return jsonify({"err": 1001, "msg": "清检查上传的图片类型，仅限于png、jpg、PNG、JPG、bmp, jpeg"})

        # user_input = request.form.get("name")
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'result', secure_filename(f.filename))
        print(upload_path)
        f.save(upload_path)

        #使用opencv转换一下图片的格式和名称
        image= cv2.imread(upload_path)

        model_dir = r'./models/train_flower.models-12000'  # 模型地址
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        classes = ['Caicai', 'Kaki', 'Hulu', 'Samsara', 'Dunge', 'Xiaohe', 'Douzi', 'Lingdang']  # 标签顺序
        pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
        pred = tf.reshape(pred, shape=[-1, 10])
        a = tf.argmax(pred, 1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_dir)
            img = Image.open(upload_path)
            img = img.resize((224, 224))
            img = tf.reshape(img, [1, 224, 224, 3])
            img1 = tf.reshape(img, [1, 224, 224, 3])
            img = tf.cast(img, tf.float32) / 255.0
            b_image, b_image_raw = sess.run([img, img1])
            t_label = sess.run(a, feed_dict={x: b_image})
            print(t_label[0])
            predict = classes[t_label[0]]
            print(predict)

        try:
            # 获取数据库连接
            conn = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='connect', port=3306,
                                   charset='utf8')
            # 打印数据库连接对象
            print('数据库连接对象为：{}'.format(conn))
            # 获取游标
            cur = conn.cursor()
            # 打印游标
            # print("游标为：{}".format(cur))
            # print("游标是："+cur) #
            print("游标是：" + str(cur))
            # 查询sql语句
            img_id=t_label[0]
            # str_list=['select* from connect.Plant where typeID=',img_id]
            # sql=''.join(str_list)

            sql = 'select* from connect.Plant where typeID='+str(img_id)
            #sql = 'select* from connect.Plant where typeID=0'
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
        user_input = data[1]
        describe=data[2]
        cv2.imwrite(os.path.join(basepath,'static/images','test.jpg'),image)
        return render_template('upload_ok.html',userinput=user_input,describe=describe,val1=time.time())

    return render_template('upload.html')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8987, debug=True)

