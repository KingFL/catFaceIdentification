import os
import cv2
import time


def factCutter(path):
    cascade_path = 'cashaarcascade_frontalcatface_extended.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    cascade.load(
        '/root/PycharmProjects/untitled/opencv-master/data/haarcascades/haarcascade_frontalcatface_extended.xml')

    catimg = cv2.imread(path)
    catimg_gray = cv2.cvtColor(catimg, cv2.COLOR_BGR2GRAY)
    catfaces = cascade.detectMultiScale(catimg_gray, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))

    print('found '+str(len(catfaces))+' cat faces at' + format(path))
    return catfaces


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("/")
    path = path.rstrip("\\")
    # 判断路径是否存在
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        return False


def list_dir(file_dir, save_dir):
    print('\n------ listdir ------')
    print('current dir : {0}'.format(file_dir))
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
            # faces = factCutter(path)

            # 使用OpenCV自带的猫脸识别模型
            cascade_path = './haarcascades/cashaarcascade_frontalcatface_extended.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            cascade.load(
                '/root/PycharmProjects/catFaceIdentification/haarcascades/haarcascade_frontalcatface.xml')

            # read the pic of cat
            catimg = cv2.imread(path)
            # transfer RGB to GRAY
            catimg_gray = cv2.cvtColor(catimg, cv2.COLOR_BGR2GRAY)
            # use the models to detect cat face
            catfaces = cascade.detectMultiScale(catimg_gray, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))
            print('found ' + str(len(catfaces)) + ' cat faces at' + format(path))

            print("cur_file is   " + cur_file)
            print("save_dir   " + save_dir)
            print("file_dir   " + file_dir)
            #time.sleep(10)

            if len(catfaces) > 0:
                for (i, (x, y, w, h)) in enumerate(catfaces):
                    # mark the face at resource picture
                    # cv2.rectangle(catimg, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
                    # cut the part of cat face
                    roiImg = catimg[y:y + h, x:x + w]
                    # save the picture of cat face
                    cur_file_cut = cur_file.split('.')
                    cur_file_name = cur_file_cut[0]
                    cur_file_ext = cur_file_cut[1]
                    print(cur_file_cut)
                    cv2.imwrite(save_dir + "/" + cur_file_name + "_" + str(i) + "." + cur_file_ext, roiImg)
                    #cv2.imwrite(save_dir + "/" + cur_file + "_" + str(i), roiImg)
                    print("save a face pic at " + save_dir + "/" + cur_file_name + "_" + str(i) + "." + cur_file_ext)
                    # print the cat face
                    # cv2.imshow('facedected', roiImg)
                    # cv2.waitKey(0)
                # cv2.imwrite('cat.jpg', catimg)
            else:
                print(path + ' not found cat face, check the picture')

rootdir = "./Data_CatFace"
savedir = "./faceData"

# mkdir("faceData")
rootdir_list=os.listdir(rootdir)
for dirname in rootdir_list:
    parent_dir = os.path.join(rootdir, dirname)
    save_dir = os.path.join(savedir, dirname)
    mkdir(save_dir)
    print(parent_dir)
    list_dir(parent_dir, save_dir)