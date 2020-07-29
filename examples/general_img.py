import os
from keras_preprocessing.image import ImageDataGenerator,img_to_array,load_img

data_gen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            fill_mode='nearest')

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("/")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
         # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def list_dir(file_dir,save_dir):
    '''
    通过 listdir 得到的是仅当前路径下的文件名，不包括子目录中的文件，如果需要得到所有文件需要递归
    '''
    print('\n\n<><><><><><> listdir <><><><><><>')
    print ('current dir : {0}'.format(file_dir))
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        # 获取文件的绝对路径
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path):# 判断是否是文件还是目录需要用绝对路径
            img = load_img(path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            n = 1
            for batch in data_gen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='train',
                                       save_format='jpg'):
                n += 1
                if n > 6:
                    break
            print ("{0} : is file!".format(cur_file))
        if os.path.isdir(path):
            print ("{0} : is dir!".format(cur_file))
            #list_dir(path) # 递归子目录


rootdir = "./Data_CatFace"  # 指明被遍历的文件夹
savedir = "./DataSet/train_images" #保存预处理后图片

rootdir_list=os.listdir(rootdir)
for dirname in rootdir_list:
    parent_dir=os.path.join(rootdir, dirname)
    save_dir = os.path.join(savedir, dirname)
    mkdir(save_dir)
    print(parent_dir)
    list_dir(parent_dir,save_dir)

