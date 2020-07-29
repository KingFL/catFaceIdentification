import tensorflow as tf
import cv2
import time

print(tf.__version__)

cascade_path = 'cashaarcascade_frontalcatface_extended.xml'
cascade = cv2.CascadeClassifier(cascade_path)
#cascade.load('/root/PycharmProjects/untitled/cashaarcascade_frontalcatface_extended.xml')
#cascade.load('./haarcascades/cashaarcascade_frontalcatface_extended.xml')
cascade.load('/root/PycharmProjects/untitled/opencv-master/data/haarcascades/haarcascade_frontalcatface.xml')


catpic = '/root/PycharmProjects/untitled/Data_CatFace/Caicai/Caicai_3.jpg'
img = cv2.imread(catpic)
#print(len(img))
# cv2.imshow('catpic', img)
# cv2.waitKey(0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', img_gray)
# cv2.waitKey(0)
catfaces = cascade.detectMultiScale(img_gray, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))

print(len(catfaces))
if len(catfaces) > 0:
    print('cat face detected')
    for(i, (x, y, w, h)) in enumerate(catfaces):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
        roiImg = img[y:y + h, x:x + w]
        cv2.imwrite('face.jpg', roiImg)
        cv2.imshow('facedected', roiImg)
        cv2.waitKey(0)
    cv2.imwrite('cat.jpg', img)

time.sleep(10)