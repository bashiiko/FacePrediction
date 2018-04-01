import cv2
import os

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_smile.xml')

# 元の画像データが格納されたフォルダ
input_dir = "C:/Users/naoko/Desktop/OpenCV_tutorial/human_data/"
# テスト用のの画像データが格納されたフォルダ
test_input_dir = "C:/Users/naoko/Desktop/OpenCV_tutorial/test_data/"
# 顔を切り出して保存するフォルダ
SAVE_PATH = "C:/Users/naoko/Desktop/OpenCV_tutorial/face_data/"
# 顔を切り出して保存するフォルダ
TEST_SAVE_PATH = "C:/Users/naoko/Desktop/OpenCV_tutorial/test/"
# 検知できた顔の数
#detect_count = 0

def face_detect(path,savepath):
    for f in os.listdir(path):
        # イメージファイルの読み込み
        img = cv2.imread(path + f)
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 顔を検知
        faces = face_cascade.detectMultiScale(gray)

        # 顔が検知出来た場合、トリミングして保存
        if len(faces) > 0:
                for (x,y,w,h) in faces:
                     #cv2.imwrite(SAVE_PATH + "face_data_" + \
                     #             "{0:03d}".format(detect_count) + ".jpg", \
                     #             img[y:y+h, x:x+w])
                     cv2.imwrite(savepath + f, img[y:y+h, x:x+w])
                     #detect_count+=1
            # 検知した顔を矩形で囲む
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # 顔画像（グレースケール）y座標、x座標
            #roi_gray = gray[y:y+h, x:x+w]
            # 顔画像（カラースケール）
            #roi_color = img[y:y+h, x:x+w]
            # 顔の中から目を検知
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
                # 検知した目を矩形で囲む
                #eye_space = img[ey:ey+eh, ex:ex+ew]
                #double_size = cv2.resize(eye_space,(ew*2,eh*2))
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#smile = smile_cascade.detectMultiScale(gray)
#for (sx,sy,sw,sh) in smile:
    #cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
# 画像表示
#cv2.imshow('detecting result',img)

# 画像保存
#cv2.imwrite("result4.jpg", img)

face_detect(input_dir,SAVE_PATH)
face_detect(test_input_dir,TEST_SAVE_PATH)
# 何かキーを押したら終了
cv2.waitKey(0)
cv2.destroyAllWindows()
