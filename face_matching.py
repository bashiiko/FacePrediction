# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

# 顔認識器の構築 for OpenCV 3
# EigenFace
#recognizer = cv2.face.createEigenFaceRecognizer()
# FisherFace
#recognizer = cv2.face.createFisherFaceRecognizer()
# LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

train_dir = "C:/Users/naoko/Desktop/OpenCV_tutorial/face_data/"
test_dir = "C:/Users/naoko/Desktop/OpenCV_tutorial/test/"

# 名前からid(番号)を与える
def Give_label(name):
    id = 1
    if name == "tetuya":
        id = 1
    elif name == "takuya":
        id = 2
    elif name == "turube":
        id = 3
    elif name == "murooo":
        id = 4

    return id


# 引数で指定されたフォルダ内の画像を同一サイズにリサイズ
# 画像、ラベル、ファイル名を取得
def get_image_data(path):
    # 画像を格納する配列
    images = []
    # ラベルを格納する配列
    labels = []
    # ファイル名を格納する配列
    files = []
    for f in os.listdir(path):
        # イメージファイルの読み込み
        img = cv2.imread(path + f)
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray,(196,196))
        images.append(resized)
        labels.append(int(Give_label(f[0:6])))
        files.append(f)

    return images,labels,files


# メイン部分
# 訓練用画像を取得
images,labels,files = get_image_data(train_dir)

# トレーニング実施
recognizer.train(images, np.array(labels))

# テスト画像を取得
test_images, test_labels, test_files = get_image_data(test_dir)
i = 0
while i < len(test_labels):
    # テスト画像に対して予測実施
    label, confidence = recognizer.predict(test_images[i])
    # 予測結果をコンソール出力
    print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
    # テスト画像を表示
    if label == 1:
        name = "Tetuya"
    elif label == 2:
        name = "KIMUTAKU"
    elif label == 3:
        name = "Sisyo-"
    elif label == 4:
        name = "MURO"

    cv2.putText(test_images[i], "he is ", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(test_images[i], name, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 4, cv2.LINE_AA)
    cv2.imshow("test image", test_images[i])
    cv2.waitKey(3000)

    i += 1

# 終了処理
cv2.destroyAllWindows()
