import os # osモジュールのインポート

# os.listdir('パス')
# 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
files = os.listdir("C:/Users/naoko/Desktop/OpenCV_tutorial/human_data/")

for file in files:
    print(file)
