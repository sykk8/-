import glob
import cv2
import numpy as np


##### 精度を上げるためにやったこと #####
#同じ向きの画像だけにした
#元画像を増やした（色ついてるやつとか）
#image.pyで白黒反転、コントラスト調節、ノイズ
#resize.pyでリサイズ→向きが変わってしまった画像は回転させて元の向きに戻す
#クラスごとのデータ数を同程度に揃える（自分で絵描いたりした笑）
#model.pyで角度を15度だけ付けで画像拡張
#EarlyStoppingで学習が進まなくなったらor過学習になったら学習を止める


classnames = ['kinsi', 'nomal_30', 'nomal_40', 'tearai', 'yowai_30', 'yowai_40'] 


for classname in classnames:
    images = glob.glob('./sentaku_mark_images/validation/' + classname +'/*.jpg') #クラスごとに画像を読み込む
    print(images)
    for n in range(len(images)):
        image = cv2.imread(images[n])

        # 1.色反転
        black_img = cv2.bitwise_not(image)
        #変換後の画像を保存
        cv2.imwrite('./sentaku_mark_images/validation/' + classname +'/results/image_black' + str(n) +'.jpg', black_img)

        # 2.コントラスト
        # ルックアップテーブルの生成
        min_table = 50
        max_table = 205
        diff_table = max_table - min_table
        LUT_HC = np.arange(256, dtype = 'uint8' )
        LUT_LC = np.arange(256, dtype = 'uint8' )
        # ハイコントラストLUT作成
        for i in range(0, min_table):
            LUT_HC[i] = 0
        for i in range(min_table, max_table):
            LUT_HC[i] = 255 * (i - min_table) / diff_table
        for i in range(max_table, 255):
            LUT_HC[i] = 255
        # ローコントラストLUT作成
        for i in range(256):
            LUT_LC[i] = min_table + i * (diff_table) / 255
        high_cont_img = cv2.LUT(image, LUT_HC)
        low_cont_img = cv2.LUT(image, LUT_LC)
        #変換後の画像を保存
        cv2.imwrite('./sentaku_mark_images/validation/' + classname +'/results/image_high' + str(n) +'.jpg',high_cont_img)
        cv2.imwrite('./sentaku_mark_images/validation/' + classname +'/results/image_low' + str(n) +'.jpg',low_cont_img)

        # 3.ノイズ
        row,col,ch= image.shape
        mean = 0
        sigma = 15
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss_img = image + gauss
        #変換後の画像を保存
        cv2.imwrite('./sentaku_mark_images/validation/' + classname +'/results/image_gauss' + str(n) +'.jpg',gauss_img)