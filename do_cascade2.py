# python train_flow.py
import os
print(os.getcwd())
# 現在のディレクトリをprint
# >>> C:\Users\Yukako.S\Desktop\sentaku_tag
import cv2

# 以下を関数化
def do_cascade(CLOTH_IMG_C):
    # cv2で画像を読み込む（行×列×色の三次元のndarrayとなる）
    # img = cv2.imread('tag_img\imgtag (16-2).jpg')
    img = cv2.imread(CLOTH_IMG_C)
    # print(img)

    # 返り値が None になっていないか (読み込みに失敗していなか) 確認
    if img is None:
        print('Failed to load image.')

    # cv2.imshow('image_test', img)

    # カスケード分類機の読み込み・ファイルパス
    cascade = cv2.CascadeClassifier('static/cascade/cascade.xml')
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # 読み込ませたxmlファイル（カスケード分類機）にグレースケールのデータ（＝gray)をセットし、検出させる
    # 検出に成功した場合は画像のサイズを出力
    S_TAG = cascade.detectMultiScale(dst, scaleFactor=1.5, minNeighbors=5)
            #scaleFactor – 各画像スケールにおける縮小量を表します
            #minNeighbors – 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
            #minSize – 物体が取り得る最小サイズ．これよりも小さい物体は無視されます

    # 検出領域の(左上の点のx座標, y座標, 幅, 高さ)のリスト（配列）を返す
    # print(S_TAG) 
    # 認識しない場合は"Failed"と出力
    if len(S_TAG) ==0:
        print("Failed")
        cv2.waitKey(100)


    i=0
    for (x,y,w,h) in S_TAG:
        # 切り出す範囲の左上と右下の座標を指定
        crop_left_top = [x, y]
        crop_right_bottom = [x+w, y+h]

        # 切り出し座標を調整
        left = crop_left_top[0]
        right = crop_right_bottom[0] + 1
        top = crop_left_top[1]
        bottom = crop_right_bottom[1] + 1
        cropped = img[top : bottom, left : right]

        # 切り出した画像を保存
        # cv2.imwrite("static/save_img_cascade/result" + '{0:04d}'.format(i) + ".jpg", cropped)
        
        i += 1

        result = cv2.rectangle(dst,(x,y),(x+w,y+h),(255,0,255),thickness=2)
        
        # cv2.rectangle(gray, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=2)

    # 画像の表示
    cv2.imshow("image",result)
    cv2.waitKey(0)
    # return result_imgs

do_cascade(r"C:\Users\Yukako.S\Desktop\pos (3).jpg")