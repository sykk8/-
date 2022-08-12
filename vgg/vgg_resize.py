from PIL import Image
import glob

classnames = ['kinsi', 'nomal_30', 'nomal_40', 'tearai', 'yowai_30', 'yowai_40'] #'kinsi', 'nomal_30', 'nomal_40', 'tearai', 'yowai_30', 'yowai_40'

for classname in classnames:
    img_path_list = glob.glob('./sentaku_mark_images/test/' + classname +'/*.jpg')
    n = 0

    for img_path in img_path_list:
        # リサイズ前の画像を読み込み
        img = Image.open(img_path)
        # 読み込んだ画像の幅、高さを指定
        (width, height) = (112, 112)
            # VGGが224×224だから
        # 画像をリサイズする
        img_resized = img.resize((width, height))
        # ファイルを保存
        img_resized.save('./sentaku_mark_images/test/' + classname + '/results' + str(n) +'.jpg', quality=90)
        n += 1