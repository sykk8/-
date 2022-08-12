import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob


# 関数にするための変数を先にまとめている
# file_name = 'vgg16_sentaku_mark'
# hw={'height':224, 'width':224}

def do_VGG16(VGG_IMG_PATH):

    imgname = VGG_IMG_PATH


    modelname_text = open("./static/vgg16_sentaku_mark_model.json").read()
    json_strings = modelname_text.split('##########')
    #マークの名前の読み込み
    textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
    #モデル構造の読み込み
    model = model_from_json(json_strings[0]) 
    #重みの読み込み
    model.load_weights('./static/temp/model-36.h5')
    print('done')
    # for imgname in imgnames:
    #画像の前処理
    img = load_img(imgname, target_size=(224, 224))
    # 正規化（原則やる、計算内で重みの力を一定にするため）
    TEST = img_to_array(img)/255
    #予測
    pred = model.predict(np.array([TEST]), batch_size=1, verbose=0)
        

    name = textlist[np.argmax(pred)].replace(",", "")

    print(name)
    return name

    ##### 結果 #####
    # kinsi:9/10(tearai)→10/10
    # nomal_30:5/6(yowai_40)→5/6(yowai_40)
    # nomal_40:3/5(tearai)→4/5
    # tearai:7/8(yowai_40)→tearai:7/8(yowai_40)
    # yowai_30:4/8(terai*1, yowai_40*3)→4/8yowai_40との間違いなので問題なし！
    # yowai_40:9/10(yowai_30)→10/10
    
# do_VGG16('../static/save_img_cascade/result0002.jpg')

# p = glob.glob('../static/save_img_cascade/*.jpg')
# print(p)