from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping

# カテゴリーは6個
n_categories=6
batch_size=32
# ディレクトリを指定
train_dir='./sentaku_mark_images/train'
validation_dir='./sentaku_mark_images/validation'
test_dir='./sentaku_mark_images/test'
file_name='vgg16_sentaku_mark'

ClassNames=['kinsi','nomal30','nomal40', 'tearai', 'yowai30', 'yowai40']


# データを保存するディレクトリのパス
# SAVE_DATA_DIR_PATH = './model'


# VGG16のモデルをインポートする
base_model=VGG16(weights='imagenet',# 重みはImageNetを利用
                 include_top=False,# 全結合層は不要なためFalse
                 input_tensor=Input(shape=(224,224,3)))# 画像は224×224のRGB（3原色）


#add new layers instead of FC networks
# FC層（全結合層）に代わる新たな層（＝新しい全結合層）を追加
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
prediction=Dense(n_categories,activation='softmax')(x)
# モデルの名前＝Model(input=,output=とそれぞれ定義)
# 一連の層を、modelという名前で一つにまとめ、使えるようにする
model=Model(inputs=base_model.input,outputs=prediction)

#fix weights before VGG16 14layers
# 0～14層目までの重みを修正する部分（修正なし）
for layer in base_model.layers[:15]:
    layer.trainable=False
    # Flase=更新しない、ということ。そのまま使う

# 学習のためのモデルを設定（学習プロセスの設定）
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# optimizer:optimizer名、loss:目的関数名、metrics:訓練時とテスト時にモデルにより評価される評価関数のリスト

# モデルの要約を出力(モデルの構造を見る)
model.summary()

#モデル構造, クラス名の保存
# これがあると、表示させた時にクラス名が出る
json_string=model.to_json()
json_string+='##########' + str(ClassNames)
open(file_name + 'model.json',"w").write(json_string)

# ImageDataGenerator
# trainデータを正規化、ランダムにズーム・左右反転・上下反転
train_datagen=ImageDataGenerator(
    rescale=1.0/255, rotation_range=15) #正規化, shear_range=35
    #brightness_range = [0.5, 1.5], #指定した値の範囲でランダムに明るさを調整（1.0以下は暗く、1.0以上は明るくなる）
    #channel_shift_range = 100) #指定した値の範囲でチャンネルをシフト（=画像を構成するRGBチャンネルの値を変更する）
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    #vertical_flip=True)

# validationデータを正規化
validation_datagen=ImageDataGenerator(rescale=1.0/255)
# カラー画像が255色あるのを255で割って0～1までに置き換えた処理　特徴量を正規化する
# それが正規化

# ディレクトリへのパスを受け取り、拡張/正規化したtrainデータのバッチを生成。
train_generator=train_datagen.flow_from_directory(
    train_dir,    # ディレクトリへのパス
    target_size=(224,224),    # 全画像がこの大きさにリサイズされる
    batch_size=batch_size,    # データのバッチのサイズ
    class_mode='categorical',    # 返すラベルの配列のshapeを決定。categolical：2次元のone-hotにエンコード化されたラベル
    shuffle=True    # データをシャッフルするかどうか
)
# ディレクトリへのパス(validation_dir)を受け取り、正規化したvalidationデータのバッチを生成
validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
    )

# 一回目の学習

from tensorflow.keras.callbacks import ModelCheckpoint
import os

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)# ＝training_1というディレクトリを取得
MODEL_DIR = "./temp"
# モデルを保存するのはtempディレクトリ
if not os.path.exists(MODEL_DIR):  # ディレクトリが存在しない場合、作成する。
    os.makedirs(MODEL_DIR)


# チェックポイントコールバックを作る
cp_callback = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"),
    # filepath=checkpoint_path, # : 文字列，モデルファイルを保存するパス
    save_weights_only=True, # :Trueなら，モデルの重みが保存される。そうでないなら，モデルの全体が保存される
    verbose=1,  # 表示される出力のモードの指定（多分）
    period=1 # :チェックポイント間の間隔（エポック数）
    )

# 一回目の学習実行！

# ?
# 実際に学習を行う
# 入力データ、教師データを引数に与える
hist=model.fit_generator(train_generator,
                         epochs=60,
                         verbose=1,
                         validation_data=validation_generator,
                         callbacks=[cp_callback,
                                    # 訓練にコールバックを渡す
                                    CSVLogger(file_name+'.csv'),
                                    # CSVLogger:lossなどをCSVに保存してくれるもの。結果をバージョン管理しておくと、変更して良くなったか悪くなったか分かりやすい 
                                    EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0,  verbose=1)
                                    # 改善が見られなくなった時点で訓練を終了する 引数Trueで、ベスト時ではなく最終時の重さが保持されるよう指定
                                    ]                        
                        # callbacks:訓練中（エポック終了時など）に呼び出した関数を[リスト]で指定する
                         )
# history = とすることで、後に結果を取得（出力）することができる（必要なければしなくてもよい）

#save weights
# モデルの重みの保存#evaluate model
# モデルの評価（テストデータに対するスコアを計算する）
score=model.evaluate_generator(train_generator)
print('\n train loss:',score[0])
# trainデータに対する損失関数の値。小さいほど汎化性能が出ている
print('\n train_acc:',score[1])
# trainデータに対する分類の正答率

# これがエポックの横にでてるやつでは！？てことは必要ない

score=model.evaluate_generator(validation_generator)
print('\n validation loss:',score[0])
# validationデータに対する損失関数の値。小さいほど汎化性能が出ている
print('\n validation_acc:',score[1])
# validationデータに対する分類の正答率
model.save(file_name+'weight.h5')
# h5:HDF5ファイル

#testデータの前処理
test_datagen=ImageDataGenerator(rescale=1.0/255)
test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#モデルの評価
score=model.evaluate_generator(test_generator)
print('\n test loss:',score[0])
print('\n test_acc:',score[1])





