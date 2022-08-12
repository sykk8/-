# ルーティング用ファイル

from distutils.log import error
import os
import shutil# 入力画像・フォルダを削除するためのもの
from flask import (
     Flask, 
     request, 
     render_template,
     url_for,
     redirect, 
     send_from_directory
     )
import glob

# 関数化したカスケード分類機を読み込む
from sentaku_tag_6.do_cascade import do_cascade
from vgg.vgg_predict import do_VGG16

#カスケード分類機に入力する画像のアップロード先ディレクトリ
UPLOAD_FOLDER_C='./static/cloth_img_c'

#FlaskでAPIを書くときのおまじない
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pic_upload', methods=['GET', 'POST'])
def pic_upload():
    shutil.rmtree('./static/save_img_cascade')
    os.mkdir('./static/save_img_cascade')
    return render_template(
        'pic_upload.html',
        enter_images=os.listdir(UPLOAD_FOLDER_C)[::-1]
        )
    # 一回目はenter_imagesが空
    # 画像がアップロードされたら同じページにリダイレクトし、htmlのif以下が表示

# アップロードされた画像を保存する（処理のみ）
@app.route('/pic_upload_do', methods=['GET', 'POST'])
def pic_upload_do():
    # 画像のアップロード保存
    if request.method == 'POST':
        upload_file = request.files['upload_file']
        img_path = os.path.join(UPLOAD_FOLDER_C,'original.jpg')
        upload_file.save(img_path)
    return redirect('/pic_upload')


#ディレクトリに保存されている画像をブラウザに送る処理
@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory(UPLOAD_FOLDER_C, path)



@app.route('/pic_take')
def pic_take():
    return render_template('pic_take.html')

@app.route('/pic_check')
def pic_check():
    return render_template('pic_check.html')




@app.route('/cascade_view', methods=['GET', 'POST'])
def cascade_view():
    # result_imgs = do_cascade(f'{UPLOAD_FOLDER_C}/original.jpg')
    do_cascade(f'{UPLOAD_FOLDER_C}/original.jpg')
    # do_cascade(UPLOAD_FOLDER_C + '/*.jpg')
    # do_cascade('static/cloth_img_c/imgtag_test.jpg')
    # do_cascade(url_for('static', filename='cloth_img_c/imgtag(14).jpg'))
    
    result_imgs = glob.glob('static/save_img_cascade/*.jpg')
    # print(result_imgs)
    return render_template('cascade_view.html',
    result_imgs = result_imgs,
    error=request.args.get('error')# urlのqueryを変数として受け取る
    )

from werkzeug.exceptions import BadRequest
@app.errorhandler(BadRequest)
def handle_bad_request(e):
    # return 'bad request!', 400
    return redirect(url_for('cascade_view',error=1))
    # urlが変わる　get query

    # return redirect('/cascade_view',)


@app.route('/vgg_view', methods=['POST'])
def vgg_view():
    # try:
        result = request.form['result']
        print('この画像の予測は' + result + 'です！')
        # >>>result=選択した画像のパス
        # do_VGG16(VGG_IMG_PATH)
        MARK_NAME = do_VGG16(result)
        # >>>マークの名前（VGGの結果）
        print('last_view' + '=' + MARK_NAME,
        'last_mark=' + result
        )
        shutil.rmtree('./static/cloth_img_c')
        os.mkdir('./static/cloth_img_c')
        return render_template(
            'vgg_view.html',
            last_view = result, #画像を出力：パスが欲しい 
            last_mark = MARK_NAME #予測結果（マークの名前）を出力：VGGの結果が欲しい 
            )
    # except BadRequestKeyError:
    #     return redirect('/vgg_view')


if __name__ == "__main__":
    app.run(debug=True)