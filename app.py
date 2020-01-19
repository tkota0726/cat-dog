# coding: utf-8

# chainerのライブラリ群
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import cv2

from flask import Flask, request, jsonify
from chainercv.transforms import resize

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


class CNN(chainer.Chain):
    #モデルの構造
    def __init__(self, n_mid_units1=224, n_out=2):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, 3, ksize=3, pad=1)
            self.fc1 = L.Linear(None,n_mid_units1)
            self.fc2 = L.Linear(None,n_out)

    #順伝播
    def __call__(self, x):
        h = self.conv(x)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        return h
    
    
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    img = cv2.imdecode(np.fromstring(file.stream.read(), np.uint8), cv2.IMREAD_COLOR)
    # chainerで扱える形式（float32）に変換
    img = img.astype(np.float32)
    # (W, H, C)を(C, H, W)に変換
    img = img.transpose(2, 0, 1)
    # 224x224にリサイズ
    img = resize(img, (224, 224))
    # 入力できる形に変換
    img = img[None, ...]

    # 推定
    y= model.predictor(img)
    # 予測結果を確率に変換
    y = F.softmax(y)

    # 結果を JSON にして返す
    return jsonify({
        'result': result_labels[np.argmax(y.array)]
    })

if __name__ == '__main__':
    model = L.Classifier(CNN())
    chainer.serializers.load_npz('models/animal.npz', model)
    result_labels = {0: '犬', 1:'猫'}
    app.run(debug=True)