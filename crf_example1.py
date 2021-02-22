import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from crf import CRF, ModelWithCRFLoss

vocab_size = 5000
hdims = 128
inputs = Input(shape=(None,), dtype=tf.int32)
# 设置mask_zero=True获得全局mask
x = Embedding(vocab_size, hdims, mask_zero=True)(inputs)
x = Bidirectional(LSTM(hdims, return_sequences=True))(x)
x = Dense(4)(x)
crf = CRF(trans_initializer="orthogonal")
outputs = crf(x)
base = Model(inputs, outputs)
model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")
X = tf.random.uniform((32*100, 64), minval=0, maxval=vocab_size, dtype=tf.int32)
y = tf.random.uniform((32*100, 64), minval=0, maxval=4, dtype=tf.int32)
model.fit(X, y)
tags = model.predict(X)
print(tags.shape)
