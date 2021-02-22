import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from crf import CRF, ModelWithCRFLoss

vocab_size = 5000
hdims = 128
inputs = Input(shape=(None,), dtype=tf.int32)
# 手动计算全局mask
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(vocab_size, hdims, mask_zero=False)(inputs)
# 用三层Conv1D替代BiLSTM
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Dense(4)(x)
crf = CRF(trans_initializer="orthogonal")
outputs = crf(x, mask=mask)
base = Model(inputs, outputs)
model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")
X = tf.random.uniform((32*100, 64), minval=0, maxval=vocab_size, dtype=tf.int32)
y = tf.random.uniform((32*100, 64), minval=0, maxval=4, dtype=tf.int32)
model.fit(X, y)
tags = model.predict(X)
print(tags.shape)
