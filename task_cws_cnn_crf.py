import os
import glob
import itertools
import collections
import numpy as np
import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from dataset import CharTokenizer
from dataset import load_ctb6_cws
from layers import MaskBiLSTM
from crf import CRF, ModelWithCRFLoss
from base import TokenizerBase

id2tag = {0:"S", 1:"B", 2:"M", 3:"E"}
def build_sbme_tags(sentences, onehot=True):
    y = []
    for sentence in sentences:
        tags = []
        for word in sentence:
            if len(word) == 1:
                tags.append(0)
            else:
                tags.extend([1] + [2]*(len(word)-2) + [3])
        tags = np.array(tags)
        if onehot:
            tags = to_onehot(tags)
        y.append(tags)
        assert len("".join(sentence)) == len(tags)
    return y

def load_dataset(file):
    sentences = load_ctb6_cws(file=file)
    X = ["".join(sentence) for sentence in sentences]
    y = build_sbme_tags(sentences, onehot=False)
    return X, y

def pad(x, maxlen):
    x = sequence.pad_sequences(
        x, 
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0
    )
    return x

def preprocess_dataset(X, y, maxlen, tokenizer):
    X = tokenizer.transform(X)
    X = pad(X, maxlen)
    y = pad(y, maxlen)
    return X, y

def segment_by_tags(tags, sentence):
    # 通过SBME序列对sentence分词
    assert len(tags) == len(sentence)
    buf = ""
    for tag, char in zip(tags, sentence):
        tag = id2tag[tag]
        # t is S or B
        if tag in ["S", "B"]:
            if buf:
                yield buf
            buf = char
        # t is M or E
        else:
            buf += char
    if buf:
        yield buf

class CWSTokenizer(TokenizerBase):
    """封装好的中文分词器"""

    def __init__(self, model, tokenizer, maxlen):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def find_word(self, sentence):
        size = len(sentence)
        ids = self.tokenizer.transform([sentence])
        padded_ids = pad(ids, maxlen)
        tags = self.model.predict(padded_ids)[0]
        tags = tags[:size]
        yield from segment_by_tags(tags, sentence)

X_train, y_train = load_dataset("train.txt")
tokenizer = CharTokenizer(mintf=5)
tokenizer.fit(X_train)

maxlen = 128
hdims = 128
num_classes = 4
vocab_size = tokenizer.vocab_size

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs) # 全局mask
x = Embedding(input_dim=vocab_size, output_dim=hdims)(inputs)
x = LayerNormalization()(x)
# 用三层Conv1D替代BiLSTM
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Dense(hdims)(x)
x = Dense(num_classes)(x)
# CRF需要mask来完成不定长序列的处理，这里是手动传入
# 可以设置Embedding参数mask_zero=True，避免手动传入
crf = CRF(trans_initializer="orthogonal")
outputs = crf(x, mask=mask)

base = Model(inputs=inputs, outputs=outputs)
model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")

X_train, y_train = preprocess_dataset(X_train, y_train, maxlen, tokenizer)
X_val, y_val = load_dataset("dev.txt")
X_val, y_val = preprocess_dataset(X_val, y_val, maxlen, tokenizer)

batch_size = 32
epochs = 5
file = "weights/weights.task.cws.cnn.crf"
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val)
)

X_test, y_test = load_dataset("test.txt")
X_test, y_test = preprocess_dataset(X_test, y_test, maxlen, tokenizer)
model.evaluate(X_test, y_test)
model.save_weights(file)

if __name__ == "__main__":
    import dataset
    trans = tf.convert_to_tensor(crf.trans)
    trans = np.array(trans, dtype=np.float32)
    print(trans)
    tokenizer = CWSTokenizer(model, tokenizer, maxlen)
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))

# 示例
# ['黑天', '鹅', '和', '灰犀牛', '是', '两', '个', '突发性', '事件']
# ['黄马', '与', '黑马', '是', '马', '，', '黄马', '与', '黑马', '不', '是', '白马', '，', '因此', '白马', '不', '是', '马', '。']
