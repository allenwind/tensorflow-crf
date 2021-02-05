import random
import collections
import itertools
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.preprocessing import sequence

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from crf import CRF, ModelWithCRFLoss

# 标签映射
# NER标注一般使用IOBES，不过这里数据集标注实体只用B-I
labels = {"O", "B-ORG", "I-ORG", "B-LOC", "I-LOC","B-PER", "I-PER"}
id2label = {i:j for i,j in enumerate(labels)}
label2id = {j:i for i,j in id2label.items()}
num_classes = len(labels)

PATH = "dataset/china-people-daily-ner-corpus/example."
def load_dataset(file, shuffle=True):
    # 返回逐位置标注形式
    file = PATH + file
    with open(file, encoding="utf-8") as fp:
        text = fp.read()
    lines = text.split("\n\n")
    if shuffle:
        random.shuffle(lines)
    X = []
    y = []
    for line in lines:
        if not line:
            continue
        chars = []
        tags = []
        for item in line.split("\n"):
            char, label = item.split(" ")
            chars.append(char)
            tags.append(label2id[label])
        X.append("".join(chars))
        y.append(tags)

        assert len(chars) == len(tags)
    return X, y

def find_entities(text, tags):
    # 根据标签提取文本中的实体
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
            tag = id2label[tag]
            if tag == "O":
                continue
            tag, label = tag.split("-")
            if tag == "B":
                if buf:
                    yield buf, plabel
                buf = char
            elif tag == "I":
                buf += char
            plabel = label

        if buf:
            yield buf, plabel

    entities = list(segment_by_tags(text, tags))
    return entities

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

class CharTokenizer:
    """字符级别Tokenizer"""

    char2id = {}
    MASK = 0
    UNK = 1 # 未登陆字用1表示

    def fit(self, X):
        chars = collections.defaultdict(int)
        for sample in X:
            for c in sample:
                chars[c] += 1
        self.char2id = {j:i for i,j in enumerate(chars, start=2)}

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNK))
            ids.append(s)
        return ids

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

class NamedEntityRecognizer:
    """封装好的实体识别器"""

    def __init__(self, model, tokenizer, maxlen):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def find(self, text):
        size = len(text)
        ids = self.tokenizer.transform([text])
        padded_ids = pad(ids, self.maxlen)
        tags = self.model.predict(padded_ids)[0][0]
        tags = tags[:size]
        return find_entities(text, tags)


X_train, y_train = load_dataset("train")
tokenizer = CharTokenizer()
tokenizer.fit(X_train)

file = "weights/weights.task.ner.bilstm.crf"
maxlen = 128
batch_size = 32
hdims = 128
vocab_size = tokenizer.vocab_size

X_train, y_train = preprocess_dataset(X_train, y_train, maxlen, tokenizer)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs) # 全局mask
x = Embedding(input_dim=vocab_size, output_dim=hdims)(inputs)
# 用三层Conv1D替代BiLSTM
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Conv1D(128, 3, activation="relu", padding="same")(x)
x = Dense(num_classes)(x)
crf = CRF(trans_initializer="orthogonal")
# CRF需要mask来完成不定长序列的处理，这里是手动传入
# 可以设置Embedding参数mask_zero=True，避免手动传入
outputs = crf(x, mask=mask)

base = Model(inputs=inputs, outputs=outputs)
base.summary()

model = ModelWithCRFLoss(base)
model.compile(optimizer="adam")

X_val, y_val = load_dataset("dev")
X_val, y_val = preprocess_dataset(X_val, y_val, maxlen, tokenizer)

model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=10,
    validation_data=(X_val, y_val)
)

if __name__ == "__main__":
    X_test, y_test = load_dataset("test")
    ner = NamedEntityRecognizer(model, tokenizer, maxlen)
    for x, y in zip(X_test, y_test):
        print(find_entities(x, y)) # 真实的实体
        print(ner.find(x)) # 预测的实体
        input()

# 输出结果示例
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国之音', 'ORG'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国', 'LOC'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]
