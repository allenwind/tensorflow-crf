import random
import itertools
import collections
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.preprocessing import sequence

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from layers import MaskBiLSTM
from crf import CRF, ModelWithCRFLoss

PATH = "dataset/ctb5/{}.char.bmes"
def load_dataset(file, shuffle=True):
    file = PATH.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    lines = text.splitlines()
    if shuffle:
        random.shuffle(lines)
    X = []
    y = []
    for line in lines:
        sentence, tags = line.split("\t")
        sentence = sentence.replace(" ", "")
        tags = tags.split(" ")
        assert len(sentence) == len(tags)
        X.append(sentence)
        y.append(tags)
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

def preprocess_dataset(X, y, maxlen, tokenizer, lbtokenizer):
    X = tokenizer.transform(X)
    y = lbtokenizer.transform(y)
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

class LabelTransformer:
    """标签映射，标签的转换和逆转换"""

    def fit(self, y):
        self.labels = set(itertools.chain(*y))
        self.id2label = {i:j for i,j in enumerate(self.labels)}
        self.label2id = {j:i for i,j in self.id2label.items()}

    def transform(self, batch_tags):
        batch_ids = []
        for tags in batch_tags:
            ids = []
            for tag in tags:
                ids.append(self.label2id[tag])
            batch_ids.append(ids)
        return batch_ids

    def inverse_transform(self, batch_ids):
        batch_tags = []
        for ids in batch_ids:
            tags = []
            for i in ids:
                tags.append(self.id2label[i])
            batch_tags.append(tags)
        return batch_tags

    @property
    def num_classes(self):
        return len(self.labels)
    

def find_poss(text, tags):
    # POS标注使用BMES
    # 根据标签提取文本中的实体
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
            tag, label = tag.split("-", 1)
            if tag == "B" or tag == "S":
                if buf:
                    yield buf, plabel
                buf = char
            else:
                # M or E
                buf += char
            plabel = label

        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))

class POSTagger:
    """封装好的词性标注器"""

    def __init__(self, model, tokenizer, lbtokenizer, maxlen):
        self.model = model
        self.tokenizer = tokenizer
        self.lbtokenizer = lbtokenizer
        self.maxlen = maxlen

    def find(self, text):
        size = len(text)
        ids = self.tokenizer.transform([text])
        padded_ids = pad(ids, self.maxlen)
        tags = self.model.predict(padded_ids)[0]
        batch_tags = [tags[:size]]
        tags = self.lbtokenizer.inverse_transform(batch_tags)[0]
        return find_poss(text, tags)

X_train, y_train = load_dataset("train")
tokenizer = CharTokenizer()
tokenizer.fit(X_train)

lbtokenizer = LabelTransformer()
lbtokenizer.fit(y_train)

maxlen = 128
hdims = 128
num_classes = lbtokenizer.num_classes
vocab_size = tokenizer.vocab_size

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs) # 全局mask
x = Embedding(input_dim=vocab_size, output_dim=hdims)(inputs)
x = MaskBiLSTM(hdims)(x, mask=mask)
x = Dense(hdims)(x)
x = Dense(num_classes)(x)
crf = CRF(trans_initializer="orthogonal")
# CRF需要mask来完成不定长序列的处理，这里是手动传入
# 可以设置Embedding参数mask_zero=True，避免手动传入
outputs = crf(x, mask=mask)

base = Model(inputs=inputs, outputs=outputs)
model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")

X_train, y_train = preprocess_dataset(X_train, y_train, maxlen, tokenizer, lbtokenizer)
X_val, y_val = load_dataset("dev")
X_val, y_val = preprocess_dataset(X_val, y_val, maxlen, tokenizer, lbtokenizer)

batch_size = 32
epochs = 10
file = "weights/weights.task.pos.bilstm.crf"
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val)
)

model.save_weights(file)

if __name__ == "__main__":
    X_test, y_test = load_dataset("test")
    tagger = POSTagger(model, tokenizer, lbtokenizer, maxlen)
    for x, y in zip(X_test, y_test):
        print(find_poss(x, y)) # 真实的标注
        print(tagger.find(x)) # 预测的标注
        input()

# 输出结果示例
# [('李鹏', 'NR'), ('指出', 'VV'), ('，', 'PU'), ('澳门', 'NR'), ('问题', 'NN'), ('是', 'VC'), ('中', 'NR'), ('葡', 'NR'), ('双边', 'JJ'), ('关系', 'NN'), ('中', 'LC'), ('的', 'DEG'), ('一', 'CD'), ('个', 'M'), ('重要', 'JJ'), ('组成', 'NN'), ('部分', 'NN'), ('。', 'PU')]
# [('李鹏', 'NR'), ('指出', 'VV'), ('，', 'PU'), ('澳门', 'NR'), ('问题', 'NN'), ('是', 'VC'), ('中', 'NR'), ('葡', 'NR'), ('双边', 'JJ'), ('关系', 'NN'), ('中', 'LC'), ('的', 'DEG'), ('一', 'CD'), ('个', 'M'), ('重要', 'JJ'), ('组成', 'NN'), ('部分', 'NN'), ('。', 'PU')]
