import tensorflow as tf
import tqdm

class Evaluator(tf.keras.callbacks.Callback):
    """prf指标评估"""

    def __init__(self, ner, data):
        self.ner = ner
        self.samples = [(x,y) for x,y in zip(*data)]

    def evaluate_prf(self, samples):
        TP = TPFP = TPFN = 1e-12
        for text, tags in tqdm.tqdm(samples):
            R = set(self.ner.find(text)) # 预测正类
            T = set(find_entities(text, tags)) # 真正类
            TP += len(R & T) # TP
            TPFP += len(R) # TP + FP
            TPFN += len(T) # TP + FN
        p = TP / TPFP * 100
        r = TP / TPFN * 100
        f1 = 2 * TP / (TPFP + TPFN) * 100
        return p, r, f1

    def on_epoch_end(self, epoch, logs=None):
        p, r, f1 = self.evaluate_prf(self.samples)
        template = "precision:{:.2f}%, recall:{:.2f}%, f1:{:.2f}%"
        print(template.format(p, r, f1))
