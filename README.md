# tensorflow-crf

提供一种基于`Tensorflow 2.x`的十分简洁易用[CRF](./crf.py)实现，并提供一些常见的例子。



命名实体识别（NER）：

- `task_ner_bilstm_crf.py`
- `task_ner_cnn_crf.py`

中文分词（CWS）：

- `task_cws_bilstm_crf.py`
- `task_cws_cnn_crf.py`
- 更多中文分词的实现见[chinese-cut-word](https://github.com/allenwind/chinese-cut-word)。

词性标注（POS）：

- `task_pos_bilstm_crf.py`
- `task_pos_cnn_crf.py`
