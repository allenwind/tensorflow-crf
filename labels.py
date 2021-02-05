import itertools

# 标签映射测试

labels = itertools.product(["B", "I"], ["PER", "LOC", "ORG"])
labels = ["-".join(i) for i in labels]
labels.append("O")
id2label = {i:j for i,j in enumerate(labels)}
label2id = {j:i for i,j in id2label.items()}
print(labels)
print(id2label)
print(label2id)
