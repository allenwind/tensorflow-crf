import itertools

def gen_ner_labels(tags, clabels):
    labels = itertools.product(tags, clabels)
    labels = ["-".join(i) for i in labels]
    labels.append("O")
    id2label = {i:j for i,j in enumerate(labels)}
    label2id = {j:i for i,j in id2label.items()}
    return labels, id2label, label2id

if __name__ == "__main__":
    labels, id2label, label2id = gen_ner_labels(["B", "I"], ["PER", "LOC", "ORG"])
    print(labels)
    print(id2label)
    print(label2id)
