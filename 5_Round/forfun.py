from collections import defaultdict

def statistics():
    filenames = ['total_train.txt', 'total_dev.txt', 'total_test.txt']
    len2count = defaultdict()
    for fil in filenames:
        readfile = codecs.open('/export/home/Dataset/incrementalFewShotTextClassification/Incremental-few-shot-text-classification-master/dataset/banking77/split/'+fil, 'r', 'utf-8')
        for line in readfile:
            sent = line.splist('\t')[1].strip().split()
            lens = len(sent)
            len2count[lens]+=1
    print(sorted(len2count))


if __name__ == "__main__":
    statistics()
