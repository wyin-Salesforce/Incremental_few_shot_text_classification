from collections import Counter
import codecs
def statistics():
    filenames = ['total_train.txt', 'total_dev.txt', 'total_test.txt']
    len2count = Counter()
    for fil in filenames:
        readfile = codecs.open('/export/home/Dataset/incrementalFewShotTextClassification/Incremental-few-shot-text-classification-master/dataset/banking77/split/'+fil, 'r', 'utf-8')
        for line in readfile:
            sent = line.split('\t')[1].strip().split()
            lens = len(sent)
            len2count[lens]+=1
    print('len2count:', len2count)
    sorted(len2count.items())
    print(len2count)


if __name__ == "__main__":
    statistics()