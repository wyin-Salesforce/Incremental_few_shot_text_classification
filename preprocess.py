import codecs
import csv
import json
def statistics(train_file, quotechar = None):
    '''first load all classes'''
    readfile = codecs.open('/export/home/Dataset/incrementalFewShotTextClassification/wenpeng/categories.json', 'r', 'utf-8')
    class_list = json.load(readfile)

    '''load train file'''
    with open(train_file, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar=quotechar)
        lines = []
        for line in reader:
            # if sys.version_info[0] == 2:
            #     l'ine = list(unicode(cell, 'utf-8') for cell in line)
            print('line:', line)
            exit(0)
            lines.append(line)
        return lines


if __name__ == "__main__":
    statistics('/export/home/Dataset/incrementalFewShotTextClassification/wenpeng/train.csv')
