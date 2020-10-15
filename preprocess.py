
import csv
import json
def statistics(input_file, quotechar = None):
    '''first load all classes'''
    readfile = codecs.open('/export/home/Dataset/incrementalFewShotTextClassification/wenpeng/categories.json', 'r', 'utf-8')
    data = json.load(readfile)
    print('data:', len(data), data)
    exit(0)
    with open(input_file, "r") as f:
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
