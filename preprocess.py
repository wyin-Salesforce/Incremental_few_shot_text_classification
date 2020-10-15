import codecs
import csv
import json
def statistics(filename_list, quotechar = None):
    '''first load all classes'''
    readfile = codecs.open('/export/home/Dataset/incrementalFewShotTextClassification/wenpeng/categories.json', 'r', 'utf-8')
    class_list = json.load(readfile)

    '''load train file'''
    class_2_textlist = {}
    for filename in filename_list:
        line_co = 0
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            for line in reader:
                # if sys.version_info[0] == 2:
                #     l'ine = list(unicode(cell, 'utf-8') for cell in line)
                class_str = line[0].strip()
                sent = line[1].strip()
                if line_co > 0:
                    textlist = class_2_textlist.get(class_str)
                    if textlist is None:
                        textlist = []
                    textlist.append(sent)
                    class_2_textlist[class_str] = textlist
                line_co+=1

        text_size = 0
        for key, value in class_2_textlist.items():
            text_size+=len(value)
        print('class_2_textlist size:', len(class_2_textlist), ' sent size:', text_size)




if __name__ == "__main__":
    statistics('/export/home/Dataset/incrementalFewShotTextClassification/wenpeng/train.csv')
