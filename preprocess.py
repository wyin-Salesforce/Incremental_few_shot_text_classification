import codecs
import csv
import json

path = '/export/home/Dataset/incrementalFewShotTextClassification/wenpeng/'
def statistics(filename_list, quotechar = None):
    '''first load all classes'''
    readfile = codecs.open(path+'categories.json', 'r', 'utf-8')
    class_list = json.load(readfile)

    '''load train file'''
    class_2_textlist = {}
    for filename in filename_list:
        line_co = 0
        with open(path+filename, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                # if sys.version_info[0] == 2:
                #     l'ine = list(unicode(cell, 'utf-8') for cell in line)

                # if len(line)
                class_str = line['category']
                sent = line['text']
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
            assert key  in set(class_list)
        print('class_2_textlist size:', len(class_2_textlist), ' sent size:', text_size)




if __name__ == "__main__":
    statistics(['train.csv', 'test.csv'])
