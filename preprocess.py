import codecs
import csv
import json
import random

path = '/export/home/Dataset/incrementalFewShotTextClassification/wenpeng/'
def load_raw_examples(filename_list, quotechar = None):
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
    return class_2_textlist


def split_into_three_rounds(class_2_textlist):
    '''first split classes into: base, r1, r2, ood'''
    all_class_list  = list(class_2_textlist.keys())
    random.shuffle(all_class_list)
    base_class_list = all_class_list[:40] # 40
    r1_class_list = all_class_list[40:50] # 10
    r2_class_list = all_class_list[50:60] # 10
    ood_class_list = all_class_list[60:] # 17

    '''base to train, dev and test'''
    base_examples_in_train = set()
    base_examples_in_dev = set()
    base_examples_in_test = set()

    for cl in base_class_list:
        textlist  = class_2_textlist.get(cl)
        random.shuffle(textlist)
        train_examples = textlist[:-60]
        dev_examples = textlist[-60:-40] #20
        test_examples = textlist[-40:] # 40
        for text in train_examples:
            base_examples_in_train.add((cl, text))
        for text in dev_examples:
            base_examples_in_dev.add((cl, text))
        for text in test_examples:
            base_examples_in_test.add((cl, text))
    '''r1 to train, dev and test'''
    r1_examples_in_train = set()
    r1_examples_in_dev = set()
    r1_examples_in_test = set()

    for cl in r1_class_list:
        textlist  = class_2_textlist.get(cl)
        random.shuffle(textlist)
        k_shot = random.randrange(1,6)
        k_shot_examples = textlist[:k_shot] #k_shot
        dev_examples = textlist[k_shot:k_shot+20] #20
        test_examples = textlist[k_shot+20:k_shot+60] # 40
        for text in k_shot_examples:
            r1_examples_in_train.add((cl, text))
        for text in dev_examples:
            r1_examples_in_dev.add((cl, text))
        for text in test_examples:
            r1_examples_in_test.add((cl, text))

    '''r2 to train, dev and test'''
    r2_examples_in_train = set()
    r2_examples_in_dev = set()
    r2_examples_in_test = set()

    for cl in r2_class_list:
        textlist  = class_2_textlist.get(cl)
        random.shuffle(textlist)
        k_shot = random.randrange(1,6)
        k_shot_examples = textlist[:k_shot] #k_shot
        dev_examples = textlist[k_shot:k_shot+20] #20
        test_examples = textlist[k_shot+20:k_shot+60] # 40
        for text in k_shot_examples:
            r2_examples_in_train.add((cl, text))
        for text in dev_examples:
            r2_examples_in_dev.add((cl, text))
        for text in test_examples:
            r2_examples_in_test.add((cl, text))
    '''ood to dev and test'''
    ood_examples_in_dev = set()
    ood_examples_in_test = set()

    all_ood_texts = []
    for cl in ood_class_list:
        textlist  = class_2_textlist.get(cl)
        all_ood_texts+=textlist
    random.shuffle(all_ood_texts)
    dev_examples = all_ood_texts[:20] #20
    test_examples = all_ood_texts[20:60] # 40
    for text in dev_examples:
        ood_examples_in_dev.add(('ood', text))
    for text in test_examples:
        ood_examples_in_test.add(('ood', text))

    '''combine them as the final train, dev and test in each round'''
    '''round base'''
    write_round_base_train = codecs.open(path+'round.base.train.txt', 'w', 'utf-8')
    write_round_base_dev = codecs.open(path+'round.base.dev.txt', 'w', 'utf-8')
    write_round_base_test = codecs.open(path+'round.base.test.txt', 'w', 'utf-8')
    round_base_train_exs = base_examples_in_train
    round_base_dev_exs = base_examples_in_dev | ood_examples_in_dev
    round_base_test_exs =  base_examples_in_test | ood_examples_in_test
    for ex in round_base_train_exs:
        write_round_base_train.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_base_train.close()
    for ex in round_base_dev_exs:
        write_round_base_dev.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_base_dev.close()
    for ex in round_base_test_exs:
        write_round_base_test.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_base_test.close()
    print('round base size:', len(round_base_train_exs), len(round_base_dev_exs), len(round_base_test_exs))
    '''round 1'''
    round_1_train_exs = base_examples_in_train | r1_examples_in_train
    round_1_dev_exs = base_examples_in_dev | r1_examples_in_dev | ood_examples_in_dev
    round_1_test_exs = base_examples_in_test| r1_examples_in_test | ood_examples_in_test
    write_round_1_train = codecs.open(path+'round.1.train.txt', 'w', 'utf-8')
    write_round_1_dev = codecs.open(path+'round.1.dev.txt', 'w', 'utf-8')
    write_round_1_test = codecs.open(path+'round.1.test.txt', 'w', 'utf-8')
    for ex in round_1_train_exs:
        write_round_1_train.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_1_train.close()
    for ex in round_1_dev_exs:
        write_round_1_dev.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_1_dev.close()
    for ex in round_1_test_exs:
        write_round_1_test.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_1_test.close()
    print('round 1 size:', len(round_1_train_exs), len(round_1_dev_exs), len(round_1_test_exs))
    '''round 2'''
    round_2_train_exs = base_examples_in_train | r1_examples_in_train | r2_examples_in_train
    round_2_dev_exs = base_examples_in_dev | r1_examples_in_dev | r2_examples_in_dev | ood_examples_in_dev
    round_2_test_exs = base_examples_in_test| r1_examples_in_test | r2_examples_in_test | ood_examples_in_test
    write_round_2_train = codecs.open(path+'round.2.train.txt', 'w', 'utf-8')
    write_round_2_dev = codecs.open(path+'round.2.dev.txt', 'w', 'utf-8')
    write_round_2_test = codecs.open(path+'round.2.test.txt', 'w', 'utf-8')
    for ex in round_2_train_exs:
        write_round_2_train.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_2_train.close()
    for ex in round_2_dev_exs:
        write_round_2_dev.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_2_dev.close()
    for ex in round_2_test_exs:
        write_round_2_test.write(ex[0]+'\t'+ex[1]+'\n')
    write_round_2_test.close()
    print('round 2 size:', len(round_2_train_exs), len(round_2_dev_exs), len(round_2_test_exs))








if __name__ == "__main__":
    class_2_textlist = load_raw_examples(['train.csv', 'test.csv'])
    split_into_three_rounds(class_2_textlist)
