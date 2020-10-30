import statistics
import codecs

import json


# initializing list
test_list = [11.43, 0.0, 3.21]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
def compute(test_list):
    average = round(sum(test_list)/len(test_list), 2)
    res = round(statistics.pstdev(test_list),2)

    return str(average)+'/'+str(res)

def extract(flag):
    filenames = ['log.supervised.'+flag+'.seed.42.debug.txt',
                 'log.supervised.'+flag+'.seed.16.debug.txt',
                 'log.supervised.'+flag+'.seed.32.debug.txt']
    result_lists = []
    for fil in filenames:
        readfile = codecs.open('/export/home/workspace/Incremental_few_shot_text_classification/5_Round/'+fil, 'r', 'utf-8')
        for line in readfile:
            line_str  = line.strip()
            if line_str.startswith('final_test_performance'):
                position = line_str.find(':')
                target_list = json.loads(line_str[position:].strip())
                print('target_list:', target_list)
                result_lists.append(target_list)
                break
        readfile.close()
    assert len(result_lists[0]) == len(result_lists[1])
    assert len(result_lists[0]) == len(result_lists[2])
    final_results = []
    for i in range(len(result_lists[0])):
        final_results.append(compute([result_lists[0][i], result_lists[1][i], result_lists[2][i]]))
    print('final_results:', final_results)

if __name__ == "__main__":
    extract('base')
