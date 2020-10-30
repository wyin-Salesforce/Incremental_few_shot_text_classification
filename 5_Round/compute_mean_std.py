import statistics
import codecs
import ast


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
    filenames = ['log.prototype.'+flag+'.seed.42.txt',
                 'log.prototype.'+flag+'.seed.16.txt',
                 'log.prototype.'+flag+'.seed.32.txt']
    result_lists = []
    for fil in filenames:
        readfile = codecs.open('/export/home/workspace/Incremental_few_shot_text_classification/5_Round/'+fil, 'r', 'utf-8')
        for line in readfile:
            line_str  = line.strip()
            if line_str.startswith('final_test_performance'):
                position = line_str.find(':')
                target_list = ast.literal_eval(line_str[position+1:].strip())
                # print('target_list:', target_list)
                result_lists.append(target_list)
                break
        readfile.close()
    assert len(result_lists[0]) == len(result_lists[1])
    assert len(result_lists[0]) == len(result_lists[2])
    final_results = []
    for i in range(len(result_lists[0])):
        final_results.append(compute([result_lists[0][i]*100.0, result_lists[1][i]*100.0, result_lists[2][i]*100.0]))
    print('final_results:', final_results)

if __name__ == "__main__":
    extract('base')
    extract('r1')
    extract('r2')
    extract('r3')
    extract('r4')
    extract('r5')
