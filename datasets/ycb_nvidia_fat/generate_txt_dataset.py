from PIL import Image
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import random
root_path = "/media/qind/Data/QIND-DATA/fat"

l = {}
mix = []
sing = []
i = 0
for dirpath, dirnames, files in os.walk(root_path):
    if 'models' in dirpath:
        continue
    print('in {0} directory'.format(dirpath))
    l['{0}'.format(i)] = []
    for file_names in files:
        
        if (file_names != '_object_settings.json') and (file_names != '_camera_settings.json'):
            path = (dirpath+"/"+file_names)
            idx = path.rfind('t.')+1
            path = path[:idx]
            if path not in l['{0}'.format(i)]:
                if "mixed" in path:
                    mix.append(path[31:])
                if "single" in path:
                    sing.append(path[31:])
                l['{0}'.format(i)].append(path)
        else:
            continue
    i+=1
arr = []
for j in range(i):
    for p in range(len(l['{0}'.format(j)])):
        arr.append(l['{0}'.format(j)][p][31:])
print(len(arr))
arr[60001:], arr[:60000] = arr[:60000], arr[60001:]
with open('dataset_config/all_data.txt','w') as f:
    f.write('\n'.join(arr))
    f.close()
with open('dataset_config/mixed.txt','w') as f:
    for line in mix:
        f.write(line)
        f.write('\n')
    f.close

with open('dataset_config/single.txt','w') as f:
    for line in sing:
        f.write(line)
        f.write('\n')
    f.close
random.seed(1506)

test_set = random.sample(arr, 3000)
train_set = [value for value in arr if value not in test_set]

with open('dataset_config/train_data_list.txt','w') as f:
    f.write('\n'.join(train_set))
    f.close()

with open('dataset_config/test_data_list.txt','w') as f:
    f.write('\n'.join(test_set))
    f.close()