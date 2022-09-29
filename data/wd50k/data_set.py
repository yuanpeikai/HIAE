import json
from tqdm import tqdm

max_quals = 12
for split in ['train', 'test', 'valid']:
    fp = open("{}.txt".format(split), "w+", encoding="utf-8")
    for line in open('statements/{}.txt'.format(split)):
        line = line.rstrip('\n')
        data_line = line.split(',')

        if len(data_line) > (max_quals + 3):
            data_line = data_line[:max_quals + 3]

        data=",".join(data_line)
        fp.write(data)
        fp.write('\n')

    fp.close()
