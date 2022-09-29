def find(dataset):
    if (dataset == 'wikipeopel'):
        splits = ['train', 'test', 'valid']
    else:
        splits = ['train', 'test']

    max_quals = 0
    for split in splits:
        for line in open('./data/{}/{}.txt'.format(dataset, split)):
            line = line.rstrip("\n")
            text = line.strip().split(',')
            max_quals = max(max_quals, (len(text) - 3))

    print("max_quals is {}".format(max_quals))


if __name__ == '__main__':
    find('jf17k_clean')
