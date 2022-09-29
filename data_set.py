import json
from tqdm import tqdm

for split in ['train', 'test', 'valid']:
    fp = open("./data/wikipeople/{}.txt".format(split), "w+", encoding="utf-8")
    for line in open('./data/wikipeople/n-ary_{}.json'.format(split)):
        line = line.rstrip("\n")
        data_line = json.loads(line)
        data = ""
        for index, (k, v) in enumerate(data_line.items()):
            if index == 0:
                data += v + "," + k[:-2]
            elif index == 1:
                data += "," + v
            elif (k != "N" and len(v) == 1):
                data += "," + k + "," + v[0]
            elif (k != "N" and len(v) == 2):
                data += "," + k + "," + v[0] + "," + k + "," + v[1]
        fp.write(data)
        fp.write('\n')

    fp.close()

# clean wikipeople data


with open('./data/wikipeople_clean/n-ary_train.json', 'r') as f:
    raw_trn = []
    for line in f.readlines():
        raw_trn.append(json.loads(line))

with open('./data/wikipeople_clean/n-ary_test.json', 'r') as f:
    raw_tst = []
    for line in f.readlines():
        raw_tst.append(json.loads(line))

with open('./data/wikipeople_clean/n-ary_valid.json', 'r') as f:
    raw_val = []
    for line in f.readlines():
        raw_val.append(json.loads(line))


def _conv_to_our_format_(data, filter_literals, dataSet):
    fp = open(dataSet, "w+", encoding="utf-8")

    dropped_statements = 0
    dropped_quals = 0
    for datum in tqdm(data):
        try:
            conv_datum = []

            # Get head and tail rels
            head, tail, rel_h, rel_t = None, None, None, None
            for rel, val in datum.items():
                if rel[-2:] == '_h' and type(val) is str:
                    head = val
                    rel_h = rel[:-2]
                if rel[-2:] == '_t' and type(val) is str:
                    tail = val
                    rel_t = rel[:-2]
                    if filter_literals and "http://" in tail:
                        dropped_statements += 1
                        raise Exception

            assert head and tail and rel_h and rel_t, f"Weird data point. Some essentials not found. Quitting\nD:{datum}"
            assert rel_h == rel_t, f"Weird data point. Head and Tail rels are different. Quitting\nD: {datum}"

            # Drop this bs
            datum.pop(rel_h + '_h')
            datum.pop(rel_t + '_t')
            datum.pop('N')
            # conv_datum += [head, rel_h, tail]
            conv_datum = head + "," + rel_h + "," + tail

            # Get all qualifiers
            for k, v in datum.items():
                for _v in v:
                    if filter_literals and "http://" in _v:
                        dropped_quals += 1
                        continue
                    # conv_datum += [k, _v]
                    conv_datum += "," + k + "," + _v

            fp.write(conv_datum)
            fp.write('\n')

        except Exception:
            continue
    print(f"\n Dropped {dropped_statements} statements and {dropped_quals} quals with literals \n ")

    fp.close()


_conv_to_our_format_(raw_trn, True, "./data/wikipeople_clean/train.txt")

_conv_to_our_format_(raw_tst, True, "./data/wikipeople_clean/test.txt")

_conv_to_our_format_(raw_val, True, "./data/wikipeople_clean/valid.txt")
