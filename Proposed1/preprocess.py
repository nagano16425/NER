"""
Title:Proposed1
Detail:pre-training
Input:Traffic**.iob2
Design:Naonori Nagano
Date:2017/12/27
"""

import collections, os, sys
import numpy as np

def _chunk_to_tag(c):
    t = c.split("-")
    if t[0] in {"B", "I"}:
        return t[1]
    return "<eos>"

def _del_from_list(target, s):
    while s in target: target.remove(s)
    return target

def _read_iob_data(path):
    phrases = open(path, "r").read().strip().split(" ")
    lines = [p.strip().split("\n") for p in phrases]
    return _del_from_list(lines, [""])

def _read_data(path, keys=["word", "chunk", "chunk_tag"]):
    data = _read_iob_data(path)
    word_chunk = [[wc.split("\t") for wc in l] for l in data]
    word = [[wc[0] for wc in wcs] for wcs in word_chunk]
    chunk = [[wc[1] for wc in wcs] for wcs in word_chunk]
    tag = [[_chunk_to_tag(c) for c in cs] for cs in chunk]
    return {k:v for k, v in zip(keys, [word, chunk, tag])}

def _get_hierarchy(tag, d, path):
    d = _make_hierarchy_dict(path)
    res = [tag]
    for i in range(7):
        tag = d[tag]
        res.append(tag)
    return res

def _make_hierarchy_dict(path):
    f = open(path, 'r').read().strip().split('\n')
    f = [f.split() for f in f]
    d = {b:a for a, b in f}
    return d

def _convert_hierarchy(targets, path):
    hierarchy_dict = _make_hierarchy_dict(path)
    hierarchy = [[_get_hierarchy(c, hierarchy_dict, path) for c in cs] for cs in targets]
    return hierarchy, hierarchy_dict

# If it is necessary to divide randomly from data
def _split_data_random(data, ratio=0.5):
    data_size = len(data["word"])
    idx = np.arange(data_size)
    np.random.seed(1234)
    np.random.shuffle(idx)
    n = int(data_size * ratio)
    train_data = {k:[v[i] for i in idx[:n]] for k, v in data.items() if k in ['word', 'hierarchy']}
    test_data = {k:[v[i] for i in idx[n:]] for k, v in data.items() if k in ['word', 'hierarchy']}
    return train_data, test_data

# Divide from data
def _split_data_standard(train, test):
    train_size = len(train["word"])
    test_size = len(test["word"])
    train_idx = np.arange(train_size)
    test_idx = np.arange(test_size)
    train_data = {k:[v[i] for i in train_idx] for k, v in train.items() if k in ['word', 'hierarchy']}
    test_data = {k:[v[i] for i in test_idx] for k, v in test.items() if k in ['word', 'hierarchy']}
    return train_data, test_data

# Making tag dictionary
def _make_tag_dict(hierarchy_dict):
    k, v = hierarchy_dict.keys(), hierarchy_dict.values()
    tags = set(k) | set(v)
    tag_to_id = {k:i for i, k in enumerate(tags)}
    id_to_tag = {vv:kk for kk, vv in tag_to_id.items()}
    return tag_to_id, id_to_tag
    
def _flatten(l):
    return [e for i in l for e in i]

def _make_word_dict(word_list):
    word_freq = collections.Counter(word_list)
    word_freq = collections.OrderedDict(sorted(word_freq.items(), key = lambda x:(-x[1], x[0])))
    word_to_id = {k:i for i, k in enumerate(word_freq.keys())}
    id_to_word = {vv:kk for kk, vv in word_to_id.items()}
    return word_to_id, id_to_word

if __name__ == "__main__":
    split = ["train", "test"]

    data_path = "datasets/"
    train_path = str(sys.argv[1])
    test_path = str(sys.argv[2])
    save_path = "data/"
    hierarchy_path = "ror_hierarchy_complete.txt"
    
    data = _read_data(data_path + "Traffic.iob2")
    data["hierarchy"], hierarchy_dict = _convert_hierarchy(data["chunk"], data_path + hierarchy_path)
    
    data1 = _read_data(data_path + "Traffic" + train_path +".iob2")
    data2 = _read_data(data_path + "Traffic" + test_path +".iob2")
    data1["hierarchy"], _ = _convert_hierarchy(data1["chunk"], data_path + hierarchy_path)
    data2["hierarchy"], _ = _convert_hierarchy(data2["chunk"], data_path + hierarchy_path)

    train_data = {k:{} for k in split}
    train_data["hierarchy_dict"] = hierarchy_dict
    train_data["train"], train_data["test"] = _split_data_standard(data1, data2)
    # train_data["train"], train_data["test"] = _split_data_random(data)
    train_data["tag_to_id"], train_data["id_to_tag"] = _make_tag_dict(hierarchy_dict)
    for s in split:
        train_data[s]["word_to_id"], train_data[s]["id_to_word"] = _make_word_dict(_flatten(train_data[s]['word']))

    np.save(save_path + "data", data)
    np.save(save_path + "train_data_" + train_path, train_data)

    """ how to load
    data = np.load(data_path + "train_data_" + train_path + ".npy")[None][0]
    """

