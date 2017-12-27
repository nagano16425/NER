"""
Title:Proposed1
Detail:Prepare for Proposed1
Design:Naonori Nagano
Date:2017/12/27
"""

import collections, os
import numpy as np

import config_model as cfg

def _add_extra_keys(data):
    keys = ["<eos>", "<unk>"]
    for s in split:
        v = len(data[s]["word_to_id"])
        for i, k in enumerate(keys):
            data[s]["word_to_id"][k] = i + v
            data[s]["id_to_word"][i + v] = k

def _hierarchy_to_i(data):
    toi = data["tag_to_id"]
    for s in split:
        data[s]["hierarchy_i"] = [[[toi[t] for t in ts] for ts in tss] for tss in data[s]["hierarchy"]]

def _word_to_i(data, k):
    toi = data[k]["word_to_id"]
    for s in split:
        data[s]["word_i"] = [[toi[w] if w in toi else toi["<unk>"] for w in ws] for ws in data[s]["word"]]

def _pad(i_list, i_list_pad, len_list, input_size):
    for i, (t, l) in enumerate(zip(i_list, len_list)):
        length = min([l, input_size - 1])
        i_list_pad[i][:length] = np.array(t[:length], dtype='int32')
    return i_list_pad

def _pad_word(data, input_size, k):
    for s in split:
        i_list = data[s]["word_i"]
        i_list_pad = np.zeros((len(i_list), input_size), dtype='int32') + data[k]["word_to_id"]["<eos>"]
        len_list = [len(ws) for ws in i_list]
        data[s]["word_len"] = len_list
        data[s]["word_i"] = _pad(i_list, i_list_pad, len_list, input_size)

def _pad_hierarchy(data, input_size):
    for s in split:
        i_list = data[s]["hierarchy_i"]
        i_list_pad = np.cast[np.int32](np.zeros([len(i_list), input_size, 8]) + data["tag_to_id"]["<EOL>"])
        len_list = [len(ws) for ws in i_list]
        data[s]["hierarchy_len"] = len_list
        data[s]["hierarchy_i"] = _pad(i_list, i_list_pad, len_list, input_size)

def _make_mask(data, input_size):
    for s in split:
        mask = np.where(data[s]["hierarchy_i"] == data["tag_to_id"]["<EOL>"], 0, 1)
        mask = mask.reshape([-1, 8])
        _mask = np.ones([mask.shape[0], 1])
        data[s]["mask_tag"] = np.cast[np.float32](np.c_[_mask, mask].reshape([-1, input_size, 9])[:,:,:8])

def _make_mask_sent(data, input_size, k):
    for s in split:
        data[s]["mask_sent"] = np.where(data[s]["word_i"] == data[k]["word_to_id"]["<eos>"], 0, 1)

def _get_tag_vocab_size(data):
    data["tag_size"] = len(data['tag_to_id'])
    for s in split:
        data[s]["vocab_size"] = len(data[s]['word_to_id'])

def prepare(input_size, k, data_path="data/"):
    data = np.load(data_path + "train_data_" + cfg.train_path + ".npy")[None][0]
    _add_extra_keys(data)
    _word_to_i(data, k)
    _hierarchy_to_i(data)
    _pad_word(data, input_size, k)
    _pad_hierarchy(data, input_size)
    _make_mask(data, input_size)
    _make_mask_sent(data, input_size, k)
    _get_tag_vocab_size(data)
    return data

split=["train", "test"]
if __name__ == "__main__":
    data = prepare(input_size)

