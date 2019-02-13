import pickle
import os
import numpy as np
import sys
from parser import args


def normalize_text(text):
    # TODO
    # use spacy
    # stopwords removal?
    # stemming (remove s?)
    #
    return text


def import_data(pkl_path, pkl_emb_path):
    emb = None
    if pkl_emb_path is not None:
        if os.path.isfile(pkl_path) and os.path.isfile(pkl_emb_path):
            if args.verbose:
                print("loading data from {} (emb from {})".format(
                    pkl_path, pkl_emb_path))
            data = load_pickle(pkl_path)
            emb = load_pickle(pkl_emb_path)
        else:
            if args.verbose:
                print("start processing data from {}...".format(args.path_data))
            data = import_and_preprocess_data(
                args.path_data, args.eval, args.test)

            if args.verbose:
                print("loading word embeddings...")
            emb = load_word_embeddings(args.path_emb, data["word2idx"])
            if not args.test:
                dump_pickle(data, pkl_path)
                dump_pickle(emb, pkl_emb_path)
                if args.verbose:
                    print("data saved to {}\nemb saved to {}".format(
                        pkl_path, pkl_emb_path))
    else:
        if os.path.isfile(pkl_path):
            if args.verbose:
                print("loading data from {} (emb not loading)".format(pkl_path))
            data = load_pickle(pkl_path)
        else:
            if args.verbose:
                print("start processing data from {}...".format(args.path_data))
            data = import_and_preprocess_data(
                args.path_data, args.eval, args.test)
            if not args.test:
                dump_pickle(data, pkl_path)
                if args.verbose:
                    print("data saved to {}".format(pkl_path))
    return data, emb


def import_and_preprocess_data(path, bool_val=False, bool_test=False):
    def preprocess_data(data, val=False):
        lbl, text = data.strip().split(" ||| ")
        lbl_idx = get_idx(lbl2idx, lbl, val)
        text = normalize_text(text)
        text = [get_idx(word2idx, w) for w in text.split()]
        return (lbl_idx, text)

    def get_idx(vocab, key, val=False):
        if key in vocab:
            return vocab[key]
        elif val:
            return 0  # return <unk> when a label in val data did not appear in the train data
        else:
            # oov word
            # TODO : classify oov words to <unk>, <num>, <ne>, <noteng>
            idx = len(vocab)
            vocab[key] = idx
            return idx
    # oov = ["<unk>", "<num>", "<ne>","<noteng>"]
    oov = ["<unk>", "<pad>"]
    word2idx, lbl2idx = {x: i for i, x in enumerate(oov)}, {"<unk>": 0}
    with open(os.path.join(path, "topicclass_train.txt"), "r") as file:
        train = file.readlines()

    if bool_test:
        train = random.sample(train, 1000)

    train = [preprocess_data(d) for d in train]

    sent_len = [len(x[1]) for x in train]
    sorted_idx = np.argsort(sent_len)
    train = [train[idx] for idx in sorted_idx]

    if bool_val:
        with open(os.path.join(path, "topicclass_valid.txt"), "r") as file:
            val = file.readlines()
        val = [preprocess_data(d, val=True) for d in val]
        sent_len = [len(x[1]) for x in val]
        sorted_idx = np.argsort(sent_len)
        val = [val[idx] for idx in sorted_idx]
    else:
        val = None

    idx2lbl = {i: l for l, i in lbl2idx.items()}
    data = {}
    data["train"] = train
    data["val"] = val
    data["word2idx"] = word2idx
    data["lbl2idx"] = idx2lbl

    return data


def get_initialize_emb(data_dict, glove_dct, glove_vec):
    vocab_size = len(data_dict)
    glove_dim = len(glove_vec[0])
    data_emb = np.empty(shape=(vocab_size, glove_dim))
    unk = []
    for word, i in data_dict.items():
        if word in glove_dct:
            data_emb[i] = glove_vec[glove_dct[word]]
        else:
            data_emb[i] = np.random.random_sample((glove_dim,)) - 0.5
            unk.append((word, i))
    if args.verbose:
        print("{} unk words ({}) were randomly initialized".format(
            len(unk), ", ".join([x[0] for x in unk])))
    return data_emb


def load_word_embeddings(path, data_vocab):
    pkl_vec_path = path.replace(".txt", ".vec")
    pkl_dict_path = path.replace(".txt", ".dict")
    if os.path.isfile(pkl_vec_path) and os.path.isfile(pkl_dict_path):
        if args.verbose:
            print("loading embedding from pre-saved pkl")
        glove_vec = load_large_pickle(pkl_vec_path)
        glove_dict = load_pickle(pkl_dict_path)
    else:
        if args.verbose:
            print("loading embedding txt file to dict/vec")
        glove_vec, glove_dict = load_glove(path)
        dump_pickle(glove_dict, pkl_dict_path)
        dump_large_pickle(glove_vec, pkl_vec_path)

    if args.verbose:
        print(
            "initializing embedding vectors of our data from loaded off-the-shelf vectors")
    init_emb = get_initialize_emb(data_vocab, glove_dict, glove_vec)
    return init_emb


def load_glove(path='./data/glove.840B.300d.txt'):
    """Load glove vectors and dictionary from the glove text file. At the end, it saves the vector and the dictionary as pickle object to load those faster in future access."""
    glove_dict = {}
    vecs = []
    # Read in the data.
    with open(path, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            toks = line.split(' ')
            if len(toks) == 301:
                word = toks[0]
                entries = toks[1:]
                if word not in glove_dict:
                    glove_dict[word] = i
                    vecs.extend(float(x) for x in entries)
                    i += 1
    dim = len(entries)
    vecs = np.array(vecs).reshape(i, dim)
    return vecs, glove_dict


def load_pickle(path):
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def dump_pickle(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def dump_large_pickle(obj, path):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    n_bytes = sys.getsizeof(bytes_out)
    with open(path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_large_pickle(path):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(path)
    bytes_in = bytearray(0)
    with open(path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
