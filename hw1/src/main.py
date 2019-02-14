import os
import torch
import pickle
import random
import numpy as np
import time

from models import CNN
from utils import import_data
from parser import args
from Dataset import Dataset
from tqdm import tqdm

random.seed(11)


def trainModel(model, loss_fn, optim, epochs, data_train, data_val=None):
    def trainEpoch(bool_eval=False):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        data = data_train
        if bool_eval:
            model.eval()
            data = data_val

        if args.test and not bool_eval:
            len_data = 30
        else:
            len_data = len(data)

        rand_order = torch.randperm(len_data)
        for i in range(len_data):
            batch, label = data[rand_order[i]]
            num_sample = len(label)

            if not bool_eval:
                model.zero_grad()
            output = model(batch)
            loss = loss_fn(output, label)
            predict = output.argmax(1)

            total_correct += int(sum(predict == label))
            total_loss += loss.data
            total_samples += num_sample

            if not bool_eval:
                loss.backward()
                optim.step()

        if bool_eval:
            model.train()

        return total_loss/total_samples, total_correct/total_samples

    print(model)
    model.train()
    best_ep, best_acc = -1, 0
    for ep in range(epochs):
        start = time.time()
        train_loss, train_acc = trainEpoch()
        val_loss, val_acc = trainEpoch(bool_eval=True)
        print("ep {0}: t {1:.3f}/v {2:.3f} (t {3:.3f}/v {4:3f}) ({5}s)".format(ep +
                                                                               1, train_acc, val_acc, train_loss, val_loss, (time.time()-start)))
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep = ep
            print("best model found!")


def predict(model, data):
    pass


def save_prediction(path, data, idx2lbl):
    pass


if __name__ == "__main__":
    pkl_path = args.path_data + \
        "processed_{}.pkl".format("eval" if args.eval else "train")
    pkl_emb_path = pkl_path.replace(".pkl", ".emb.pkl") if args.emb else None
    data, emb = import_data(pkl_path, pkl_emb_path)
    train_data = data["train"]
    val_data = data["val"]

    if args.emb:
        assert emb.shape[1] == args.emb_dim
        emb = torch.Tensor(emb)

    if args.model == "cnn":
        model = CNN(len(data["word2idx"]), args.emb_dim,
                    args.out_dim, args.window_dim, len(data["lbl2idx"]), emb)

    if args.fix_emb:
        model.embedding.weight.requires_grad = False

    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainData = Dataset(data["train"], args.batch_size, args.cuda)
    valData = Dataset(data["val"], args.batch_size,
                      args.cuda) if args.eval else None

    trainModel(model, loss, optim, args.epochs, trainData, valData)

    raise
    prediction_val = predict(model, data["val"])
    save_prediction(args.path_savedir+"{}_{}.val".format(args.model,
                                                         args.epochs), prediction_val, idx2lbl)

    if args.submit:
        prediction_test = predict(model, data["test"])
        save_prediction(args.path_savedir+"{}_{}.test".format(args.model,
                                                              args.epochs), prediction_test, idx2lbl)
