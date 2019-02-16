import os
import torch
import pickle
import random
import numpy as np
import time

from models import CNN
from utils import import_data, plot_figure
from parser import args
from Dataset import Dataset
from tqdm import tqdm

random.seed(11)


def trainModel(args, model, loss_fn, optim, data_train, data_val=None):
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
            if args.verbose and (i+1) % args.report == 0:
                print("batch {0}/{1}: loss {2:.3f}/ acc {3:.3f}".format(i+1, len_data, total_loss/total_samples, total_correct/total_samples))
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
    best_ep, best_acc, best_loss = -1, 0, 0
    plot_res = []

    for ep in range(args.epochs):
        start = time.time()
        train_loss, train_acc = trainEpoch()
        val_loss, val_acc = trainEpoch(bool_eval=True)
        print("ep {0}: t {1:.3f}/v {2:.3f} (t {3:.3f}/v {4:.3f}) ({5:.2f}s)".format(ep +
                                                                               1, train_acc, val_acc, train_loss, val_loss, (time.time()-start)))
        plot_res.append([train_acc, train_loss, val_acc, val_loss])
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            best_ep = ep
            print("best model found!")

    if not args.test:
        plot_figure(args.path_savedir+"{}_{}".format(args.model,args.epochs), plot_res)
    print("\nbest epoch: {}\nbest_acc: {}\nbest_loss: {}".format(best_ep, best_acc, best_loss))



def predict(model, data):
    preds = []
    for i in range(len(data)):
        batch, label = data[i]
        output = model(batch)
        pred = output.argmax(1)
        preds.extend(pred)
    return preds


def save_prediction(path, data, idx2lbl):
    import codecs
    predF = codecs.open(path, 'w', 'utf-8')
    for label in data:
        predF.write(idx2lbl[int(label)]+"\n")
        predF.flush()


if __name__ == "__main__":
    pkl_path = args.path_data + \
        "processed_{}.pkl".format("eval" if args.eval else "train")
    pkl_emb_path = pkl_path.replace(".pkl", ".emb.pkl") if args.emb else None
    data, emb = import_data(pkl_path, pkl_emb_path)
    # train_data = data["train"]
    # val_data = data["val"]

    trainData = Dataset(data["train"], args.batch_size, args.cuda)
    valData = Dataset(data["val"], args.batch_size,
                      args.cuda) if args.eval else None

    if args.emb:
        assert emb.shape[1] == args.emb_dim
        emb = torch.Tensor(emb)

    if args.model == "cnn":
        model = CNN(len(data["word2idx"]), args.emb_dim,
                    args.out_dim, args.window_dim, len(data["lbl2idx"]), args.dp, emb)

    if args.fix_emb:
        model.embedding.weight.requires_grad = False

    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.l2)

    if args.cuda:
        model.cuda()
    trainModel(args, model, loss, optim, trainData, valData)

    preds_val = predict(model, valData)
    save_prediction(args.path_savedir+"{}_{}.val".format(args.model,
                                                            args.epochs), preds_val, data["idx2lbl"])

    if args.submit:
        testData = Datset(data["test"], args.batch_size, args.cuda)
        preds_test = predict(model, testData)
        save_prediction(args.path_savedir+"{}_{}.test".format(args.model,
                                                                args.epochs), preds_test, data["idx2lbl"])
