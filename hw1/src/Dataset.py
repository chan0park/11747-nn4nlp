import math
import random
import torch
from torch.autograd import Variable


class Dataset(object):
    def __init__(self,  data, batch_size, cuda, volatile=False):
        self.data = [x[1] for x in data]
        self.label = [x[0] for x in data]
        self.cuda = cuda
        self.batch_size = batch_size
        self.num_batch = math.ceil(len(self.data)/batch_size)
        self.volatile = volatile

    def __len__(self):
        return self.num_batch

    def _batchify(self, data, align_right=False):
        lengths = [len(x) for x in data]
        max_length = max(lengths)
        if not align_right:
            data = [x+[1]*(max_length-lengths[i]) for i, x in enumerate(data)]
        else:
            data = [[1]*(max_length-lengths[i])+x for i, x in enumerate(data)]
        return data

    def __getitem__(self, index):
        def wrap(data):
            data = torch.LongTensor(data)
            if self.cuda:
                data = data.cuda()
            return Variable(data, volatile=self.volatile)

        assert index < self.num_batch, "index exceeding the number of batch"
        dataBatch = self._batchify(
            self.data[index*self.batch_size:(index+1)*self.batch_size],
            align_right=False)
        labelBatch = self.label[index *
                                self.batch_size:(index+1)*self.batch_size]

        return wrap(dataBatch), wrap(labelBatch)
