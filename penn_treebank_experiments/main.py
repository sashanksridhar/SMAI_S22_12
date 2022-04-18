import os
import hashlib
import torch
from data import Corpus
from utils import batchify
# from splitcross import SplitCrossEntropyLoss



if __name__ == '__main__':

    data_file = 'data/penn/'
    eval_batch_size = 10
    test_batch_size = 1
    batch_size = 80
    cuda = False
    criterion = None

    fn = 'corpus.{}.data'.format(hashlib.md5(data_file.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = Corpus(data_file)
        torch.save(corpus, fn)


    train_data = batchify(corpus.train, batch_size, cuda)
    val_data = batchify(corpus.valid, eval_batch_size, cuda)
    test_data = batchify(corpus.test, test_batch_size, cuda)

