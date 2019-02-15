import argparse

# parse arg
parser = argparse.ArgumentParser(description='Model parameters.')
parser.add_argument('-model', type=str, action='store', default='cnn')
parser.add_argument('-epochs', type=int, action='store', default=10)

parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-eval', action='store_true', default=True)
parser.add_argument('-submit', action='store_true', default=False)
parser.add_argument('-verbose', action='store_true', default=False)
parser.add_argument('-cuda', action='store_true', default=False)
parser.add_argument('-report', type=int, default=4000)

parser.add_argument('-emb', action='store_true', default=True)
parser.add_argument('-fix_emb', action='store_true', default=False)
parser.add_argument('-emb_dim', type=int, action='store', default=300)
parser.add_argument('-out_dim', type=int, action='store', default=100)
parser.add_argument('-window_dim', type=int, action='store', default=4)

parser.add_argument('-batch_size', type=int, action='store', default=16)
parser.add_argument('-dp', type=float, action='store', default=0.1)
parser.add_argument('-lr', type=float, action='store', default=0.00005)
parser.add_argument('-l2', type=float, action='store', default=0.0001)

parser.add_argument('-path_data', type=str, action='store',
                    default='data/topicclass/')
parser.add_argument('-path_emb', type=str, action='store',
                    default='data/glove.840B.300d.txt')
parser.add_argument('-path_savedir', type=str, action='store', default='res/')

args = parser.parse_args()
