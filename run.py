from utils.exp_utils import create_exp_dir
from utils.text_utils import MonoTextData
import argparse
import os
import torch
import time
import config
# from models.decomposed_vae import DecomposedVAE
import numpy as np
from file_io import *
from vocab import Vocabulary, build_vocab
from models.vae import TrainerVAE

def main(args):
    # if args.train:
    #     train0 = load_sent(args.train + '.0')
    #     train1 = load_sent(args.train + '.1')
    #     print('#sents of training file 0:', len(train0))
    #     print('#sents of training file 1:', len(train1))
    #
    #     if not os.path.isfile(args.vocab):
    #         build_vocab(train0 + train1, args.vocab)
    #
    # vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    # print('vocabulary size:', vocab.size)
    #
    # if args.dev:
    #     dev0 = load_sent(args.dev + '.0')
    #     dev1 = load_sent(args.dev + '.1')
    #
    # if args.test:
    #     test0 = load_sent(args.test + '.0')
    #     test1 = load_sent(args.test + '.1')
    # print("Positive dataset size:", len(train0) + len(dev0) + len(test0))
    # print("Negative dataset size:", len(train1) + len(dev1) + len(test1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    conf = config.CONFIG[args.data_name] # Need to update !!
    data_pth = "data/%s" % args.data_name
    train_data_pth0 = os.path.join(data_pth, "sentiment.train.0")
    train_data_pth1 = os.path.join(data_pth, "sentiment.train.1")
    train_class = MonoTextData(train_data_pth0, train_data_pth1, glove=True)
    train_data, train_labels = train_class.create_data_batch_labels(args.bsz, device)

    vocab = train_class.vocab
    print('Vocabulary size: %d' % len(vocab))

    val_data_pth0 = os.path.join(data_pth, "sentiment.dev.0")
    val_data_pth1 = os.path.join(data_pth, "sentiment.dev.1")
    val_class = MonoTextData(val_data_pth0, val_data_pth1, vocab=vocab, glove=True)
    val_data, val_labels = val_class.create_data_batch_labels(args.bsz, device)

    test_data_pth0 = os.path.join(data_pth, "sentiment.test.0")
    test_data_pth1 = os.path.join(data_pth, "sentiment.test.1")
    test_class = MonoTextData(test_data_pth0, test_data_pth1, vocab=vocab, glove=True)
    test_data, test_labels = test_class.create_data_batch_labels(args.bsz, device)

    save_path = '{}-{}'.format(args.save, args.data_name)
    folder_name = time.strftime("%Y%m%d-%H%M%S") + '-' + args.exp_name
    save_path = os.path.join(save_path, folder_name)
    scripts_to_save = [
        'run.py', 'models/vae.py',
        'models/base_network.py', 'config.py']
    logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
                             debug=args.debug)

    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device

    trainerVAE = TrainerVAE(train_data, val_data, test_data, train_labels, val_labels, test_labels, save_path, logging, 
    params["num_epochs"], params["log_interval"], params["warm_up"], params["kl_start"], params["vae_params"], params["lr_params"])

    trainerVAE.fit()
    # trainerVAE.train(1)
    #
    # vocab = train_data.vocab
    # print('Vocabulary size: %d' % len(vocab))
    #
    # dev_data_pth = os.path.join(data_pth, "dev_data.txt")
    # dev_feat_pth = os.path.join(data_pth, "dev_%s.npy" % args.feat)
    # dev_data = MonoTextData(dev_data_pth, True, vocab=vocab)
    # dev_feat = np.load(dev_feat_pth)
    # test_data_pth = os.path.join(data_pth, "test_data.txt")
    # test_feat_pth = os.path.join(data_pth, "test_%s.npy" % args.feat)
    # test_data = MonoTextData(test_data_pth, True, vocab=vocab)
    # test_feat = np.load(test_feat_pth)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # save_path = '{}-{}-{}'.format(args.save, args.data_name, args.feat)
    # save_path = os.path.join(save_path, time.strftime("%Y%m%d-%H%M%S"))
    # scripts_to_save = [
    #     'run.py', 'models/decomposed_vae.py', 'models/vae.py',
    #     'models/base_network.py', 'config.py']
    # logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
    #                          debug=args.debug)
    #
    # if args.text_only:
    #     train = train_data.create_data_batch(args.bsz, device)
    #     dev = dev_data.create_data_batch(args.bsz, device)
    #     test = test_data.create_data_batch(args.bsz, device)
    #     feat = train
    # else:
    #     train = train_data.create_data_batch_feats(args.bsz, train_feat, device)
    #     dev = dev_data.create_data_batch_feats(args.bsz, dev_feat, device)
    #     test = test_data.create_data_batch_feats(args.bsz, test_feat, device)
    #     feat = train_feat
    #
    # kwargs = {
    #     "train": train,
    #     "valid": dev,
    #     "test": test,
    #     "feat": feat,
    #     "bsz": args.bsz,
    #     "save_path": save_path,
    #     "logging": logging,
    #     "text_only": args.text_only,
    # }
    # params = conf["params"]
    # params["vae_params"]["vocab"] = vocab
    # params["vae_params"]["device"] = device
    # params["vae_params"]["text_only"] = args.text_only
    # params["vae_params"]["mlp_ni"] = train_feat.shape[1]
    # kwargs = dict(kwargs, **params)
    #
    # model = DecomposedVAE(**kwargs)
    # try:
    #     valid_loss = model.fit()
    #     logging("val loss : {}".format(valid_loss))
    # except KeyboardInterrupt:
    #     logging("Exiting from training early")
    #
    # model.load(save_path)
    # test_loss = model.evaluate(model.test_data, model.test_feat)
    # logging("test loss: {}".format(test_loss[0]))
    # logging("test recon: {}".format(test_loss[1]))
    # logging("test kl1: {}".format(test_loss[2]))
    # logging("test kl2: {}".format(test_loss[3]))
    # logging("test mi1: {}".format(test_loss[4]))
    # logging("test mi2: {}".format(test_loss[5]))


def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--train', type=str, default='./data/yelp/sentiment.train',
                        help='train data path')
    parser.add_argument('--dev', type=str, default='./data/yelp/sentiment.dev',
                        help='val data path')
    parser.add_argument('--test', type=str, default='./data/yelp/sentiment.test',
                        help='test data path')
    parser.add_argument('--save', type=str, default='checkpoint/ours',
                        help='directory name to save')
    parser.add_argument('--exp_name', type=str, default='',
                        help='experiment Name')
    parser.add_argument('--bsz', type=int, default=256,
                        help='batch size for training')
    parser.add_argument('--text_only', default=False, action='store_true',
                        help='use text only without feats')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    parser.add_argument('--feat', type=str, default='glove',
                        help='feat repr')
    parser.add_argument('--vocab', type=str, default='./tmp/yelp.vocab')
    parser.add_argument('--embedding', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--dim_emb', type=int, default=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
