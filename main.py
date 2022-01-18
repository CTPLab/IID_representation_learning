#!/usr/bin/env python3

import logging
import csv
import time
import torch
import torchvision.utils as tv_utils
import numpy as np
from argparse import Namespace
from pathlib import Path

import dataset as dataset
from args import parse_args
from restyle.models.psp import pSp
from model import ModelAndLoss
from util import transform_input, get_learning_rate, setup_logging, setup_determinism, compute_avg_img


@torch.no_grad()
def infer(args,
          data_loader,
          model,
          restyle,
          avg_img,
          csvwriter=None):
    """ The model inference during the training

    Args:
        args: critical parameters specified in args.py
        data_loader: the val, test (ood_test) or id_test data loader 
            called via wilds api
        model: the molecular prediction model
        restyle: the trained Restyle auto-encoder
        avg_img: the average image that appends to the input image of
            Restyle auto-encoder
        csvwriter: the csvwriter buffer for recording the results 
            line by line, the saved csv is compatible to wilds evaluation
    """

    model.eval()
    tic = time.time()
    # the dict storing the data numbers for each group type
    grp_num = dict()
    # the dict storing the correct predictions for each group type
    grp_cor = dict()
    # the total correct predictions
    correct = 0.
    # the total amount of data
    total = 0.
    for i, (X, Y, meta) in enumerate(data_loader):
        X = X.cuda()
        X = transform_input(args, X, Y,
                            is_train=False,
                            restyle=restyle,
                            avg_img=avg_img)[0]
        y = model.eval_forward(X).cpu()
        correct += (y.argmax(dim=-1) == Y).sum().numpy()
        total += Y.shape[0]
        # record the wilds compatible predicted class
        # line by line
        if csvwriter is not None:
            csv_res = np.expand_dims(y.argmax(dim=-1).numpy(), axis=1)
            csvwriter.writerows(csv_res.tolist())

        # record the stratified results based on cell type for rxrx1
        # and tumor regions for scrc
        groups = meta[:, 0] if args.data_type == 'rxrx1' else Y
        for id, grp in enumerate(groups):
            grp = int(grp.numpy())
            if grp not in grp_num:
                grp_num[grp] = 0
            if grp not in grp_cor:
                grp_cor[grp] = 0

            grp_num[grp] += 1
            grp_cor[grp] += (y[id].argmax() == Y[id]).numpy()

        if (i + 1) % args.disp_batches == 0:
            logging.info('Infer Iter: {:4d}  ->  speed: {:6.1f}'.format(
                i + 1, args.disp_batches * args.eval_batch_size / (time.time() - tic)))
            tic = time.time()

    msg = 'Eval: acc: '
    tot_num = 0
    tot_cor = 0
    for grp in sorted(grp_num.keys()):
        num = grp_num[grp]
        cor = grp_cor[grp]
        tot_num += num
        tot_cor += cor
        msg += '{}: {:6f}| '.format(grp, cor / num)
    acc = tot_cor / tot_num
    acc1 = correct / total
    # acc and acc1 should be identical
    # tot_num and total should be identical
    # just for sanity check
    logging.info(
        msg + '({:.2%}), ({:.2%}), {}, {}'.format(acc, acc1, tot_num, total))
    return acc


def train(args,
          data_loader,
          model,
          restyle,
          avg_img):
    """ The model inference during the training

    Args:
        args: critical parameters specified in args.py
        data_loader: the train, val, test (ood_test) or id_test data loader 
            called via wilds api
        model: the molecular prediction model
        restyle: the trained Restyle auto-encoder
        avg_img: the average image that appends to the input image of
            Restyle auto-encoder
    """
    # split the data_loader list
    train_loader, val_loader, test_loader = data_loader[0], data_loader[1], data_loader[2]
    msg = 'Data size for Train: {}, Val: {}, Test: {}'.format(len(train_loader),
                                                              len(val_loader),
                                                              len(test_loader))
    if args.data_type == 'rxrx1':
        id_test_loader = data_loader[3]
        msg += ' ID_Test: {}'.format(len(id_test_loader))
    logging.info(msg)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0,
                                 weight_decay=args.weight_decay)

    if args.load is not None:
        best_acc = infer(args,
                         val_loader,
                         model,
                         restyle,
                         avg_img)
    else:
        best_acc = 0
    acc_test = 0

    for epoch in range(args.start_epoch, args.epochs):
        logging.info('Train: epoch {}'.format(epoch))
        model.train()
        optimizer.zero_grad()

        cum_loss = 0
        cum_acc = 0
        cum_count = 0
        tic = time.time()
        # train the model for one epoch
        for i, (X, Y, meta) in enumerate(train_loader):
            # update the learning rate for each step
            lr = get_learning_rate(args.lr,
                                   args.epochs,
                                   epoch + i / len(train_loader))
            for g in optimizer.param_groups:
                g['lr'] = lr

            X = X.cuda()
            Y = Y.cuda()
            X, Y = transform_input(args, X, Y,
                                   is_train=True,
                                   restyle=restyle,
                                   avg_img=avg_img)
            loss, acc = model.train_forward(X, Y)
            loss.backward()
            if (i + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            cum_count += 1
            cum_loss += loss.item()
            cum_acc += acc
            if (i + 1) % args.disp_batches == 0:
                logging.info('Epoch: {:3d} Iter: {:4d}  ->  speed: {:6.1f}   lr: {:.9f}   loss: {:.6f}   acc: {:.6f}'.format(
                    epoch, i + 1, cum_count * args.batch_size /
                    (time.time() - tic), optimizer.param_groups[0]['lr'],
                    cum_loss / cum_count, cum_acc / cum_count))
                cum_loss = 0
                cum_acc = 0
                cum_count = 0
                tic = time.time()

        # run the inference stage
        acc = infer(args,
                    val_loader,
                    model,
                    restyle,
                    avg_img)
        if acc > best_acc:
            best_acc = acc
            logging.info('Saving best to {} with score {}'.
                         format(args.save_dir, best_acc))
            if args.save_model:
                torch.save(model.state_dict(),
                           str(Path(args.save_dir) / 'best_{}.pth'.format(epoch)))
            acc_test = infer(args,
                             test_loader,
                             model,
                             restyle,
                             avg_img)
            logging.info('Test score {}'.format(acc_test))

            if args.data_type == 'rxrx1':
                acc_id_test = infer(args,
                                    id_test_loader,
                                    model,
                                    restyle,
                                    avg_img)
                logging.info('Id_Test score {}'.format(acc_id_test))

    best_msg = 'Best val acc {:.2%} test acc {:.2%}'.format(
        best_acc, acc_test)
    if args.data_type == 'rxrx1':
        best_msg += ' id_test acc {:.2%}'.format(acc_id_test)
    logging.info(best_msg)


@torch.no_grad()
def evaluate(args,
             data_loader,
             model,
             restyle,
             avg_img):
    """ Evaluate the trained model, output the overall prediction 
    accuracy and generate the csv files that are compatbile to wilds evaluation.

    Args:
        args: critical parameters specified in args.py
        data_loader: the val, test (ood_test) or id_test data loader 
            called via wilds api
        model: the trained molecular prediction model
        restyle: the trained Restyle auto-encoder
        avg_img: the average image that appends to the input image of
            Restyle auto-encoder
    """

    # split the data_loader list
    _, val_loader, test_loader = data_loader[0], data_loader[1], data_loader[2]
    data_msg = 'Data size for Val: {}, Test: {}'.format(len(val_loader),
                                                        len(test_loader))
    if args.data_type == 'rxrx1':
        id_test_loader = data_loader[3]
        data_msg += ' ID_Test: {}'.format(len(id_test_loader))
    print(data_msg)

    # record the overall prediction accuracies,
    # create the csv file for wilds evaluation
    # for val, test or id_test data
    csv_list = ['val', 'test']
    if args.data_type == 'rxrx1':
        csv_list += ['id_test']
    best_msg = 'Best '
    for csv_id, csv_nm in enumerate(csv_list):
        file_name = '{}_split:{}_seed:{}_epoch:best_pred.csv'.format(
            args.data_type,
            csv_nm,
            args.seed)
        with open(str(args.save_dir / file_name), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            acc = infer(args,
                        data_loader[1 + csv_id],
                        model,
                        restyle,
                        avg_img,
                        csvwriter)
            best_msg += '{} acc {:.2%}: '.format(csv_nm, acc)
    print(best_msg)


@torch.no_grad()
def recon(save_dir,
          data_loader,
          restyle,
          avg_img):
    """ Reconstruct the input image with Restyle auto-encoder.

    Args:
        save_dir: the folder path storing the fake image
        data_loader: the train, val, test (ood_test) or id_test data loader 
            called via wilds api
        restyle: the trained Restyle auto-encoder
        avg_img: the average image that appends to the input image of
            Restyle auto-encoder
    """

    for dt_id, dt_loader in enumerate(data_loader):
        for (X, _, _) in dt_loader:
            X = X[:, :3].cuda()
            avg = avg_img.unsqueeze(0)
            avg = avg.repeat(X.shape[0], 1, 1, 1)
            avg = avg.float().to(X)
            # concatenate the average image to input
            X_inp = torch.cat([(X - 0.5) / 0.5, avg], dim=1)
            # output styles (codes) and noises
            codes, noises = restyle.encoder(X_inp)
            # the same latent_avg generated by
            # restyle.decoder.mean_latent(int(1e5))[0],
            # which is also used for generate avg_img.
            latent_avg = restyle.latent_avg.repeat(codes.shape[0], 1, 1)
            # X_out is the reconstructed image
            X_out = restyle.decoder([codes + latent_avg.to(codes)],
                                    noise=noises,
                                    input_is_latent=True)[0]
            # Since the input image is normalized by (image - 0.5) / 0.5,
            # now we need to convert the pixel interval back to [0, 1]
            X_out = (X_out + 1) / 2
            X_out[X_out < 0] = 0
            X_out[X_out > 1] = 1

            # Visually compare the reconstructed and gt images side by side
            X_merge = torch.zeros([X.shape[0] * 2,
                                   X.shape[1],
                                   X.shape[2],
                                   X.shape[3]]).to(X)

            bb_col = [1, 0, 0]
            bb_len = 4
            # add red bbox to gt image
            for i in range(3):
                X[:, i, :bb_len, :] = bb_col[i]
                X[:, i, -bb_len:, :] = bb_col[i]
                X[:, i, :, :bb_len] = bb_col[i]
                X[:, i, :, -bb_len:] = bb_col[i]

            X_merge[::2] = X
            X_merge[1::2] = X_out

            filename = Path(save_dir) / \
                '{}.jpg'.format(dt_id)
            tv_utils.save_image(X_merge.cpu().float(),
                                str(filename),
                                nrow=8,
                                padding=2)
            print(str(filename))
            break


@torch.no_grad()
def synth(save_dir, batch_size, restyle):
    """ Synthesize fake images with StyleGAN decoder,
    where the input are random gaussian noise.

    Args:
        save_dir: the folder path storing the fake image
        restyle: the trained Restyle auto-encoder
    """

    latent = torch.randn(batch_size, 512).cuda()
    fake = restyle.decoder([latent])[0]
    fake = (fake + 1) / 2
    fake[fake < 0] = 0
    fake[fake > 1] = 1
    filename = Path(save_dir) / 'fake.jpg'
    tv_utils.save_image(fake.cpu().float(),
                        str(filename),
                        nrow=8,
                        padding=2)


def main(args):
    """ The main function running the experiments 
    reported in the paper.
    if args.mode == 'train', then train the molecular predictor
        with and without IID representation integration.
    elif args.mode == 'evaluate', run the model evaluation 
        and generate the wilds compatible csv results, which
        can be used for leaderboard submission.
    elif args.mode == 'recon', save the image reconstruction results
        achieved by Restyle auto-encoder
    elif args.mode == 'synth', save the images synthesized with 
        StyleGAN decoder.

    Args:
        args: critical parameters specified in args.py.
    """

    # load the data via wilds api
    data_loader = dataset.get_dataloader(args)

    # initialize the restyle model parameters and weights
    ckpt_path = Path(args.checkpoint_path) / \
        '{}_{}_iteration_90000.pt'.format(args.data_type, args.split_scheme)
    restyle_ckpt = torch.load(str(ckpt_path), map_location='cpu')
    restyle_opts = restyle_ckpt['opts']
    restyle_opts.update({'checkpoint_path': str(ckpt_path)})
    restyle_opts = Namespace(**restyle_opts)
    logging.info('Restyle auto-encoder opts:\n{}'.format(str(restyle_opts)))
    # load the restyle model parameters and weights
    # switch the restyle model to evaluation mode, i.e.,
    # freezing the weight during molecular prediction training
    affine = True if args.mode in ('recon', 'synth') else False
    restyle = pSp(restyle_opts, affine).cuda().eval()
    logging.info('Restyle auto-encoder with loaded weights:\n{}'.
                 format(str(restyle)))

    # initialize the molecular prediction model
    model = ModelAndLoss(args,
                         restyle_opts.output_size,
                         restyle.decoder.style_dim).cuda()
    logging.info('Model:\n{}'.format(str(model)))
    if args.load is not None:
        logging.info('Loading model from {}'.format(args.load))
        model.load_state_dict(torch.load(str(args.load)))

    avg_img = compute_avg_img(restyle)
    if args.mode == 'train':
        train(args, data_loader, model, restyle, avg_img)
    elif args.mode == 'evaluate':
        evaluate(args, data_loader, model, restyle, avg_img)
    elif args.mode == 'recon':
        recon(args.save_dir, data_loader, restyle, avg_img)
    elif args.mode == 'synth':
        synth(args.save_dir, args.batch_size, restyle)
    else:
        assert 0


if __name__ == '__main__':
    args = parse_args()
    setup_determinism(args.seed)
    setup_logging(args)
    main(args)
