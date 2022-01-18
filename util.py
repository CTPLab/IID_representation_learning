import math
import random
import torch
import sys
import logging
import numpy as np
from pathlib import Path


def setup_logging(args):
    """ configure the logging document that records the 
    critical information during training and create 
    args.save_dir parameter used for saving visual and training
    results.

    Args:
        args: arguments that are implemented in args.py file 
            and determined in the training command,
            such as data_type, split_scheme, etc. 
    """

    save_msg = '{}_{}_{}_cutmix{}_batch{}_seed{}_coef{}'
    save_msg = save_msg.format(args.data_type,
                               args.split_scheme,
                               args.backbone,
                               args.cutmix,
                               args.batch_size,
                               args.seed,
                               args.loss_coef)
    if args.noise:
        save_msg += '_noise_drop{}'.format(args.noise_drop)
    if args.style:
        save_msg += '_style_drop{}'.format(args.style_drop)
    args.save_dir = Path(args.save) / args.backbone / save_msg
    args.save_dir.mkdir(parents=True, exist_ok=True)
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(str(args.save_dir / 'log'),
                                        mode='w'))
    logging.basicConfig(level=logging.INFO,
                        format=head,
                        style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))


def setup_determinism(seed):
    """ 
    Args:
        seed: the seed for reproducible randomization. 
    """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_avg_img(restyle):
    """ Compuate the average image that is used 
    to append to the input image. The code snippet 
    is copied from coach_restyle_psp.py

    Args:
        restyle: the trained restyle encoder 
            and stylgan decoder
    """

    restyle.latent_avg = restyle.decoder.mean_latent(int(1e5))[0]
    restyle.latent_avg = restyle.latent_avg.detach()

    avg_img = restyle(restyle.latent_avg.unsqueeze(0),
                      input_code=True,
                      randomize_noise=False,
                      return_latents=False,
                      average_code=True)[0]
    return avg_img.detach()


def get_learning_rate(lr_schedule,
                      tot_epoch,
                      cur_epoch):
    """ Compuate learning rate based on the initialized learning schedule,
    e.g., cosine, 1.5e-4,90,6e-5,150,0 -> ('cosine', [1.5e-4,90,6e-5,150,0]).

    Args:
        lr_schedule: the learning schedule specified in args.py
        tot_epoch: the total epoch
        cur_epoch: the current (float) epoch
    """

    # cosine
    lr_mtd = lr_schedule[0]
    # 90, 150
    ep_list = lr_schedule[1][1::2]
    # 1.5e-4,6e-5,0
    lr_list = lr_schedule[1][::2]
    assert len(ep_list) + 1 == len(lr_list)
    for start, end, lr in zip([0] + ep_list,
                              ep_list + [tot_epoch],
                              lr_list):
        if start <= cur_epoch < end:
            if lr_mtd == 'cosine':
                return lr * (math.cos((cur_epoch - start) / (end - start) * math.pi) + 1) / 2
            elif lr_mtd == 'const':
                return lr
            else:
                raise TypeError('Invalid learning method {}'.format(lr_mtd))
    raise TypeError('Invalid learning schedule {}'.format(lr_schedule))


@torch.no_grad()
def transform_input(args, X, Y,
                    is_train=False,
                    **aug_kargs):
    """ The customized data transform function including
    processing the iid latent vectors (morph and stain),
    normalizing the rxrx1 input with adding additional gaussian noise,
    cutmix argumentation, etc.

    Args:
        args: critical parameters specified in args.py
        X: input image
        Y: input label
        is_train: whether is training or evaluation
        **aug_kargs: other parameters not in args.py, e.g.,
            average image, restyle.
    """
    # the one-hot vector is meant for CutMix augmentation and ArcFace loss
    Y = torch.nn.functional.one_hot(Y, args.classes).float()

    # output the stain (style) and morph (noise) latent vectors
    if args.noise or args.style:
        avg = aug_kargs['avg_img'].detach()
        avg = avg.unsqueeze(0)
        avg = avg.repeat(X.shape[0], 1, 1, 1)
        avg = avg.float().to(X)

        X_in = torch.cat([(X - 0.5) / 0.5, avg], dim=1)
        codes, noises = aug_kargs['restyle'].encoder(X_in)

    X_lat = []
    if args.noise:
        for i in range(5):
            x_lat = noises[2*i + 1]
            assert torch.all(x_lat == noises[2*i + 2])
            X_lat = [x_lat] + X_lat
    else:
        X_lat = [None] * 5

    if args.style:
        X_lat.append(codes)

    # normalize rxrx1 input image
    if args.data_type == 'rxrx1':
        mean = X.mean(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
        std = X.std(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
        std[std == 0.] = 1.
        X = (X - mean.detach()) / std.detach()

    if not is_train:
        return [X] + X_lat, Y

    # add small gaussian noise for rxrx1
    if args.data_type == 'rxrx1':
        a = np.random.normal(1, args.pw_aug[0], (X.shape[0], X.shape[1], 1, 1))
        b = np.random.normal(0, args.pw_aug[1], (X.shape[0], X.shape[1], 1, 1))
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        X = X * a.detach().to(X) + b.detach().to(X)

    # cutmix augmentation
    if args.cutmix != 0:
        perm = torch.randperm(X.shape[0]).cuda()
        img_height, img_width = X.size()[2:]
        lambd = np.random.beta(args.cutmix, args.cutmix)

        column = np.random.uniform(0, img_width)
        row = np.random.uniform(0, img_height)
        height = (1 - lambd) ** 0.5 * img_height
        width = (1 - lambd) ** 0.5 * img_width

        r1 = round(max(0, row - height / 2))
        r2 = round(min(img_height, row + height / 2))
        c1 = round(max(0, column - width / 2))
        c2 = round(min(img_width, column + width / 2))

        if r1 < r2 and c1 < c2:
            X[:, :, r1:r2, c1:c2] = X[perm, :, r1:r2, c1:c2]
            lambd = 1 - (r2 - r1) * (c2 - c1) / (img_height * img_width)
            Y = Y * lambd + Y[perm] * (1 - lambd)

    return [X] + X_lat, Y
