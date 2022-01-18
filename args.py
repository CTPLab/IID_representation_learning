import random
from argparse import ArgumentParser
from pathlib import Path


def lr_type(x):
    x = x.split(',')
    return x[0], list(map(float, x[1:]))


def parse_args():
    parser = ArgumentParser()
    # basic training hyper-parameters
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='global seed (for weight initialization, data sampling, etc.). '
                        'If not specified it will be randomized (and printed on the log)')
    parser.add_argument('-m', '--mode',
                        default='train',
                        choices=('train', 'evaluate', 'recon', 'synth'))
    parser.add_argument('-e', '--epochs',
                        default=150,
                        type=int)
    parser.add_argument('--lr',
                        type=lr_type,
                        default=('cosine', [1.5e-4]),
                        help='learning rate values and schedule given in format: schedule,value1,epoch1,value2,epoch2,...,value{n}. '
                        'in epoch range [0, epoch1) initial_lr=value1, in [epoch1, epoch2) initial_lr=value2, ..., '
                        'in [epoch{n-1}, total_epochs) initial_lr=value{n}, '
                        'in every range the same learning schedule is used. Possible schedules: cosine, const')
    parser.add_argument('--backbone',
                        default='resnet50')

    # data related hyper-parameters
    parser.add_argument('--data',
                        type=Path,
                        help='path to the data root.')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['rxrx1', 'scrc'],
                        help='experiments to run')
    parser.add_argument('--split_scheme',
                        type=str,
                        choices=['official', '012', '120', '201'],
                        help='official for rxrx1, the rests for scrc')
    parser.add_argument('--classes',
                        type=int,
                        help='number of classes predicting by the network')
    parser.add_argument('--batch_size',
                        type=int,
                        default=24)
    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--num-data-workers',
                        type=int,
                        default=10,
                        help='number of data loader workers')

    # model related hyper-parameters
    parser.add_argument('--save',
                        type=str,
                        help='path for the checkpoint with best accuracy. '
                        'Checkpoint for each epoch will be saved with suffix .<number of epoch>')
    parser.add_argument('--save-model',
                        action='store_true',
                        help='if true, save trained model')
    parser.add_argument('--checkpoint_path',
                        default=None,
                        type=str,
                        help='Path to ReStyle model checkpoint')
    parser.add_argument('--load',
                        type=str,
                        help='path to the checkpoint which will be loaded for prediction or fine-tuning')

    # other relevant hyper-parameters
    parser.add_argument('--noise',
                        action='store_true',
                        help='if true, then inject the noise latent to the model')
    parser.add_argument('--style',
                        action='store_true',
                        help='if true, then inject the style latent to the model')
    parser.add_argument('--noise-drop',
                        type=float,
                        default=0.5,
                        help='noise dropout probability')
    parser.add_argument('--style-drop',
                        type=float,
                        default=0.5,
                        help='style dropout probability')
    parser.add_argument('--cutmix',
                        type=float,
                        default=1,
                        help='parameter for beta distribution. 0 means no cutmix')
    parser.add_argument('--loss-coef',
                        type=float,
                        default=0.2,
                        help='the loss coefficient balancing the CrossEntropy and ArcFace loss')
    parser.add_argument('--embedding-size',
                        type=int,
                        default=1024)
    parser.add_argument('--bn-mom',
                        type=float,
                        default=0.05)
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-5)
    parser.add_argument('--gradient-accumulation',
                        type=int,
                        default=2,
                        help='number of iterations for gradient accumulation')
    parser.add_argument('--pw-aug',
                        type=lambda x: tuple(map(float, x.split(','))),
                        default=(0.1, 0.1),
                        help='pixel-wise augmentation in format (scale std, bias std). scale will be sampled from N(1, scale_std) '
                        'and bias from N(0, bias_std) for each channel independently')
    parser.add_argument('--scale-aug',
                        type=float,
                        default=0.5,
                        help='zoom augmentation. Scale will be sampled from uniform(scale, 1). '
                        'Scale is a scale for edge (preserving aspect)')

    parser.add_argument('--start-epoch',
                        type=int,
                        default=0)
    parser.add_argument('--pred-suffix',
                        default='',
                        help='suffix for prediction output. '
                        'Predictions output will be stored in <loaded checkpoint path>.output<pred suffix>')
    parser.add_argument('--disp-batches',
                        type=int,
                        default=50,
                        help='frequency (in iterations) of printing statistics of training / inference '
                        '(e.g., accuracy, loss, speed)')

    args = parser.parse_args()

    assert args.save is not None
    if args.mode == 'predict':
        assert args.load is not None

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args
