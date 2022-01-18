import cv2
import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


def initialize_transform(args, is_training):
    """ Initialize the transformation functions for 
    rxrx1 and scrc.

    Args:
        args: critical parameters specified in args.py
        is_train: whether is training or evaluation
    """

    if args.data_type == 'rxrx1':
        return initialize_rxrx1_transform(args, is_training)
    elif args.data_type == 'scrc':
        return initialize_scrc_transform(is_training)
    else:
        raise ValueError(f"{args.data_type} not recognized")


def initialize_rxrx1_transform(args, is_training):
    """ Initialize the rxrx1 transformation.

    Args:
        args: critical parameters specified in args.py
        is_train: whether is training or evaluation
    """

    def trn_transform(image):
        image = np.asarray(image.convert('RGB'))
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        if random.random() < 0.5:
            image = image[::-1, :, :]
        if random.random() < 0.5:
            image = image.transpose([1, 0, 2])
        image = np.ascontiguousarray(image)

        if args.scale_aug != 1:
            size = random.randint(round(256 * args.scale_aug), 256)
            x = random.randint(0, 256 - size)
            y = random.randint(0, 256 - size)
            image = image[x:x + size, y:y + size]
            image = cv2.resize(image, (256, 256),
                               interpolation=cv2.INTER_NEAREST)

        return image

    if is_training:
        transforms_ls = [
            transforms.Lambda(lambda x: trn_transform(x)),
            transforms.ToTensor()]
    else:
        transforms_ls = [
            transforms.ToTensor()]
    transform = transforms.Compose(transforms_ls)
    return transform


def initialize_scrc_transform(is_training):
    """ Initialize the scrc transformation.

    Args:
        is_train: whether is training or evaluation
    """

    angles = [0, 90, 180, 270]

    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = F.rotate(x, angle)
        return x

    if is_training:
        transforms_ls = [
            transforms.Lambda(lambda x: random_rotation(x)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
        ]
    transform = transforms.Compose(transforms_ls)
    return transform


def get_dataloader(args):
    """ Get wilds dataloader.
    rxrx1: get train, val, test(ood_test), id_test
        dataloader splitted the wilds official split scheme
    scrc0,1,2: get train, val, test(ood_test)
        dataloader splitted with the cutomized wilds scrc0,1,2 split scheme

    Args:
        args: critical parameters specified in args.py
    """

    dataset = get_dataset(dataset=args.data_type,
                          root_dir=args.data,
                          split_scheme=args.split_scheme)
    trn_data = dataset.get_subset('train',
                                  transform=initialize_transform(args, True))
    val_data = dataset.get_subset('val',
                                  transform=initialize_transform(args, False))
    tst_data = dataset.get_subset('test',
                                  transform=initialize_transform(args, False))

    trn_loader = get_train_loader('standard',
                                  trn_data,
                                  batch_size=args.batch_size,
                                  **{'drop_last': True,
                                     'num_workers': args.num_data_workers,
                                     'worker_init_fn': worker_init_fn,
                                     'pin_memory': True})
    val_loader = get_eval_loader('standard',
                                 val_data,
                                 batch_size=args.eval_batch_size,
                                 **{'drop_last': False,
                                     'num_workers': args.num_data_workers,
                                     'worker_init_fn': worker_init_fn,
                                     'pin_memory': True})
    tst_loader = get_eval_loader('standard',
                                 tst_data,
                                 batch_size=args.eval_batch_size,
                                 **{'drop_last': False,
                                    'num_workers': args.num_data_workers,
                                    'worker_init_fn': worker_init_fn,
                                    'pin_memory': True})

    dt_loader = [trn_loader, val_loader, tst_loader]
    if args.data_type == 'rxrx1':
        id_tst_data = dataset.get_subset('id_test',
                                         transform=initialize_transform(args, False))
        id_tst_loader = get_eval_loader('standard',
                                        id_tst_data,
                                        batch_size=args.eval_batch_size,
                                        **{'drop_last': False,
                                           'num_workers': args.num_data_workers,
                                           'worker_init_fn': worker_init_fn,
                                           'pin_memory': True})
        dt_loader.append(id_tst_loader)

    return dt_loader


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)
