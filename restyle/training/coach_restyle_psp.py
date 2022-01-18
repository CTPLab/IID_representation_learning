from training.ranger import Ranger
from models.psp import pSp
from criteria.lpips.lpips import LPIPS
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from configs import data_configs
from criteria import id_loss, w_norm, moco_loss
from utils import common, train_utils, data_utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network
        self.net = pSp(self.opts).to(self.device)

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[
                0].detach()

        # get the image corresponding to the latent average
        self.avg_image = self.net(self.net.latent_avg.unsqueeze(0),
                                  input_code=True,
                                  randomize_noise=False,
                                  return_latents=False,
                                  average_code=True)[0]
        self.avg_image = self.avg_image.to(self.device).float().detach()
        if self.opts.dataset_type == "cars_encode":
            self.avg_image = self.avg_image[:, 32:224, :]
        common.tensor2im(self.avg_image).save(
            os.path.join(self.opts.exp_dir, 'avg_image.jpg'))

        # Initialize loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError(
                'Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(
                start_from_latent_avg=self.opts.start_from_latent_avg)
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = get_train_loader('standard',
                                                 self.train_dataset,
                                                 batch_size=self.opts.batch_size,
                                                 **{'num_workers': int(self.opts.workers),
                                                    'drop_last': True})
        self.test_dataloader = get_eval_loader('standard',
                                               self.test_dataset,
                                               batch_size=self.opts.test_batch_size,
                                               **{'num_workers': int(self.opts.test_workers),
                                                  'drop_last': False})

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def perform_train_iteration_on_batch(self, x, y):
        y_hat, latent = None, None
        loss_dict, id_logs = None, None
        y_hats = {idx: [] for idx in range(x.shape[0])}
        for iter in range(self.opts.n_iters_per_batch):
            if iter == 0:
                avg_image_for_batch = self.avg_image.unsqueeze(
                    0).repeat(x.shape[0], 1, 1, 1)
                x_input = torch.cat([x, avg_image_for_batch], dim=1)
                y_hat, latent = self.net.forward(x_input,
                                                 latent=None,
                                                 return_latents=True)
            else:
                y_hat_clone = y_hat.clone().detach().requires_grad_(True)
                latent_clone = latent.clone().detach().requires_grad_(True)
                x_input = torch.cat([x, y_hat_clone], dim=1)
                y_hat, latent = self.net.forward(x_input,
                                                 latent=latent_clone,
                                                 return_latents=True)

            if self.opts.dataset_type == "cars_encode":
                y_hat = y_hat[:, :, 32:224, :]

            loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
            loss.backward()
            # store intermediate outputs
            for idx in range(x.shape[0]):
                y_hats[idx].append([y_hat[idx], id_logs[idx]['diff_target']])

        return y_hats, loss_dict, id_logs

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, (x, _, _) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, y = x.to(self.device).float(), x.to(self.device).float()

                y_hats, loss_dict, id_logs = self.perform_train_iteration_on_batch(
                    x, y)

                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 100 == 0):
                    self.parse_and_log_images(
                        id_logs, x, y, y_hats, title='images/train')

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def perform_val_iteration_on_batch(self, x, y):
        y_hat, latent = None, None
        cur_loss_dict, id_logs = None, None
        y_hats = {idx: [] for idx in range(x.shape[0])}
        for iter in range(self.opts.n_iters_per_batch):
            if iter == 0:
                avg_image_for_batch = self.avg_image.unsqueeze(
                    0).repeat(x.shape[0], 1, 1, 1)
                x_input = torch.cat([x, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([x, y_hat], dim=1)

            y_hat, latent = self.net.forward(
                x_input, latent=latent, return_latents=True)
            if self.opts.dataset_type == "cars_encode":
                y_hat = y_hat[:, :, 32:224, :]

            loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
            # store intermediate outputs
            for idx in range(x.shape[0]):
                y_hats[idx].append([y_hat[idx], id_logs[idx]['diff_target']])

        return y_hats, cur_loss_dict, id_logs

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, (x, lab, meta) in enumerate(self.test_dataloader):
            with torch.no_grad():
                x, y = x.to(self.device).float(), x.to(self.device).float()
                y_hats, cur_loss_dict, id_logs = self.perform_val_iteration_on_batch(
                    x, y)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, y, y_hats, title='images/test', subscript='{:04d}'.format(batch_idx),
                                      display_count=x.shape[0])

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

            if batch_idx >= 8:
                break

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(
            self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(
                    self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            raise Exception('{} is not a valid dataset_type'.format(
                self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](
            self.opts).get_transforms()

        dataset = get_dataset(dataset=self.opts.dataset_type,
                              root_dir=dataset_args['train_source_root'],
                              split_scheme=self.opts.split_scheme)
        train_dataset = dataset.get_subset('train',
                                           transform=transforms_dict['transform_gt_train'])
        test_dataset = dataset.get_subset('train',
                                          transform=transforms_dict['transform_test'])
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(
                '{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            if type(y_hat) == dict:
                output_face = [
                    [common.tensor2im(y_hat[i][iter_idx][0]),
                     y_hat[i][iter_idx][1]]
                    for iter_idx in range(len(y_hat[i]))
                ]
            else:
                output_face = [common.tensor2im(y_hat[i])]

            cur_im_data = {
                'input_face': common.tensor2im(x[i]),
                'target_face': common.tensor2im(y[i]),
                'output_face': output_face,
            }

            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name,
                                '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name,
                                '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts),
            'latent_avg': self.net.latent_avg
        }
        return save_dict
