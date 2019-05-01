import os
import logging
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from models.base_model import BaseModel
from models.modules.loss import L1CharbonnierLoss
from models.modules.MInfoLoss import MutualInfoLoss
from models.modules.MInfoLossDotProduct import MutualInfoLoss

logger = logging.getLogger('base')

class MICSNet(BaseModel):
    def __init__(self, opt):
        super(MICSNet, self).__init__(opt)
        train_opt = opt['train']
        self.pretrain = opt['sensing_matrix']['pretrain']
        self.stages = opt['network_G']['stages']
        self.decoder_name = opt['network_G']['which_model_G']
        self.encoder_name = opt['network_G']['which_model_encoder']
        self.proj_method = opt['sensing_matrix']['proj_method']
        self.sensing_matrix = np.load(opt['sensing_matrix']['root'])
        # self.sensing_matrix = np.expand_dims(self.sensing_matrix, axis=0)
        # self.sensing_matrix = np.repeat(self.sensing_matrix, opt['datasets']['train']['batch_size'], axis=0)
        self.sensing_matrix = torch.from_numpy(self.sensing_matrix).float().to(self.device)

        # define networks and load pretrained models
        self.encoder_finetune = opt['network_G']['which_model_encoder'] is not None
        self.decoder_finetune = 'finetune' in opt['name']

        self.Encoder = networks.define_Encoder(opt).to(self.device)
        self.netG = networks.define_G(opt).to(self.device)  # G

        if self.is_train:
            if self.encoder_finetune:
                self.Encoder.train()
            self.netG.train()
        self.load()

        # define losses, optimizer and scheduler
        if self.is_train:
            # self.MILoss = MutualInfoLoss(opt)
            self.MILoss = MutualInfoLoss(opt)

            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'charbonnier':
                    self.cri_pix = L1CharbonnierLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                print('Remove pixel loss.')
                self.cri_pix = None

            # G consistant loss
            if train_opt['consistant_weight'] > 0:
                l_cons_type = train_opt['consistant_criterion']
                if l_cons_type == 'l1':
                    self.cri_cons = nn.L1Loss().to(self.device)
                elif l_cons_type == 'l2':
                    self.cri_cons = nn.MSELoss().to(self.device)
                elif l_cons_type == 'charbonnier':
                    self.cri_cons = L1CharbonnierLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_cons_type))
                self.l_cons_w = train_opt['consistant_weight']
            else:
                print('Remove consistant loss.')
                self.cri_cons = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'charbonnier':
                    self.cri_pix = L1CharbonnierLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                print('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = self.netG.parameters()
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            # Encoder
            optim_params_encoder = self.Encoder.parameters()
            self.optimizer_encoder = torch.optim.Adam(optim_params_encoder, lr=train_opt['lr_G'],
                                                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_encoder)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer,
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_HR=True):
        if self.proj_method == 'conv':
            self.input_H = data['HR'].to(self.device)
            self.input_sparse = data['sparse'].to(self.device)
            if 'LR' in data:
                input_l = data['LR']
                self.input_L_1 = input_l[0].to(self.device)
                self.input_L_2 = input_l[1].to(self.device)
        elif self.proj_method == 'linear':
            self.measurements = data['measurements'].to(self.device)
            self.input_sparse = data['sparse'].to(self.device)
            self.input_H = data['HR'].to(self.device)
        elif self.proj_method == 'sparse':
            self.input_sparse = data['sparse'].to(self.device)
            self.input_H = data['HR'].to(self.device)

            # if self.opt['network_G']['stages'] != 1:
            #     self.truth_0 = data['LR_0'].to(self.device)
            #     self.truth_1 = data['LR_1'].to(self.device)

            # input_ref = data['ref'] if 'ref' in data else data['HR']
            # self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        if self.proj_method == 'conv':
            if 'Encoder' in self.encoder_name:
                self.enc_out = self.Encoder(self.input_H)
                # self.fake_H = self.netG(self.enc_out.detach())
                self.fake_H = self.netG(self.enc_out)
            elif 'Spa' in self.decoder_name:
                self.fake_H = self.netG(self.input_H, self.input_sparse)
            else:
                self.fake_H = self.netG(self.input_H)

        elif self.proj_method == 'linear':
            if 'Spa' in self.decoder_name:
                self.fake_H = self.netG(self.measurements, self.input_sparse)
            else:
                self.fake_H = self.netG(self.measurements)

        elif 'SpaOnly' in self.decoder_name:
            self.fake_H = self.netG(self.input_sparse)

        l_g_pix, l_g_fea, l_g_cons, l_g_total = 0, 0, 0, 0

        if self.decoder_finetune:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.input_H)   # 1x32x32
                l_g_total += l_g_pix

            if self.cri_cons: # consistant loss
                g_measurements = [torch.matmul(self.sensing_matrix, im.view(-1)) for im in self.fake_H]
                g_measurements = torch.stack(g_measurements, dim=0)
                l_g_cons = self.l_cons_w * self.cri_cons(g_measurements, self.measurements)   # 1x32x32
                l_g_total += l_g_cons


            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.input_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

        # x_fake = torch.cat((self.input_H[1:], self.input_H[0].unsqueeze(0)), dim=0)

        # concatenate MI loss
        # MILoss = self.MILoss(self.enc_out, self.input_H, x_fake)

        # dot-product MI loss
        MILoss = self.MILoss(self.input_H, self.enc_out)

        self.optimizer_encoder.zero_grad()
        if self.decoder_finetune:
            self.optimizer_G.zero_grad()

        MILoss.backward(retain_graph=True)
        if self.decoder_finetune:
            l_g_total.backward()
        self.optimizer_encoder.step()
        if self.decoder_finetune:
            self.optimizer_G.step()

        # set log
        # G
        self.log_dict['MILoss'] = MILoss.item()
        if self.cri_pix and self.decoder_finetune:
            self.log_dict['l_g_pix'] = l_g_pix.item()

        if self.cri_cons:
            self.log_dict['l_g_cons'] = l_g_cons.item()

        if self.cri_fea:
            self.log_dict['l_g_fea'] = l_g_fea.item()

    def test(self):
        self.netG.eval()
        self.Encoder.eval()

        with torch.no_grad():
            if self.proj_method == 'conv':
                if 'Encoder' in self.encoder_name:
                    self.enc_out = self.Encoder(self.input_H)
                    # self.fake_H = self.netG(self.enc_out.detach())
                    self.fake_H = self.netG(self.enc_out)
                elif 'Spa' in self.decoder_name:
                    self.fake_H = self.netG(self.input_H, self.input_sparse)
                else:
                    self.fake_H = self.netG(self.input_H)

            elif self.proj_method == 'linear':
                if 'Spa' in self.decoder_name:
                    self.fake_H = self.netG(self.measurements, self.input_sparse)
                else:
                    self.fake_H = self.netG(self.measurements)

            elif 'SpaOnly' in self.decoder_name:
                self.fake_H = self.netG(self.input_sparse)

        if self.encoder_finetune:
            self.Encoder.train()
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        # out_dict['LR'] = self.var_L.detach()[0].float().cpu()

        if self.pretrain or self.stages==1:
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        else:
            out_dict['SR'] = self.fake_H[-1].detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.input_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            # s, n = self.get_network_description(self.netD)
            # if isinstance(self.netD, nn.DataParallel):
            #     net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
            #                                      self.netD.module.__class__.__name__)
            # else:
            #     net_struc_str = '{}'.format(self.netD.__class__.__name__)
            # logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            # logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_encoder = self.opt['path']['pretrain_model_E']
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        if load_path_encoder is not None:
            logger.info('Loading pretrained model for E [{:s}] ...'.format(load_path_encoder))
            self.load_network(load_path_encoder, self.Encoder)

    def save(self, iter_step):
        self.save_network(self.Encoder, 'E', iter_step)
        self.save_network(self.netG, 'G', iter_step)

def main():
    opt = {
         'gpu_ids': [0],
         'model': 'NLCSNet',
         'network_G': {'which_model_G': 'NLCSGen_ResNet',
          'is_train': True,
          'norm_type': None,
          'act_type': 'leakyrelu',
          'mode': 'CNA',
          'k': 6,
          'in_nc': 1,
          'out_nc': 1,
          'nf': 16,
          'nb': 5,
          'upscale': 2,
          'ksize_enc': 8,
          'patch_size': 4,
          'patch_stride': 4,
          'group': 1,
          'gc': 32,
          'upsample_mode': 'pixelshuffle',
          'fusion': 'cat',
          'enc_enable': False},
         'train': {'lr_G': 0.0001,
          'weight_decay_G': 0,
          'beta1_G': 0.9,
          'lr_scheme': 'MultiStepLR',
          'lr_steps': [20000.0, 50000.0],
          'lr_gamma': 0.5,
          'pixel_criterion': 'l2',
          'pixel_weight': 1,
          'feature_criterion': 'l2',
          'feature_weight': 0,
          'manual_seed': 0,
          'niter': 100000.0,
          'val_freq': 8,
          'lr_decay_iter': 10},
         }

    opt['network_G']['enc_enable'] = True
    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')

    x = torch.rand([16, 1, 128, 128], requires_grad=True).double().to(device)
    z = torch.rand([16, 1, 32, 32], requires_grad=True).double().to(device)

    # model1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=4).double().to(device)
    # y = model1(x)
    # cri_pix = nn.MSELoss().to(device)
    # l_g_pix = cri_pix(y, z)
    # l_g_pix.backward()
    # for p in model1.parameters():
    #     print(p.grad)

    model = networks.define_G(opt).double().to(device)
    y = model(x)
    cri_pix = nn.MSELoss().to(device)
    l_g_pix = cri_pix(y, z)
    l_g_pix.backward()
    for p in model.parameters():
        print(p.grad)

if __name__ == '__main__':
    main()
