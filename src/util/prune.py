import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import logging
from models.resnet_prune import *

skip = {
    'A': [2, 8, 14, 16, 26, 28, 30, 32],
    'B': [2, 8, 14, 16, 26, 28, 30, 32],
}

prune_prob = {
    'A': [0.3, 0.3, 0.3, 0.0],
    'B': [0.5, 0.6, 0.4, 0.0],
}

def generate_prue_cfg(model,opt):
    layer_id = 1
    cfg = []
    cfg_mask = []
    for i, m in enumerate(model.modules()):
        # prune conv layer
        if isinstance(m, nn.Conv2d):
            # logging.info(layer_id, m)
            # skip identity layer
            if m.kernel_size == (1, 1):
                continue
            # out_channels, in_channels, kernel_size
            out_channels = m.weight.data.shape[0]
            # skip the special layer
            if layer_id in skip[opt.v]:
                # filters mask
                cfg_mask.append(torch.ones(out_channels))
                # filters num
                cfg.append(out_channels)
                layer_id += 1
                continue
            if layer_id % 2 == 0:
                # divide resnet stage by layer_id
                if layer_id <= 6:
                    stage = 0
                elif layer_id <= 14:
                    stage = 1
                elif layer_id <= 26:
                    stage = 2
                else:
                    stage = 3
                # prune prob
                prune_prob_stage = prune_prob[opt.v][stage]
                # firsrt abs,then sum
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                # filters num
                num_keep = int(out_channels * (1 - prune_prob_stage))
                # sort weight of sum
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:num_keep]
                # filters mask
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                cfg.append(num_keep)
                layer_id += 1
                continue
            layer_id += 1
    return cfg,cfg_mask



def generate_new_model(model, new_model,cfg_mask):
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    conv_count = 1
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.Conv2d):
            if m0.kernel_size == (1, 1):
                # Cases for down-sampling convolution.
                m1.weight.data = m0.weight.data.clone()
                continue
            # skip first conve layer
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            # prune first layer
            if conv_count % 2 == 0:
                # select by channel idx
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1, ))
                w = m0.weight.data[idx.tolist(),:,:,:].clone()
                # copy data
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            # prune second layer
            if conv_count % 2 == 1:
                # select by channel idx
                mask = cfg_mask[layer_id_in_cfg - 1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1, ))
                w = m0.weight.data[:, idx.tolist(),:,:].clone()
                # copy data
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg - 1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1, ))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
    return new_model


def load_trained_weight(opt):
    logging.info("load trained model from " + opt.checkpoint_name)
    model.load_state_dict(torch.load(opt.checkpoint_name))
    return model


def save_prue_model(model, opt):
    checkpoint_name = opt.out_dir + "/prune.pth"
    logging.info("save prune model from " + checkpoint_name)
    if torch.cuda.is_available():
        torch.save(model.module.state_dict(), checkpoint_name)
    else:
        torch.save(model.cpu().state_dict(), checkpoint_name)


def print_modules(model):
    for m in model.modules():
        logging.info(m)

if __name__ == "__main__":
    # Prune settings
    parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
    parser.add_argument('--out_dir', type=str, default='/home/yl/Code/ped_attribute/output/', help='out directory')
    parser.add_argument('--v', default='A', type=str, help='version of the pruned model')
    parser.add_argument('--checkpoint_name', type=str, default='/home/yl/Code/model/resnet34-333f7ec4.pth',
                                                         help='path to pretrained model or model to deploy')
    opt = parser.parse_args()
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    # log setting
    log_path = opt.out_dir + "/prune.log"
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)

    # old model
    model = resnet34()
    model = load_trained_weight(opt)
    # print_modules(model)

    # new model
    cfg, cfg_mask = generate_prue_cfg(model,opt)
    new_model = resnet34(cfg=cfg)
    new_model = generate_new_model(model, new_model, cfg_mask)
    # print_modules(new_model)

    save_prue_model(new_model, opt)
    