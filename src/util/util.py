import os
import copy
import numpy as np
import logging
import collections

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    sef update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def print_loss(loss_list, epoch, batch_iter, opt):
    logging.info("Loss of epoch %d batch %d" % (epoch, batch_iter))
    attr_name = list(opt.attribute.keys())
    for index, loss in enumerate(loss_list):
        logging.info("Attribute %s:  %f" % (attr_name[index], loss))


def print_accuracy(opt):
    for attr_name, attr_val in opt.attribute.items():
        num_right, num_label = 0, 0
        for attr_val_name, attr_val_res in attr_val.items():
            num_right += attr_val_res[0]
            num_label += attr_val_res[2]
            # the attr is not labeled
            if attr_val_res[2] == 0:
                continue
            recall = attr_val_res[0] / float(attr_val_res[2])
            acc = attr_val_res[0] / float(attr_val_res[1])
            logging.info("Attribute %s | Value %s Recall: %f Accuracy: %f" %
                         (attr_name, attr_val_name, recall, acc))

            # reset
            attr_val_res = [0, 0, 0]
        acc_all = num_right / float(num_label)
        logging.info("Attribute %s Accuracy: %f " % (attr_name, acc_all))


def opt2file(opt, dst_file):
    args = vars(opt)
    with open(dst_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
