import os
import torch
import logging
from .resnet import resnet18


def load_model(opt):
    if opt.model == "Resnet18":
        model = resnet18(opt)
    else:
        logging.error("unknown model type")
        sys.exit(0)

    # load exsiting model
    if opt.checkpoint_name != "":
        if os.path.exists(opt.checkpoint_name):
            logging.info("load pretrained model from " + opt.checkpoint_name)
            if opt.mode == 'Train':
                model_cur = model.state_dict()
                model_pretrain = torch.load(opt.checkpoint_name)
                # model_cur_ = {k: v for k, v in model_pretrain.items() if k in model_cur}
                model_cur_ = {}
                for k, v in model_pretrain.items():
                    if k in model_cur:
                        logging.info("Init {}".format(k))
                        model_cur_[k] = v
                model_cur.update(model_cur_)
                model.load_state_dict(model_cur)
            else:
                model.load_state_dict(torch.load(opt.checkpoint_name))
        else:
            logging.warning("WARNING: unknown pretrained model, skip it.")

    return model


def save_model(model, opt, epoch):
    checkpoint_name = opt.model_dir + "/epoch_%s_snapshot.pth" % (epoch)
    if torch.cuda.is_available():
        torch.save(model.module.state_dict(), checkpoint_name)
    else:
        torch.save(model.cpu().state_dict(), checkpoint_name)
