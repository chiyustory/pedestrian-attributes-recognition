from util.header import *
from options.options import Options
from models.model import load_model
from data.loader import MultiLabelDataLoader
import util
import pdb


def test(model, test_set, opt):
    for i, data in enumerate(test_set):
        inputs, targets = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        attr_idx = -1  
        for attr_name, attr_val in opt.attribute.items():
            attr_idx += 1
            attr_val_name = list(attr_val.keys())
            attr_outs = outputs[attr_idx]
            _, attr_preds = torch.max(attr_outs.detach(), dim=1)
            
            attr_labels = targets[:, attr_idx].detach()

            for attr_pred, attr_label in zip(attr_preds, attr_labels):
                # the attr is not labeled
                if attr_label.item() == -1:
                    continue
                # attr val name
                attr_val_pred = attr_val_name[attr_pred.item()]
                attr_val_label = attr_val_name[attr_label.item()]
                # statistics the numbers of pred and groundtruth
                opt.attribute[attr_name][attr_val_pred][1] += 1
                opt.attribute[attr_name][attr_val_label][2] += 1
                if attr_val_pred == attr_val_label:
                    opt.attribute[attr_name][attr_val_pred][0] += 1
    util.print_accuracy(opt)        


def main():
    # parse options 
    op = Options()
    opt = op.parse()

    # save log to disk
    if opt.mode == "Test":
        log_path = opt.out_dir + "/test.log"

    
    # log setting 
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    logging.getLogger().setLevel(logging.INFO)
    

    # load train or test data
    data_loader = MultiLabelDataLoader(opt)
    test_set = data_loader.GetTestSet()

    # load model
    model = load_model(opt)
    model.eval()
    
    # use cuda
    if torch.cuda.is_available():
        model = model.cuda(opt.device_ids[0])
        cudnn.benchmark = True
    
    # Test model
    if opt.mode == "Test":
        test(model, test_set, opt)

if __name__ == "__main__":
    main()
