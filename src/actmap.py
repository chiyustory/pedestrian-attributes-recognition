from util.header import *
from options.options import Options
from models.model import load_model
from data.loader import MultiLabelDataLoader
import util

GRID_SPACING = 10


def visual_actmap(model, test_set, opt):
    width, height = opt.input_size[0], opt.input_size[1]
    for i, data in enumerate(test_set):
        inputs, targets, image_file = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs, return_feature=True)
        if outputs.dim() != 4:
            raise ValueError(
                'The model output is supposed to have shape of (b, c, h, w), \
                                but got {} dimensions. '.format(outputs.dim()))

        # square and sum in channels
        outputs = (outputs**2).sum(1)
        b, h, w = outputs.size()
        # normalize
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        if torch.cuda.is_available():
            inputs, outputs = inputs.cpu(), outputs.cpu()
        # iter every sample
        for j in range(outputs.size(0)):
            # pdb.set_trace()
            # get image name
            img_name = image_file[j].split('/')[-1]

            # RGB image normalize
            img = inputs[j, ...]
            img_np = np.uint8(np.floor(img.numpy() * 255))
            # (c, h, w) -> (h, w, c)
            img_np = img_np.transpose((1, 2, 0))

            # activation map
            am = outputs[j, ...].detach().numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:, width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
            cv2.imwrite(os.path.join(opt.out_dir, img_name + '.jpg'), grid_img)


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
        visual_actmap(model, test_set, opt)


if __name__ == "__main__":
    main()
