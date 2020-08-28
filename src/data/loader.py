import sys
sys.path.append('../')
from util.header import *


class BaseDataset(Dataset):
    def __init__(self, opt, mode):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        if self.mode == 'Train':
            self.data_file = self.opt.train_pth
        elif self.mode == 'Val':
            self.data_file = self.opt.val_pth
        else:
            self.data_file = self.opt.test_pth
        self.images, self.labels = self._load_data()
        self.transformer = self.get_transformer()
        self.data_size = len(self.images)

    def get_transformer(self):
        transform_list = []
        # resize
        transform_list.append(
            transforms.Resize(self.opt.input_size, Image.BICUBIC))

        # flip
        if self.opt.mode == "Train":
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(self.opt.mean,
                                                   self.opt.std))

        return transforms.Compose(transform_list)

    def _load_data(self):
        images, labels = list(), list()
        if not os.path.exists(self.data_file):
            raise ValueError("data file is not exists!")
        rf = open(self.data_file, 'r')
        lines = rf.readlines()
        for line in lines:
            inf = line.strip().split(' ')
            img_pth = self.opt.img_dir + inf[0]
            if not os.path.isfile(img_pth):
                continue
            if not os.path.exists(img_pth):
                continue
            images.append(img_pth)
            labels.append(list(map(int, inf[1:])))

        return images, labels

    def load_image(self, image_file):
        img = Image.open(image_file)
        img = img.convert('RGB')
        # transform
        input = self.transformer(img)
        return input

    def __getitem__(self, index):
        image_file = self.images[index % self.data_size]
        input = self.load_image(image_file)

        labels_inf = self.labels[index % self.data_size]
        labels = torch.from_numpy(np.array(labels_inf))
        return input, labels

    def __len__(self):
        return self.data_size


class MultiLabelDataLoader():
    def __init__(self, opt):
        self.opt = opt

        # load dataset
        if opt.mode == "Train":
            logging.info("Load Train Dataset...")
            self.train_set = BaseDataset(self.opt, "Train")
            logging.info("Load Validate Dataset...")
            self.val_set = BaseDataset(self.opt, "Val")
        else:
            logging.info("Load Test Dataset...")
            self.test_set = BaseDataset(self.opt, "Test")

    def GetTrainSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.train_set, shuffle=True)
        else:
            raise ("Train Set DataLoader NOT implemented in Test Mode")

    def GetValSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.val_set, shuffle=False)
        else:
            raise ("Validation Set DataLoader NOT implemented in Test Mode")

    def GetTestSet(self):
        if self.opt.mode == "Test":
            return self._DataLoader(self.test_set)
        else:
            raise ("Test Set DataLoader NOT implemented in Train Mode")

    def _DataLoader(self, dataset, shuffle=False):
        dataloader = DataLoader(dataset,
                                batch_size=self.opt.batch_size,
                                shuffle=shuffle,
                                num_workers=self.opt.load_thread,
                                pin_memory=False,
                                drop_last=False)
        return dataloader
