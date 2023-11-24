import torch
import glob, os
from torch.utils.data import Dataset, DataLoader
import csv
import random
from torchvision import transforms, utils
from PIL import Image
import visdom

class Image_Data(Dataset):
    def __init__(self, root, hr_shape):
        self.hr_height, self.hr_width = hr_shape, hr_shape
        self.root = root

        self.imgs_file = []
        for name in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            self.imgs_file += glob.glob(os.path.join(self.root, name, '*.*'))

    def __len__(self):
        return len(self.imgs_file)

    def __getitem__(self, idx):
        lr_transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.hr_height // 4, self.hr_width // 4), Image.BICUBIC),
            transforms.Resize((self.hr_height, self.hr_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        print(lr_transform)
        hr_transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.hr_height, self.hr_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        img_file = self.imgs_file[idx]
        img_1 = lr_transform(img_file)
        img_2 = hr_transform(img_file)

        # Save img1 and img2
        utils.save_image(img_1, f'./results/img/img1_{idx}.png')
        utils.save_image(img_2, f'./results/img/mg2_{idx}.png')

        return img_1, img_2

if __name__ == '__main__':
    root = './data/train/'

    viz = visdom.Visdom()
    img = Image_Data(root, 256)

    loader = DataLoader(img, batch_size=4, shuffle=True)

    for idx, (lr, hr) in enumerate(loader):
        viz.images(hr, nrow=4, win='hr', opts=dict(title='dataloader'))
        viz.images(lr, nrow=4, win='lr', opts=dict(title='dataloader'))
        cat_image = torch.cat((lr, hr), dim=1)
        print(cat_image.size())
        break
