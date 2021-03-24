import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

data = pd.read_csv('data_entry_labels.csv')

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):

        image_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                image_name = line.split()[0]
                image_names.append(image_name)

        self.image_names = image_names
        self.data = pd.read_csv(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image_path = 'images/' + image_name
        image = Image.open(image_path).convert('RGB')
        label = self.data[self.data['Image Index'] == image_name].iloc[:, 12:-1].to_numpy()
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label).squeeze()

    def __len__(self):
        return len(self.image_names)

# ds = ChestXrayDataSet('data_entry_labels.csv', 'train_val_list.txt')
# print(ds[0][1].shape)