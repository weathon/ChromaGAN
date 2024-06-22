from torch.utils.data import Dataset, DataLoader
import json
import torch 

with open('config.json', 'r') as f:
    config = json.load(f)

import numpy as np
from PIL import Image
import random
import os
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, raw_img_size, mz, time):
        self.filenames = [path + f for f in os.listdir(path)]
        self.raw_img_size = raw_img_size
        self.mz = mz
        self.time = time

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        while 1:
            try:
                img_x = Image.open(self.filenames[idx])
                img_y = img_x.copy()
                noise_img = Image.open(random.choice(self.filenames)).resize((self.time, self.mz))
                a, b = random.randint(1024, self.raw_img_size), random.randint(1024, self.raw_img_size)
                a2, b2 = a * np.random.uniform(0.9, 1), b * np.random.uniform(0.9, 1)
                w, h = img_x.size
                img_x = np.array(img_x.crop((np.random.uniform(0, 0.01 * w), np.random.uniform(0, 0.01 * h), a, b)).resize((self.time, self.mz)))
                img_y = np.array(img_y.crop((np.random.uniform(0, 0.01 * w), np.random.uniform(0, 0.01 * h), a2, b2)).resize((self.time, self.mz)))

                mean = img_x.mean()
                noise = np.random.normal(0, 1, img_x.shape) * mean * 0.1
                img_x = img_x + noise

                noise = np.random.normal(0, 1, img_x.shape) * mean * 0.1
                img_y = img_y + noise
                list_mz = list(range(self.mz))
                random_earse_indcies_x = np.random.choice(list_mz, random.randint(0, int(0.05 * self.mz)), replace=False)
                random_earse_indcies_y = np.random.choice(list_mz, random.randint(0, int(0.05 * self.mz)), replace=False)
                img_x[random_earse_indcies_x] *= 0.05
                img_y[random_earse_indcies_y] *= 0.05

                img_x = img_x + np.random.uniform(0, 0.1) * np.array(noise_img)
                img_x = (img_x - img_x.min()) / (img_x.max() - img_x.min() + 1e-6)
                img_y = (img_y - img_y.min()) / (img_y.max() - img_y.min() + 1e-6)
                return torch.tensor(img_x*np.random.uniform(0.9,1.1)).to(torch.float32), torch.tensor (img_y*np.random.uniform(0.9,1.1)).to(torch.float32)
            except Exception as e:
                idx = random.randint(0, len(self.filenames))
                print(e)




def loader():
    dataset = Dataset("./2048/", config["raw_img_size"], config["mz"], config["time"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["BS"], shuffle=True, num_workers=4, drop_last=True)
    return dataloader

if __name__ == "__main__":
    for i in loader():
        print(i.shape)
