import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset
from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class MyCTDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.targets = os.listdir(os.path.join(self.root_dir, "A"))
        self.targets = os.listdir(os.path.join(self.root_dir, "V"))
        self.sources = []
        for filename in self.targets:
            case, index = filename.split('_')[0], filename.split('_')[-1]
            self.sources.append(os.path.join(self.root_dir, "N_V", case + "_N_" + index))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # source = cv2.imread(self.sources[idx])
        source = np.load(self.sources[idx])
        # target = cv2.imread(os.path.join(self.root_dir, "A", self.targets[idx]))
        # target = np.load(os.path.join(self.root_dir, "A", self.targets[idx]))
        target = np.load(os.path.join(self.root_dir, "V", self.targets[idx]), allow_pickle=True)
        # item = self.data[idx]
        # apply_canny = CannyDetector()

        # source_filename = item['source']
        # target_filename = item['target']
        # prompt = item['prompt']
        # prompt = "abdomen CT slices; gray scale image; contrast agent in the artery;"
        prompt = ''
        # detected_map = apply_canny(source, 100, 200)
        # detected_map = HWC3(detected_map)
        

        # source = cv2.imread('./training/fill50k/' + source_filename)
        source = np.expand_dims(source, axis=2)
        source = np.repeat(source, 3, axis = 2)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        target = np.expand_dims(target, axis=2)
        target = np.repeat(target, 3, axis = 2)
        
        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = np.clip(source.astype(np.float32), 0, 2000.0) / 2000.0
        # detected_map = detected_map.astype(np.float32) / 1500

        # Normalize target images to [-1, 1].
        target = np.clip(target.astype(np.float32), 0, 2000.0) / 2000.0
        source = np.zeros((32,512,512,3), dtype=np.float32)
        target = source

        return dict(jpg=target, txt=prompt, hint=source)


class MyCTDataset_2(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.targets_A = os.listdir(os.path.join(self.root_dir, "A"))
        self.targets_V = os.listdir(os.path.join(self.root_dir, "V"))
        self.targets_D = os.listdir(os.path.join(self.root_dir, "D"))
        self.data = []
        for filename in self.targets_A:
            case, index = filename.split('_')[0], filename.split('_')[-1]
            if os.path.exists(os.path.join(self.root_dir, "N_A", case + "_N_" + index)):
                self.data.append(dict(source=os.path.join(self.root_dir, "N_A", case + "_N_" + index),
                                  target=os.path.join(self.root_dir, "A", filename),
                                  # prompt="Artery phase contrast enhanced CT image, bright in artery and liver tumer vessels"))
                                  prompt="Artery phase contrast enhanced CT image"))
        for filename in self.targets_V:
            case, index = filename.split('_')[0], filename.split('_')[-1]
            if os.path.exists(os.path.join(self.root_dir, "N_V", case + "_N_" + index)):
                self.data.append(dict(source=os.path.join(self.root_dir, "N_V", case + "_N_" + index),
                                  target=os.path.join(self.root_dir, "V", filename),
                                  # prompt="Vein phase contrast enhanced CT image, dark in liver tumers, bright in hepatic portal vein"))
                                  prompt="Vein phase contrast enhanced CT image"))
        for filename in self.targets_D:
            case, index = filename.split('_')[0], filename.split('_')[-1]
            if os.path.exists(os.path.join(self.root_dir, "N_D", case + "_N_" + index)):
                self.data.append(dict(source=os.path.join(self.root_dir, "N_D", case + "_N_" + index),
                                  target=os.path.join(self.root_dir, "D", filename),
                                  # prompt="Delay phase contrast enhanced CT image, dark in liver tumers"))
                                  prompt="Delay phase contrast enhanced CT image"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # source = cv2.imread(self.sources[idx])
        # source = np.load(self.sources[idx], allow_pickle=True)
        # target = cv2.imread(os.path.join(self.root_dir, "A", self.targets[idx]))
        # target = np.load(os.path.join(self.root_dir, "A", self.targets[idx]), allow_pickle=True)
        # target = np.load(os.path.join(self.root_dir, "V", self.targets[idx]), allow_pickle=True)
        # item = self.data[idx]
        data = self.data[idx]
        source = np.load(data["source"], allow_pickle=True)
        target = np.load(data["target"], allow_pickle=True)

        # source_filename = item['source']
        # target_filename = item['target']
        prompt = data['prompt']
        # prompt = "gray scale"

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)

        # source = cv2.imread('./training/fill50k/' + source_filename)
        source = np.expand_dims(source, axis=2)
        source = np.repeat(source, 3, axis = 2)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        target = np.expand_dims(target, axis=2)
        target = np.repeat(target, 3, axis = 2)

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = (np.clip(source.astype(np.float32), 0, 2000) - 0) / 2000.0

        # Normalize target images to [-1, 1].
        target = ((np.clip(target.astype(np.float32), 0, 2000) - 0) / 2000.0)
        target = target - source

        return dict(jpg=target, txt=prompt, hint=source)
    

class MyCTDatasetbrain(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.targets = os.listdir(os.path.join(self.root_dir, "A"))
        self.targets = os.listdir(os.path.join(self.root_dir, "W"))
        self.sources = []
        for filename in self.targets:
            case, index = filename.split('_')[0], filename.split('_')[-1]
            self.sources.append(os.path.join(self.root_dir, "N", filename.replace('_W_', '_N_')))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # source = cv2.imread(self.sources[idx])
        source = np.load(self.sources[idx])
        # target = np.load(os.path.join(self.root_dir, "A", self.targets[idx]))
        target = np.load(os.path.join(self.root_dir, "W", self.targets[idx]), allow_pickle=True)
        # item = self.data[idx]
        # apply_canny = CannyDetector()

        # source_filename = item['source']
        # target_filename = item['target']
        # prompt = item['prompt']
        # prompt = "abdomen CT slices; gray scale image; contrast agent in the artery;"
        prompt = ''
        # detected_map = apply_canny(source, 100, 200)
        # detected_map = HWC3(detected_map)
        

        # source = cv2.imread('./training/fill50k/' + source_filename)
        source = np.expand_dims(source, axis=2)
        source = np.repeat(source, 3, axis = 2)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        target = np.expand_dims(target, axis=2)
        target = np.repeat(target, 3, axis = 2)
        
        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = (np.clip(source.astype(np.float32), 85, 185) - 85) / 100.0
        # detected_map = detected_map.astype(np.float32) / 1500

        # Normalize target images to [-1, 1].
        # target = (np.clip(target.astype(np.float32), 90, 120) - 90) / 15.0 - 1.0
        target = (np.clip(target.astype(np.float32), 1106.5, 1159.5) - 1106.5) / 26.5 - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
