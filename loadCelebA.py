from torch.utils.data import Dataset
import os
import json
import random
from PIL import Image

from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, datapath = "/data1/tianqi.chen/CelebA"):
        self.datapath = datapath
        self.id_dict = dict()
        with open(datapath + "/data.json", "r") as outfile:
            self.id_dict = json.load(outfile)
        self.imagefeature = dict()
        self.num_lines = 0
        self.feature_list = []
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
        with open(datapath + "/Anno/list_attr_celeba.txt", "r") as f:
            lines = f.readlines()
            self.num_lines = int(lines[0].replace("\n", ""))
            self.feature_list = lines[1].replace("\n", "").split(" ")
            for i in range(len(self.feature_list)):
                self.feature_list[i] = self.feature_list[i].replace("_", " ")
            self.feature_list = self.feature_list[:len(self.feature_list) - 1]
            idx = self.feature_list.index("Male")
            for i in range(2, 2 + self.num_lines):
                # s = "a photo of human face with"
                l = lines[i].replace("\n", "").replace("  ", " ").split(" ")
                # for j in range(len(self.feature_list)):
                #     if l[j + 1] == "-1":
                #         s += " no " + self.feature_list[j] + ","
                #     elif l[j + 1] == "1":
                #         s += " " + self.feature_list[j] + ","
                #     else:
                #         print(f"{i - 1}: {l[j + 1]}")
                if l[idx + 1] == "-1":
                    s = "a real picture of a female 's face"
                else:
                    s = "a real picture of a male 's face"
                    
                self.imagefeature[l[0]] = s
        self.idlist = list(self.id_dict.keys())

    def __len__(self):
        return self.num_lines
    
    def __getitem__(self, idx):
        id = random.choice(self.idlist)
        image1, image2 = random.choices(self.id_dict[id], k = 2)
        sample = Image.open(self.datapath + "/Img/img_align_celeba/" + image1).convert("RGB")
        target = Image.open(self.datapath + "/Img/img_align_celeba/" + image2).convert("RGB")
        tar_text = self.imagefeature[image2]
        sample = self.transform(sample)
        target  = self.transform(target)
        return sample, tar_text, target