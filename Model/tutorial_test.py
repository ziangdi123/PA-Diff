from share import *
import cv2
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from tutorial_dataset import MyCTDataset
# from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import torch
import einops
import matplotlib.pyplot as plt
import numpy as np

def psnr(img1, img2):
    im1 = img1.astype(np.float32)
    im2 = img2.astype(np.float32)
    mse = np.mean((im1 - im2) ** 2)
    return 10 * np.log10(1.0 ** 2 / mse)

# resume_path = '.\\models\\epoch=175-step=329119.ckpt'
# resume_path = '.\\models\\V-epoch=43-step=116819.ckpt'
# resume_path = '.\\models\\D-epoch=289-step=527799.ckpt'
resume_path = '.\\models\\AVD-epoch=74-step=426074.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('.\\models\\cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control
model = model.to('cuda:0')
ddim_sampler = DDIMSampler(model)

ddim_steps = 50
num_samples = 1
shape = (4, 64, 64)
prompt = 'gray scale, first phase'
# prompt = ''
scale = 9.0
eta = 0.0
guess_mode = False
a_prompt = ''
n_prompt = ''
# strength = 1.5
strength = 1.0
model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

# dataroot = "C:\\Users\\Administrator\\Desktop\\hcc_test_06_17\\2XY-047\\N"
dataroot = 'D:\\data\\liver_raw\\N_A'
saveroot = 'D:\\data\\liver_gen_3\\A'
files = os.listdir(dataroot)
dataset = []
for file in files:
    if file[-3:] != 'npy':
        continue
    if '_N_' not in file:
        continue
    # source = np.clip(np.load(os.path.join(dataroot, "N",  file)), 0.0, 2000.0) / 2000.0
    source = np.clip(np.load(os.path.join(dataroot,  file)), 0.0, 2000.0) / 2000.0
    # source = np.flipud(source.T)
    source = np.expand_dims(source, 2)
    source = np.repeat(source, 3, axis=2)
    source = np.expand_dims(source, 0)
    dataset.append(dict(hint=source, name=file))
print(len(dataset))

import nibabel as nib
# os.makedirs(os.path.join(dataroot, "D"), exist_ok=True)
# os.makedirs(os.path.join(dataroot, "D_GT"), exist_ok=True)
with torch.no_grad():
    for data in dataset:
        control = torch.from_numpy(data['hint']).to('cuda:0').float()
        print(data["name"])
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        # cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)
        x_samples = model.decode_first_stage(samples)
        x_samples = ((einops.rearrange(x_samples, 'b c h w -> b h w c') + 1.0) / 2.0).cpu().numpy() # [0, 1]
        results = [x_samples[i] for i in range(num_samples)]
        # target = (data['jpg'][0] + 1.0) / 2.0
        hint = data['hint'][0]
        print(hint.max(), hint.min(), hint.shape)
        res = results[0]
        res = (res[:,:,0] + res[:,:,1] + res[:,:,2]) / 3
        # res = res[:,:,0]
        # print(res.shape)
        # plt.imshow((np.clip(data['hint'][0] * 2000.0, 840, 1240)- 840) / 400.0)
        # plt.imshow(np.concatenate(((np.clip(res * 2000.0, 840, 1240) - 840) / 400.0, (np.clip(data['hint'][0,:,:,0] * 2000.0, 840, 1240)- 840) / 400.0), axis=1), cmap='gray')
        # plt.imshow(np.concatenate((res, data['hint'][0,:,:,0]), axis=1), cmap='gray')
        # plt.show()
        # exit()
        # gt = np.load(os.path.join('D:\\data\\liver_raw\\D', data["name"].replace('_N_', '_D_R_')))
        # cv2.imwrite(os.path.join(dataroot, "D_GT", data["name"][:-4].split('_')[0]+"-D_GT"+data["name"][:-4].split('_')[-1]+".jpg"), ((np.clip(gt, 840, 1240) - 840.0) / 400.0 * 255).astype(np.uint8))
        # cv2.imwrite(os.path.join(dataroot, "D", data["name"][:-4].split('_')[0]+"-D"+data["name"][:-4].split('_')[-1]+".jpg"),
        #             ((np.clip(res * 2000.0, 840, 1240) - 840.0) / 400.0 * 255).astype(np.uint8))
        # nii = nib.Nifti1Image(res * 2000 - 1000, np.eye(4))
        res = res * 2000.0
        np.save(os.path.join(saveroot, data["name"].replace('_N_', '_A_gen_')), res)
        # nib.save(nii, os.path.join(dataroot, "D", data["name"][:-4].split('_')[0]+"-D"+data["name"][:-4].split('_')[-1]+".nii"))
        