import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import clip
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import argparse
from datetime import datetime
from torch.optim import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
from utils import * 
from dataset import *
from models import *
import clip
from torchvision.transforms import  Normalize
from info_nce import InfoNCE
import shutil
import pyiqa
import lpips
from collections import OrderedDict
from torch.nn import functional as F
from pytorch_msssim import SSIM
from metrics import *

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

if __name__ == '__main__':
    # parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default='test')
    parser.add_argument('--data_type', type=str,default='CALTECH') # CALTECH
    parser.add_argument('--seed', type=int, default=42)
    
    opt = parser.parse_args()
    # labels
    if opt.data_type == 'CALTECH':
        labels = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background']
        opt.base_folder = '/home/chenkang455/chenk/myproject/SpikeCLIP/clip_code/Data/bad/U-CALTECH'
        opt.save_folder = 'exp/CALTECH_eval'
    elif opt.data_type == 'CIFAR':
        labels =  ["frog","horse","dog","truck","airplane","automobile","bird","ship","cat","deer"]
        opt.base_folder = 'data/U-CIFAR'
        opt.save_folder = 'exp/CIFAR_eval'
        
    # prepare
    ckpt_folder = f"{opt.save_folder}/{opt.exp_name}/ckpts"
    img_folder = f"{opt.save_folder}/{opt.exp_name}/imgs"
    os.makedirs(ckpt_folder,exist_ok= True)
    os.makedirs(img_folder,exist_ok= True)
    set_random_seed(opt.seed)
    save_opt(opt,f"{opt.save_folder}/{opt.exp_name}/opt.txt")
    log_file = f"{opt.save_folder}/{opt.exp_name}/results.txt"
    logger = setup_logging(log_file)
    if os.path.exists(f'{opt.save_folder}/{opt.exp_name}/tensorboard'):
        shutil.rmtree(f'{opt.save_folder}/{opt.exp_name}/tensorboard')
    writer = SummaryWriter(f'{opt.save_folder}/{opt.exp_name}/tensorboard')
    logger.info(opt)

    # train and test data splitting
    train_dataset = SpikeData(opt.base_folder,labels,stage = 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=4,pin_memory=True)
    test_dataset = SpikeData(opt.base_folder,labels,stage = 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1,pin_memory=True)
    
    # network 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(device)
    clip_model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
    clip_model = clip_model.to(device)
    for name, param in clip_model.named_parameters():
        param.requires_grad_(False)
    recon_net = LRN(inDim = 50,outDim=1).to(device)
    recon_net.load_state_dict(torch.load('models/LRN_CALTECH.pth'))
    
    # functions
    mean = np.array(preprocess.transforms[4].mean)
    std = np.array(preprocess.transforms[4].std)
    weight = np.array((0.299, 0.587, 0.114))
    gray_mean = sum(mean * weight)
    gray_std = np.sqrt(np.sum(np.power(weight,2) * np.power(std,2)))
    normal_clip = Normalize((gray_mean, ), (gray_std, ))

    # CLIP process - validation
    val_prompts = ['image of a ' + prompt for prompt in labels]
    text = clip.tokenize(val_prompts).to(device)
    val_features = clip_model.encode_text(text)
    val_features = val_features / val_features.norm(dim=-1, keepdim=True)
    
    # -------------------- train ----------------------  
    train_start = datetime.now()
    logger.info("Start Evaluation!")
    # Metrics 
    metrics = {}
    metric_list = ['niqe','brisque']
    num_all = 0
    num_right = 0
    for metric_name in metric_list:
        metrics[metric_name] = AverageMeter()
    # visual
    for batch_idx, (spike,label,label_idx) in enumerate(tqdm(test_loader)):
        # Visual results
        spike = spike.float()
        voxel = torch.sum(spike.reshape(-1,50,4,224,224), axis=2).to(device) # [200,224,224] -> [50,224,224]
        spike_recon = recon_net(voxel).repeat((1,3,1,1))
        tfp = torch.mean(spike,dim = 1,keepdim = True)
        tfi = middleTFI(spike,len(spike[0])//2,len(spike[0])//4-1)
        if batch_idx % 100 == 0:
            save_img(img = normal_img(spike_recon[0,0,30:-30,10:-10]),path = f'{img_folder}/{batch_idx:04}_SpikeCLIP.png')
            save_img(img = normal_img(tfp[0,0,30:-30,10:-10]),path = f'{img_folder}/{batch_idx:04}_tfp.png')
            save_img(img = normal_img(tfi[0,0,30:-30,10:-10]),path = f'{img_folder}/{batch_idx:04}_tfi.png')
        # metric
        for key in metric_list:
            metrics[key].update(compute_img_metric_single(spike_recon,key))
        # cls
        spike_recon = normal_clip(spike_recon)
        image_features = clip_model.encode_image(spike_recon)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ val_features.t()
        probs = logits.softmax(dim=-1)
        index = torch.max(probs, dim=1).indices.detach().cpu()
        mask = index == label_idx
        num_right += sum(mask)
        num_all += len(mask)
    logger.info(f"Acc: {100 * num_right / num_all:.2f}")
    re_msg = ''
    for metric_name in metric_list:
        re_msg += metric_name + ": " + "{:.4f}".format(metrics[metric_name].avg) + "  "
    logger.info(re_msg)