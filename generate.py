from matplotlib.colors import Normalize
from models import Generator 
from models import weights_init
import torch
import torch.nn as nn
import torchvision.utils as vutils
import argparse
import os
from datetime import datetime

def generate(opt, device):
    ngpu = opt.ngpu
    PATH = opt.weight
    save_path = opt.save_path
    nz = 100
    netG = Generator(ngpu).to(device)
    if (device.type=='cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.load_state_dict(torch.load(PATH))
    z = torch.randn(1, nz, 1, 1, device=device)
    
    print("generating an image ...")
    fake = netG(z)
    filename = datetime.now().strftime('%Y%m%d%H%M') + '.png'
    vutils.save_image(fake, os.path.join(save_path, filename), Normalize=True)
    print("gerated image is saved at {}!".format(os.path.join(save_path+filename)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help="number of GPUs available. Use 0 for CPU mode")
    parser.add_argument('--weight', type=str, default='./weights/netG.pt', help="pretrained weight path")
    parser.add_argument('--save-path', type=str, default='./generate', help="genarated image save path")

    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    generate( device)