import random
import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
"""
dataroot = dataset folder root경로
num_workers = DataLoader로 데이터 로드하기 위한 worker threads수
batch_size = 훈련에 사용되는 batch_size, DCGAN논문 : 128
image_size = 훈련에 사용되는 이미지 크기(default: 64x64) , 변경하려면 D와 G의 구조 변경이 필요
nc = 입력 이미지 채널수(color image->3)
nz = latent vector 길이
ngf = G를 통해 전달되는 피쳐맵의 depth와 관련
ndf = D를 통해 전파되는 피쳐맵들의 depth의 집합
num_epochs = 진행할 학습 epoch 수
lr = learning rate(DCGAN논문 : 0.0002)
beta1 = Adam Optimizer를 위한 hyper parameter(DCGAN논문: 0.5)
ngpu = GPU수. 0이면 CPU mode로 돌아가게 됨. 0보다 클 경우 해당 수의 GPU로 돌아가게됨.
"""
def train():
    # Set random seed for reproducibility
    manualSeed, num_epochs, batch_size, dataroot, ngpu = opt.manualseed, opt.num_epochs, opt.batch_size, opt.dataroot, opt.ngpu
    image_size = 64
    nc = 3 # number of channel
    ngf = 64 # size of feature maps in G(output)
    ndf = 64 # size of feature maps in D(input)
    num_workers = 2  # num_workers for DataLoader
    lr = 0.0002
    beta1 = 0.5 # beta1 hyperparam for Adam Optimizer

    #create dataset with ImageFolder class
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([]))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manualseed', type=int, default=999, help="randomseed for reproductibility")
    parser.add_argument('--dataroot', type=str, default='./data/celeba', help="root directory for dataset")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for training")
    parser.add_argument('--num-epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--ngpu', type=int, default=1, help="number of GPUs available. Use 0 for CPU mode")

    opt = parser.parse_args()
    print(opt)
    train()