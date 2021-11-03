import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from models import Generator, Discriminator
from models import weights_init




def create_dataloader(dataroot, image_size, batch_size, num_workers, show_samples=False):
    """
    dataroot = dataset folder root경로
    num_workers = DataLoader로 데이터 로드하기 위한 worker threads수
    batch_size = 훈련에 사용되는 batch_size, DCGAN논문 : 128
    image_size = 훈련에 사용되는 이미지 크기(default: 64x64) , 변경하려면 D와 G의 구조 변경이 필요
    """
    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
    # ImageFolder dataset
    dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    # DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    if show_samples:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1,2,0))) # (C, H, W) -> (H, W, C)
    return dataloader



def train(opt):
    """
    nc = 입력 이미지 채널수(color image->3)
    nz = latent vector 길이
    ngf = G를 통해 전달되는 피쳐맵의 depth와 관련
    ndf = D를 통해 전파되는 피쳐맵들의 depth의 집합
    num_epochs = 진행할 학습 epoch 수
    lr = learning rate(DCGAN논문 : 0.0002)
    beta1 = Adam Optimizer를 위한 hyper parameter(DCGAN논문: 0.5)
    ngpu = GPU수. 0이면 CPU mode로 돌아가게 됨. 0보다 클 경우 해당 수의 GPU로 돌아가게됨.
    """
    dataroot, batch_size, num_epochs, ngpu, show_samples = \
        opt.dataroot, opt.batch_size, opt.num_epochs, opt.ngpu, opt.show_samples
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    image_size = 64
    nc = 3 # number of channel
    nz = 100 # length of latent vector 
    ndf = 64 # size of feature maps in D(input)
    num_workers = 2  # num_workers for DataLoader
    lr = 0.0002
    beta1 = 0.5 # beta1 hyperparam for Adam Optimizer

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dataloader = create_dataloader(dataroot, image_size, batch_size, num_workers, show_samples=opt.show_samples)
    
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
    if (device.type=='cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    #  nn.Sequential이나 nn.Module 내부의 모든 element에 대해서 recursive하게 weight를 초기화 하고 싶다면 torch.nn.module.apply() 함수를 이용하면 된다
    netG.apply(weights_init)
    netD.apply(weights_init)
    print("-----Generator-----")
    print(netG)
    print("-----Discriminator-----")
    print(netD)

    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("--Training Start -- {} epochs --".format(num_epochs))
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device) # Creates a tensor of size size filled with fill_value. The tensor’s dtype is inferred from fill_value.
            output = netD(real_cpu).view(-1)
            D_real_loss = criterion(output, label)
            D_real_loss.backward()
            D_x = output.mean().item()
    
            # Generate
            z_batch = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(z_batch)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            D_fake_loss = criterion(output, label)
            D_fake_loss.backward()
            D_G_z = output.mean().item()
            D_loss = D_real_loss + D_fake_loss
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z))) == minimize log(1-D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            G_loss = criterion(output, label)
            G_loss.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}]\t[{i}/{len(dataloader)}]\tLoss_D:{round(D_loss.item(), 4)} Loss_G:{round(G_loss.item(), 4)}\tD(x): {D_x}, D(G(z)) (in D/in G): {D_G_z}/{D_G_z2}")
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if  (epoch == num_epochs-1) :
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    
    torch.save(netG.state_dict(), './weights/netG.pt')
    torch.save(netD.state_dict(), './weights/netD.pt')
    if opt.save_output:
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('output_fig.png')

        # visualize the training progression of G with an animation
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        ani.save('prog.mp4')

        # take a look some real images and fake images side by side
        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))
        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.savefig('real_n_fake.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data', help="root directory for dataset")
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for training")
    parser.add_argument('--num-epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--ngpu', type=int, default=1, help="number of GPUs available. Use 0 for CPU mode")
    parser.add_argument('--show-samples', action='store_true', help="show some samples of data")
    parser.add_argument('--save-output', default=True, help="save output")

    opt = parser.parse_args()
    print(opt)
    train(opt)