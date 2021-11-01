import torch.nn as nn
"""
nz = latent vector 길이
ngf = Size of feature maps in generator
ndf = Size of feature maps in discriminator
num_epochs = 진행할 학습 epoch 수
lr = learning rate(DCGAN논문 : 0.0002)
beta1 = Adam Optimizer를 위한 hyper parameter(DCGAN논문: 0.5)
ngpu = GPU수. 0이면 CPU mode로 돌아가게 됨. 0보다 클 경우 해당 수의 GPU로 돌아가게됨.
"""
class Generator(nn.Module):
    def __init__(self, ngpu, nz=100):
        """
        nz : latent vector 길이
        ngpu : GPU수
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        ngf = 64 # size of feature maps in G(output)
        # https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a
        # https://towardsdatascience.com/gans-part2-dcgans-deep-convolution-gans-for-generating-images-c5d3c7c3510e
        self.main = nn.Seqeuntial(
            # input Z (latent vector) : (100, 1, 1)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False)

        )