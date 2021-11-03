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
def weights_init(m):
    # From DCGAN paper, all model weights shall be randomly initialized from Normal distribution with mean=0, stdev=0.02
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, nc=3):
        """
        nz : latent vector 길이
        ngpu : GPU수
        ngf : Size of feature maps in generator
        """
        # converting z to data-space means ultimately creating a RGB image with the same size as the training images (i.e. 3x64x64).
        # In practice, this is accomplished through a series of strided two dimensional convolutional transpose layers, each paired with a 2d batch norm layer and a relu activation. 
        # The output of the generator is fed through a tanh function to return it to the input data range of [-1,1]. 
        super(Generator, self).__init__()
        self.ngpu = ngpu
        ngf = 64 # size of feature maps in G(output)
        # https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a
        # https://towardsdatascience.com/gans-part2-dcgans-deep-convolution-gans-for-generating-images-c5d3c7c3510e
        self.main = nn.Seqeuntial(
            # input Z (latent vector) : (N, 100, 1, 1)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (N, ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (N, ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (N, ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (N, ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), # no batchnorm last layer
            nn.Tanh()
            # (N, nc, 64, 64)
        )
    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Modle):
    # D, is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake).
    # D takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. 
    # This architecture can be extended with more layers if necessary
    # The DCGAN paper mentions it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function. 
    # Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both G and D.
    def __init__(self, ngpu, nc=3):
        super(Discriminator, self).__init__()
        ndf = 64
        self.main = nn.Seqeuntial(
            # input (N, 3, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, ndf*2, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, ndf*4, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, ndf*8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # (N, 1, 1, 1)
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    