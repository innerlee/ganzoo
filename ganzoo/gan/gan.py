import os
import sys
import argparse
import torch
import torch.backends.cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

sys.path.insert(0, os.path.abspath('../../../ganbase'))
import ganbase as gb  # pylint: disable=C0413, E0401

#region arguments yapf: disable
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--dataset', required=True, help='celeba')
parser.add_argument('--dataroot', required=True, help='path for dataset')
parser.add_argument('--workdir', required=True, help='where all the generated files and logs save')
parser.add_argument('--imsize', type=int, default=128, help='image size')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--width', type=int, default=64, help='net width for G and D')
parser.add_argument('--nz', type=int, default=100, help='latent dim')
parser.add_argument('--activation', default='leakyrelu', help='activation for G and D')
parser.add_argument('--normalize', default='batch', help='normalize for G and D')
parser.add_argument('--optimizer', default='adam', help='adam | adamax | rmsprop | sgd')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for D and G')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam/adamax')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam/adamax')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for D and G')
parser.add_argument('--epoch', type=int, default=200, help='how many epochs to train')
parser.add_argument('--repeatD', type=int, default=1, help='how many trainig of D per iteration')
parser.add_argument('--drawepoch', type=int, default=10, help='draw images each how many epochs')
parser.add_argument('--nsample', type=int, default=0, help='how many samples')
opt = parser.parse_args()

#endregion yapf: enable

#region prepare
# setup gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
torch.backends.cudnn.benchmark = True
# setup workdir
os.system(f'mkdir -p {opt.workdir}/png')
saver = gb.Saver(2)
# setup logger
sys.stdout = gb.Logger(opt.workdir)
sys.stderr = gb.ErrorLogger(opt.workdir)
print(sys.argv)
print(opt)

#endregion

#1. load data
dataset, dataloader = gb.loaddata(
    opt.dataset, opt.dataroot, opt.imsize, opt.bs, opt.nsample, droplast=True)
latent = gb.GaussLatent(opt.nz, opt.bs)
print(f'{len(dataset)} samples')
print(f'{len(dataloader)} batches')

#2. load model and init
D = gb.DCGAN_D(
    opt.imsize,
    3,
    opt.width,
    activation=opt.activation,
    normalize=opt.normalize,
    outactivation='none').cuda()
G = gb.DCGAN_G(
    opt.imsize,
    3,
    opt.nz,
    opt.width,
    activation=opt.activation,
    normalize=opt.normalize).cuda()
print(D)
print(G)

D.apply(gb.weights_init_msra)
G.apply(gb.weights_init_msra)

#3. optimizer
optimizerD = gb.get_optimizer(
    D.parameters(),
    opt.optimizer,
    lr=opt.lr,
    beta1=opt.beta1,
    beta2=opt.beta2,
    weight_decay=opt.weight_decay)
optimizerG = gb.get_optimizer(
    G.parameters(),
    opt.optimizer,
    lr=opt.lr,
    beta1=opt.beta1,
    beta2=opt.beta2,
    weight_decay=opt.weight_decay)
print(optimizerD)
print(optimizerG)

#4. loss
loss = torch.nn.BCEWithLogitsLoss()
ones = torch.ones(opt.bs, 1).cuda(async=True)
zeros = torch.zeros(opt.bs, 1).cuda(async=True)

#5. start training
iters = 0
for epoch in range(opt.epoch):
    for i in range(len(dataloader)):

        # update D
        for p in D.parameters():
            p.requires_grad = True
        for p in G.parameters():
            p.requires_grad = False

        lossD_real, lossD_fake = 0, 0
        for _ in range(opt.repeatD):
            D.zero_grad()

            x, _ = next(dataloader)
            x = x.cuda(async=True)
            z = next(latent).cuda(async=True)
            assert x.size() == (opt.bs, 3, opt.imsize, opt.imsize)
            assert z.size() == (opt.bs, opt.nz)

            x_real = Variable(x)
            x_fake = Variable(G(Variable(z, volatile=True)).data)

            err_real = loss(D(x_real), Variable(ones)).mean()
            err_fake = loss(D(x_fake), Variable(zeros)).mean()
            assert err_real.size() == (1, )
            assert err_fake.size() == (1, )

            err_real.backward()
            err_fake.backward()

            lossD_real += err_real.data[0]
            lossD_fake += err_fake.data[0]

            optimizerD.step()

        lossD_real /= opt.repeatD
        lossD_fake /= opt.repeatD

        # update G
        for p in G.parameters():
            p.requires_grad = True
        for p in D.parameters():
            p.requires_grad = False

        G.zero_grad()

        z = next(latent).cuda(async=True)
        gen = G(Variable(z))
        errG = loss(D(gen), Variable(ones)).mean()
        assert errG.size() == (1, )

        errG.backward()

        optimizerG.step()

        lossG = errG.data[0]

        print(
            f'{epoch:03}:{i:04}/{len(dataloader)} loss D real/fake {lossD_real:.7}/{lossD_fake:.7}, G {lossG:.7}'
        )

    if epoch % opt.drawepoch == 0 or epoch == opt.epoch - 1:
        G.eval()

        z = Variable(latent.sample(64).cuda(async=True))
        gen = G(z)
        vutils.save_image(
            gen.data.mul(0.5).add(0.5), f'{opt.workdir}/png/{epoch:06}.png')

        G.train()

    saver.save(G.state_dict(), f'{opt.workdir}/G_epoch{epoch:06}.pth')
    saver.save(D.state_dict(), f'{opt.workdir}/D_epoch{epoch:06}.pth')
