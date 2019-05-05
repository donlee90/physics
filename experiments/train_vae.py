""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

from models.vae import VAE
from data.dataset import Experience

from utils.misc import save_checkpoint
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau


cuda = torch.cuda.is_available()
torch.manual_seed(123)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

def KLDivergence(mu1, logsigma1, mu2, logsigma2):
    """ Compute KL(p1 || p2) where p1 ~ N(mu1, sigma1) and p2 ~ N(mu2, sigma2) """
    KLD = - 0.5 * torch.sum(1 + (2 * logsigma1)\
                            - (2 * logsigma2)\
                            - ((2 * logsigma1).exp() + (mu1 - mu2).pow(2)) / (2 * logsigma2).exp())

    return KLD


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    B, L, D = mu.size()
    BCE = F.mse_loss(recon_x, x, size_average=False)

    mu0 = torch.zeros((B, 1, D))
    logsigma0 = torch.zeros((B, 1, D))

    mu_prior = torch.cat((mu0, mu), dim=1)[:,L,:]
    logsigma_prior = torch.cat((logsigma0, logsigma), dim=1)[:,L,:]

    KLD = KLDivergence(mu, logsigma, mu_prior, logsigma_prior)
    return BCE + KLD


def train(epoch, model, train_loader, optimizer, beta=1.0):
    """ One training epoch """
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(x)
        loss = loss_function(recon_batch, x, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(x)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader):
    """ One test epoch """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon_batch, mu, logvar = model(x)
            test_loss += loss_function(recon_batch, x, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def main(args):


    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_train = datasets.ImageFolder(args.data, transform_train)
    dataset_test = datasets.ImageFolder(args.data, transform_test)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size*10, shuffle=True, num_workers=2)


    model = VAE(3, args.latent_size).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    # check vae dir exists, if not, create it
    if not exists(args.logdir):
        mkdir(args.logdir)

    vae_dir = join(args.logdir, 'vae_z{}_b{}'.format(args.latent_size, args.beta))
    if not exists(vae_dir):
        mkdir(vae_dir)
        mkdir(join(vae_dir, 'samples'))

    reload_file = join(vae_dir, 'best.tar')
    if not args.noreload and exists(reload_file):
        state = torch.load(reload_file)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
                  state['epoch'],
                  state['precision']))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])


    cur_best = None

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer)
        test_loss = test(model, test_loader)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        # checkpointing
        best_filename = join(vae_dir, 'best.tar')
        filename = join(vae_dir, 'checkpoint.tar')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': test_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict()
        }, is_best, filename, best_filename)



        if not args.nosamples:
            with torch.no_grad():
                sample = torch.randn(64, args.latent_size).to(device)
                sample = model.decoder(sample).cpu()
                save_image(sample.view(64, 3, 64, 64),
                           join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('data', type=str)
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for training (default: 32)')

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='Number of epochs to train (default: 1000)')

    parser.add_argument('--logdir', type=str, default='log',
                        help='Directory where results are logged')

    parser.add_argument('--latent-size', type=int, default=64,
                        help='Size of latent variable z of VAE')

    parser.add_argument('--noreload', action='store_true',
                        help='Best model is not reloaded if specified')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='Variational free energy')

    parser.add_argument('--nosamples', action='store_true',
                        help='Does not save samples during training if specified')


    args = parser.parse_args()
    main(args)