import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

from poincare import transform
# from poincare import distance_batch


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    typ = type(mu.data)
    eps = typ(std.size()).normal_()
    eps = Variable(eps)
    z = eps.mul(std).add_(mu)
    kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld).mul_(-0.5)
    return z, kld


def distance_batch(x, y, eps=1e-12):
    xa = x.unsqueeze(0)
    xb = x.unsqueeze(1)
    d2 = ((xa - xb + eps)**2.0).sum(dim=-1)
    ret = torch.sqrt(d2)
    return ret



class VTSNE(nn.Module):
    def __init__(self, n_points, n_topics, n_dim, student=True, parallel=True):
        super(VTSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits_mu = nn.Embedding(n_points, n_topics)
        self.logits_lv = nn.Embedding(n_points, n_topics)
        self.n_points = n_points
        self.n_dim = n_dim
        self.logits_lv.weight.data.add_(5)
        self.parallel = parallel
        self.student = student

    @property
    def logits(self):
        return self.logits_mu

    def sample_logits(self, i=None):
        if i is None:
            return reparametrize(self.logits_mu.weight,
                                 self.logits_lv.weight)
        else:
            return reparametrize(self.logits_mu(i), self.logits_lv(i))

    def forward_poincare(self, pij, i, j):
        row = Variable(torch.arange(0, len(j)).long())
        if self.cuda:
            row = row.cuda()
        # Get  for all points
        # we'll compute the kl divergence(pij, qij)
        # qij = probability of picking distance(i, j)
        # given all other distances involving i
        x, loss_kldrp = self.sample_logits()
        # p = transform(x)
        p = x
        if not self.student:
            xi, _ = self.sample_logits(i)
            pi = transform(transform(xi))
            # Measure distance between every point i and every point in datatset
            # shape is (batchsize, n_data)
            dik = distance_batch(pi, p)
            # set all dii entries to super high distance, so low prob of picking
            dik[row, i] += 1e8
            # (batchsize, n_data)
            log_qik = F.log_softmax(-dik, 1)
            # (batchsize, )
            log_qij = log_qik[row, j]
        else:
            dkl = distance_batch(p, p)
            n_diagonal = dkl.size()[0]
            # The sums over all elements k and l, but in reality we want to
            # skip summing over the diagonal, so we correct by subtracting
            # the diagonal off
            term = (1 + dkl**2.0).pow(-1.0)
            num = term[i, j]
            part = term.sum() - n_diagonal
            log_qij = torch.log(num) - torch.log(part)
        kld = pij * (torch.log(pij + 1e-39) - log_qij)
        loss = kld.sum()
        n_obs = (self.n_points * self.n_points)
        frac = len(i) / (n_obs * n_obs)
        # frac = 1e-9
        assert loss.data[0] == loss.data[0]
        # import pdb; pdb.set_trace()
        return loss + loss_kldrp * frac

    def forward(self, *args):
        return self.forward_poincare(*args)

    def __call__(self, *args):
        return self.forward(*args)
