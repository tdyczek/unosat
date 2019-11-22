import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn.modules.loss import _Loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input, target) + 0.0001
        self.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = 2 * self.inter.float() / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceLoss(_Loss):
    def forward(self, input, target):
        return 1 - dice_coeff(F.sigmoid(input), target)


def dice_loss(input, target):
    smooth = 0.001

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def jaccard(input, target):
    smooth = 0.001

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return intersection / (iflat.sum() + tflat.sum() - intersection + smooth)


def iou(input, target):
    iflat = input.view(-1)
    tflat = target.view(-1)
    smooth = 0.001
    b_input = iflat > 0.5

    tp = (b_input & tflat.bool()).sum().float()
    fp = ((b_input == 1) & (tflat == 0)).sum().float()
    fn = ((b_input == 0) & (tflat == 1)).sum().float()

    return tp / (tp + fp + fn + smooth)
