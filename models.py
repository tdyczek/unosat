import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes // 4),
                                   nn.ReLU(inplace=True), )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.ReLU(inplace=True), )

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=1):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()

        base = models.resnet.resnet18(pretrained=True)
        conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        conv1.weight.data = base.conv1.weight.data[:, :2, :, :]
        self.in_block = nn.Sequential(
            conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        # d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y


class UnknownArchitecture(Exception):
    pass


def load_classifier_model(model_conf):
    if model_conf['architecture'] == 'ResNet18':
        model = resnet18_classifier()
    elif model_conf['architecture'] == 'VGG16_BN':
        model = vgg16_bn_classifier()
    elif model_conf['architecture'] == 'VGG11_BN':
        model = vgg11_bn_classifier()
    else:
        raise UnknownArchitecture()

    if model_conf['input_weights']:
        model.load_state_dict(torch.load(model_conf['input_weights']))
    return model


def load_segmentation_model(model_conf):
    if model_conf['architecture'] == 'UNet11':
        model = UNet11(pretrained=True)
    elif model_conf['architecture'] == 'UNet16':
        model = UNet16(pretrained=True)
    elif model_conf['architecture'] == 'UNetResNet18':
        model = UNetResNet18(pretrained=True, is_deconv=True)
    else:
        raise UnknownArchitecture()

    if model_conf['input_weights']:
        model.load_state_dict(torch.load(model_conf['input_weights']))
    return model


def resnet18_classifier():
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 1)
    return resnet


def vgg16_bn_classifier():
    vgg = models.vgg16_bn(pretrained=True)
    num_ftrs = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(num_ftrs, 1)
    return vgg


def vgg11_bn_classifier():
    vgg = models.vgg11_bn(pretrained=True)
    num_ftrs = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(num_ftrs, 1)
    return vgg


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32, pretrained=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1.weight.data = self.encoder[0].weight.data[:, :2, :, :]
        self.conv1.bias.data = self.encoder[0].bias.data
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2,
                                   num_filters * 8 * 2,
                                   num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8),
                                 num_filters * 8 * 2,
                                 num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8),
                                 num_filters * 8 * 2,
                                 num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4),
                                 num_filters * 4 * 2,
                                 num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2),
                                 num_filters * 2 * 2,
                                 num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels,
                 middle_channels, out_channels,
                 is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts,
                following link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class UNetResNet18(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 pretrained=True,
                 is_deconv=False,
                 ):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet18(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2,
                                     num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8,
                                   num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8,
                                   num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8,
                                   num_filters * 4 * 2,
                                   num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2,
                                   num_filters * 2 * 2,
                                   num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2,
                                   num_filters * 2 * 2,
                                   num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.conv1,
                  self.conv2,
                  self.conv3,
                  self.conv4,
                  self.conv5]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        """
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
        """
        x_out = self.final(dec0)

        return x_out


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32,
                 pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2,
                                     num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8,
                                   num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8,
                                   num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8,
                                   num_filters * 4 * 2,
                                   num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2,
                                   num_filters * 2 * 2,
                                   num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out