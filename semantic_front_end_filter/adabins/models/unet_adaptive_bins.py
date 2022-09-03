import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .miniViT import mViT
import os

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, deactivate_bn):
        super(UpSampleBN, self).__init__()
        if(deactivate_bn):
            self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                    # nn.BatchNorm2d(output_features),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                    # nn.BatchNorm2d(output_features),
                                    nn.LeakyReLU())
        else:
            self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(output_features),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(output_features),
                                    nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='nearest', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048, deactivate_bn = False, skip_connection = False):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.skip_connection = skip_connection
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2, deactivate_bn = deactivate_bn)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4, deactivate_bn = deactivate_bn)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8, deactivate_bn = deactivate_bn)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16, deactivate_bn = deactivate_bn)
        self.up5_add = UpSampleBN(skip_input=features // 16 + 1, output_features=features // 32, deactivate_bn = deactivate_bn)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        self.distance_maintainer = nn.ReLU()
        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        # if(self.skip_connection):
        #     self.conv3 = nn.Conv2d(features // 32, num_classes, kernel_size=3, stride=1, padding=1)
        # else:
        #     self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block_skip, x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        if(self.skip_connection):
            # x_d5 = self.up5_add(x_d4, x_block_skip[:, 3, :, :])
            x_d5 = self.conv3(x_d4)
            out = x_block_skip[:, 3:, :, :] +  self.distance_maintainer(F.interpolate(x_d5, size=[x_block_skip.size(2), x_block_skip.size(3)], mode='nearest', align_corners=True))
        else:
            out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins, min_depth, max_depth, norm,
                    normalize_output_mean, normalize_output_std, 
                    use_adabins, deactivate_bn, skip_connection,
                    **kwargs):
        super(UnetAdaptiveBins, self).__init__()
        self.use_adabins = use_adabins
        self.num_classes = n_bins
        self.min_val = min_depth
        self.max_val = max_depth
        self.encoder = Encoder(backend)
        self.skip_connection = skip_connection
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        if(use_adabins):
            self.decoder = DecoderBN(num_classes=128, deactivate_bn = deactivate_bn, skip_connection = self.skip_connection)
        else:
            self.decoder = DecoderBN(num_classes=1, deactivate_bn = deactivate_bn, skip_connection = self.skip_connection)

        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.normalize = transforms.Normalize(mean=[-normalize_output_mean/normalize_output_std], std=[1/normalize_output_std])

    def forward(self, x, **kwargs):
        unet_out = self.decoder(self.encoder(x), **kwargs)
        if(self.use_adabins==True):
            bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
            out = self.conv_out(range_attention_maps)

            # Post process
            # n, c, h, w = out.shape
            # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

            bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
            bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
            bin_edges = torch.cumsum(bin_widths, dim=1)

            centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
            n, dout = centers.size()
            centers = centers.view(n, dout, 1, 1)

            pred = torch.sum(out * centers, dim=1, keepdim=True)
        else:
            pred = unet_out
            pred = self.normalize(pred)
            return pred

        pred = self.normalize(pred)
        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, input_channel, use_adabins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        models_path = os.path.dirname(__file__)
        try:
            basemodel = torch.load(os.path.join(models_path, "%s.pth"%basemodel_name))
        except Exception as e:
            basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
            torch.save(basemodel,os.path.join(models_path, "%s.pth"%basemodel_name))
            # NOTE: No internet connection on euler nodes, execute `python3 download_and_save_basemodel.py` to prepare tf_efficientnet_b5_ap.p


        print('Done.')
        if(input_channel == 4):
            # Change first layer to 4 channel
            orginal_first_layer_weight = basemodel.conv_stem.weight
            basemodel.conv_stem= torch.nn.Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
            with torch.no_grad():
                basemodel.conv_stem.weight[:, 0:3, :, :] = orginal_first_layer_weight
                # basemodel.conv_stem.weight[:, 0:3, :, :] = 0


        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        if(kwargs['deactivate_bn']):
            basemodel.apply(deactivate_batchnorm)
        m = cls(basemodel, n_bins=n_bins, use_adabins=use_adabins, **kwargs)
        print('Done.')
        return m
    
    def transform(self):
        orginal_first_layer_weight = self.encoder.original_model.conv_stem.weight
        self.encoder.original_model.conv_stem = torch.nn.Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
        with torch.no_grad():
            self.encoder.original_model.conv_stem.weight[:, 0:3, :, :] = orginal_first_layer_weight
            # self.encoder.original_model.conv_stem.weight[:, 3:, :, :] = 0

if __name__ == '__main__':
    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
