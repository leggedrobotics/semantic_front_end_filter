# This file is part of SemanticFrontEndFilter.
#
# SemanticFrontEndFilter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SemanticFrontEndFilter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SemanticFrontEndFilter.  If not, see <https://www.gnu.org/licenses/>.


from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, deactivate_bn, mode = 'convT', input_size = None, concat_size = None):
        super(UpSampleBN, self).__init__()
        self.mode = mode
        if(self.mode == 'convT'):
            paddingx = ceil((input_size[2]*2 - concat_size[2])/2)
            output_paddingx = (input_size[2]*2 - concat_size[2])%2
            paddingy = ceil((input_size[3]*2 - concat_size[3])/2) 
            output_paddingy = (input_size[3]*2 - concat_size[3])%2
            self.up = nn.ConvTranspose2d(input_size[1], input_size[1], kernel_size=2, stride=2, padding=(paddingx, paddingy), output_padding=(output_paddingx, output_paddingy))
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
        if(self.mode=='convT'):
            up_x = self.up(x)
        else:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='nearest')
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048, deactivate_bn = False, skip_connection = False, mode = 'convT', output_mask = False, decoder_num = None, output = 'mask'):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.skip_connection = skip_connection
        self.output_mask = output_mask
        self.decoder_num = decoder_num
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.output = output
        input_size = [[1, 2048, 19, 25], [1, 1024, 34, 45], [1, 512, 68, 90], [1, 256, 135, 180]]
        concat_size = [[1, 176, 34, 45], [1, 64, 68, 90], [1, 40, 135, 180], [1, 24, 269, 359]]
        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2, deactivate_bn = deactivate_bn, input_size=input_size[0], concat_size=concat_size[0], mode=mode)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4, deactivate_bn = deactivate_bn, input_size=input_size[1], concat_size=concat_size[1], mode=mode)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8, deactivate_bn = deactivate_bn, input_size=input_size[2], concat_size=concat_size[2], mode=mode)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16, deactivate_bn = deactivate_bn, input_size=input_size[3], concat_size=concat_size[3], mode=mode)
        # self.up5_add = UpSampleBN(skip_input=features // 16 + 1, output_features=features // 32, deactivate_bn = deactivate_bn)
        self.conv3 = nn.Conv2d(features // 16, features // 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(features // 32, num_classes, kernel_size=1, stride=1)
        self.distance_maintainer = nn.ReLU()
        self.mask_softer = nn.Sigmoid()
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
        if(self.skip_connection and not self.output_mask):
            # x_d5 = self.up5_add(x_d4, x_block_skip[:, 3, :, :])
            x_d5 = self.conv4(self.conv3(x_d4))
            out = x_block_skip[:, 3:, :, :] +  self.distance_maintainer(F.interpolate(x_d5, size=[x_block_skip.size(2), x_block_skip.size(3)], mode='nearest'))
        
        elif self.output_mask and self.decoder_num == 1:
            ## One encoder output two channel one channel is original prediction
            x_d5 = self.conv4(self.conv3(x_d4))
            out_with_mask = F.interpolate(x_d5, size=[x_block_skip.size(2), x_block_skip.size(3)], mode='nearest')
            # mask = self.mask_softer(out_with_mask[:, 1:, :, :])
            # out = mask * out_with_mask[:, :1, :, :] + (1-mask)*x_block_skip[:, 3:, :, :]
            # out_with_mask[:, 1:, :, :]= self.mask_softer(out_with_mask[:, 1:, :, :])
            out = out_with_mask
        elif self.output_mask and self.decoder_num == 2:
            x_d5 = self.conv4(self.conv3(x_d4))
            out = F.interpolate(x_d5, size=[x_block_skip.size(2), x_block_skip.size(3)], mode='nearest')
            # pred = x_d5[:, :1, :, :]
            # if(self.output == 'mask'):
            #     out = self.mask_softer(out)
            # out = mask * out_with_mask[:, :1, :, :] + (1-mask)*x_block_skip[:, 3:, :, :]
        else:
            out = self.conv4(self.conv3(x_d4))
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
    def __init__(self, backend, min_depth, max_depth, norm,
                    normalize_output_mean, normalize_output_std, 
                    deactivate_bn, skip_connection,
                    **kwargs):
        super(UnetAdaptiveBins, self).__init__()

        self.min_val = min_depth
        self.max_val = max_depth
        self.encoder = Encoder(backend)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.skip_connection = skip_connection
        self.interpolate_mode = kwargs['interpolate_mode']
        self.args = kwargs

        if kwargs['output_mask'] and kwargs['decoder_num'] == 1:
            self.decoder = DecoderBN(num_classes=3, deactivate_bn = deactivate_bn, skip_connection = self.skip_connection, mode = self.interpolate_mode, output_mask = kwargs['output_mask'], decoder_num= kwargs['decoder_num'])
        elif kwargs['output_mask'] and kwargs['decoder_num'] == 2:
            self.decoder_pred = DecoderBN(num_classes=1, deactivate_bn = deactivate_bn, skip_connection = self.skip_connection, mode = self.interpolate_mode, output_mask = kwargs['output_mask'], decoder_num= kwargs['decoder_num'], output = 'prediction')
            self.decoder_mask = DecoderBN(num_classes=2, deactivate_bn = deactivate_bn, skip_connection = self.skip_connection, mode = self.interpolate_mode, output_mask = kwargs['output_mask'], decoder_num= kwargs['decoder_num'], output = 'mask')
        else:
            self.decoder = DecoderBN(num_classes=1, deactivate_bn = deactivate_bn, skip_connection = self.skip_connection, mode = self.interpolate_mode)

        self.normalize = transforms.Normalize(mean=[-normalize_output_mean/normalize_output_std], std=[1/normalize_output_std])

    def forward(self, x, **kwargs):
        if(self.args['decoder_num']==1):
            unet_out = self.decoder(self.encoder(x), **kwargs)
        if(self.args['decoder_num']==2):
            encoding = self.encoder(x)
            pred_origin = self.decoder_pred(encoding, **kwargs)
            mask = self.decoder_mask(encoding, **kwargs)
            unet_out = torch.cat([mask, pred_origin], dim=1)

        pred = unet_out
        pred[:, 0 ,:, :] = self.normalize(pred[:, 0 ,:, :])
        return pred

        pred = self.normalize(pred)
        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rateQ
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, input_channel, **kwargs):
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
        # Change input
        if(input_channel == 4):
            # Change first layer to 4 channel
            orginal_first_layer_weight = basemodel.conv_stem.weight
            basemodel.conv_stem= torch.nn.Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
            with torch.no_grad():
                basemodel.conv_stem.weight[:, 0:3, :, :] = orginal_first_layer_weight


        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        if(kwargs['deactivate_bn']):
            basemodel.apply(deactivate_batchnorm)
        m = cls(basemodel, **kwargs)
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
