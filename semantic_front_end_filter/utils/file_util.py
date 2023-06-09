import os
import torch
from ruamel.yaml import YAML


def load_param_from_path(data_path):
    model_cfg = YAML().load(open(os.path.join(data_path, "ModelConfig.yaml"), 'r'))
    train_cfg = YAML().load(open(os.path.join(data_path, "TrainConfig.yaml"), 'r'))
    return model_cfg, train_cfg


def save_weights(model, filename, path="./saved_models"):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, filename)
    torch.save(model.state_dict(), fpath)
    return


def save_checkpoint(model, optimizer, epoch, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        , fpath)


def load_weights(model, filename, path="./saved_models"):
    fpath = os.path.join(path, filename)
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified, strict=False)
    return model, optimizer, epoch
