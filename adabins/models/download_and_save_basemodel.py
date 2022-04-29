# Execute this to download a basemodel manuly

import torch
basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ap', pretrained=True)
torch.save(basemodel,"tf_efficientnet_b5_ap.pth")
