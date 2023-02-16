
import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models import create_model

class TimmModel(nn.Module):
    def __init__(self, model, model_name):
        super(TimmModel, self).__init__()

        self.model = model
        self.model_name = model_name

        if 'efficient' in model_name or 'resnet' in model_name:
            self.global_pool = model.global_pool
        else:
            self.global_pool = model.flatten


        if 'resnet' in model_name:
            self.classifier = model.fc
        else:
            self.classifier = model.classifier

        self.drop_rate = model.drop_rate
        self.num_classes = model.num_classes
        self.num_features = model.num_features

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)

        if self.drop_rate > 0:
            x = F.dropout(x, p=float(self.drop_rate), training=self.model.training)

        ft = x
        x = self.classifier(x)
        return x, ft

def get_model_from_name( args, model_name, model_type='timm', pretrained=True ):
    if model_type == 'timm':
        model = create_model(
          model_name,
          pretrained=pretrained, #args.pretrained,
          num_classes=args.num_classes,
          drop_rate=args.drop,
          drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
          drop_path_rate=args.drop_path,
          drop_block_rate=args.drop_block,
          global_pool=args.gp,
          bn_momentum=args.bn_momentum,
          bn_eps=args.bn_eps,
          scriptable=args.torchscript,
          checkpoint_path=args.initial_checkpoint)
        model_cfg = model.default_cfg
        model = TimmModel(model, model_name)
    else:
        print('Model type ', model_type, ' not supported!!')
        assert(1==2)
    return model, model_cfg

