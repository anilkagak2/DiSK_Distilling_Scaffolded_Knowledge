

from models import get_cifar_models, get_instant_weight_model
from resnet import get_resnet_models
from mobilenetv2 import get_mobilenetv2_models
from ShuffleNetv2 import get_shufflenetv2_models


def get_model_from_name(config, name):

    mobilenet = set(['MobileNetV2', 'MobileNetV2x2'])
    shufflenet = set(['ShuffleNetV2' ])

    new_resnet = set(['resnet32x4', 'resnet8x4', 'resnet20', 'resnet32', 'resnet56', 'resnet110', 
        'wrn_40_2', 'wrn_40_1', 'wrn_16_2'])

    original_resnet = set(['ResNet10_xxxs', 'ResNet10_xxs', 'ResNet10_xs', 'ResNet10_s', 
            'ResNet10_m', 'ResNet10_l','ResNet10','ResNet18','ResNet34','ResNet50',
            'efficientnet','l_efficientnet','efficientnetv2_s'])

    if name in original_resnet:
        return get_cifar_models( config, name )

    if name in new_resnet:
        return get_resnet_models( config, name, class_num=config.class_num )
 
    if name in mobilenet:
        return get_mobilenetv2_models( config, name )

    if name in shufflenet:
        return get_shufflenetv2_models( config, name )


def test():
    from utils import get_model_infos
    from collections import namedtuple

    Arguments = namedtuple("Configure", ('dataset', 'class_num'))
    md_dict = {}
    md_dict['dataset'] = 'cifar100'
    md_dict['class_num'] = 100
    config = Arguments(**md_dict)

    #name = 'ShuffleNetV2'
    #name = 'MobileNetV2x2'
    #name = 'wrn_16_2'
    #name = 'wrn_40_2'
    name = 'wrn_40_1'
    #name = 'resnet8x4'
    #name = 'resnet32x4'
    #name = 'ResNet34' 

    net = get_model_from_name(config, name)
    xshape = (1,3,32, 32)
    flop, param = get_model_infos(net, xshape)

    print(
        "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )

    import numpy as np
    counts = sum(np.prod(v.size()) for v in net.parameters())
    print(counts)

#test()


