import numpy as np
import torch
from torch import nn
from collections import OrderedDict

# from nets.hourglass import get_hourglass
# from nets.resdcn import get_pose_net
from nets.resnet import get_pose_net as get_pose_net_resnet
from nets.mobilenetv2 import get_mobilenetv2

from utils.post_process import ctdet_decode #, tennis_decode
from utils.image import get_affine_transform, affine_transform, transform_preds


def load_model(model, pretrain_dir, is_nested=True, map_location='cuda:0'):
  state_dict_ = torch.load(pretrain_dir, map_location=map_location)
  if is_nested:
    state_dict_ = state_dict_[ 'model' ]
  print('loaded pretrained weights form %s !' % pretrain_dir)
  state_dict = OrderedDict()

  # convert data_parallal to model
  for key in state_dict_:
    if key.startswith('module') and not key.startswith('module_list'):
      state_dict[key[7:]] = state_dict_[key]
    else:
      state_dict[key] = state_dict_[key]

  # check loaded parameters and created model parameters
  model_state_dict = model.state_dict()
  for key in state_dict:
    if key in model_state_dict:
      if state_dict[key].shape != model_state_dict[key].shape:
        print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
          key, model_state_dict[key].shape, state_dict[key].shape))
        state_dict[key] = model_state_dict[key]
    else:
      print('Drop parameter {}.'.format(key))
  for key in model_state_dict:
    if key not in state_dict:
      print('No param {}.'.format(key))
      state_dict[key] = model_state_dict[key]
  model.load_state_dict(state_dict, strict=False)

  return model

def load_network_arch( arch_name, num_classes, head_conv, pretrained=True, export_mode=False, deconv_size=None ):
    '''
    is_nested=True
    if 'hourglass' in arch_name:
        model = get_hourglass[arch_name]
        is_nested=False
    # elif 'resdcn' in arch_name:
    #   model = get_pose_net(num_layers=int(arch_name.split('_')[-1]), num_classes=num_classes)
    #   is_nested=False
    '''
    common_args = {
        'num_classes' : num_classes,
        'head_conv' : head_conv,
        'pretrained' : pretrained,
        'deconv_size' : deconv_size,
    }

    shift_buffer = None
    if arch_name == 'resnet_18':
        model = get_pose_net_resnet(num_layers=int(arch_name.split('_')[-1]), num_classes=num_classes, head_conv=64)
    elif arch_name == 'mobilenetv2_tsmv2':
        model = get_mobilenetv2_tsmv2( export_mode=export_mode, **common_args )
    elif arch_name == 'mobilenetv2_tsmright':
        if export_mode:
            print( 'exporting mobilenetv2_tsmright' )
            model, shift_buffer = get_mobilenetv2_tsmright_for_export( export_mode=export_mode, **common_args )
        else:
            model = get_mobilenetv2_tsmright( export_mode=export_mode, **common_args )
    elif arch_name == 'mobilenetv2_tsmright2':
        if export_mode:
            model, shift_buffer = get_mobilenetv2_tsmright2_for_export( export_mode=export_mode, **common_args )
        else:
            model = get_mobilenetv2_tsmright2( export_mode=export_mode, **common_args )
    elif arch_name == 'mobilenetv2_tsm':
        model = get_mobilenetv2_tsm( export_mode=export_mode, **common_args )
    elif arch_name == 'mobilenetv2':
        model = get_mobilenetv2( export_mode=export_mode, **common_args )
    else:
        raise NotImplementedError
    return model, shift_buffer


