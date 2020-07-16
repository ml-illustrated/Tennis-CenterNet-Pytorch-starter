import argparse
import torch
import onnx
import os, pickle
import numpy as np
from types import MethodType

from torch import nn
from torch.onnx import OperatorExportTypes

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.process import load_network_arch, load_model

from datasets.pascal import VOC_MEAN, VOC_STD


def get_cfg():
    parser = argparse.ArgumentParser(description='centernet')

    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--head_conv', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--arch', type=str, default='resnet_18')

    parser.add_argument('--model_path', type=str, default='ckpt/pascal_resnet18_384_dp/checkpoint.t7' )

    parser.add_argument("--output_name", required=True, type=str)
    
    cfg = parser.parse_args()
    return cfg


def load_test_image( img_size ):
    fn_image = '../drop-shot.jpg'
    from PIL import Image  
    img = Image.open( fn_image )
    img = img.resize( (img_size, img_size), resample=Image.BILINEAR)
    return img

def preproc_image( image ):
    img_data = np.array(image).astype('float32')

    #normalize
    img_data = (img_data/255. - VOC_MEAN) / VOC_STD    
    img_data = img_data.astype('float32') # float64 by default for some reason

    img_data = img_data.transpose(2, 0, 1)

    return img_data
    
def load_test_img_batch( img_size, batch_size=1 ):
    img = load_test_image( img_size )
    img = preproc_image( img )
    img_batch = [img]*batch_size
    return np.array( img_batch )

def save_model_output_as_json( fn_output, model_outputs ):
    import json
    print( 'torch_outputs: ', model_outputs[0].shape )

    output_data = [
        model_outputs[0][0,14,:].tolist(), # hmap, save only class 14 [1, 20, 96, 96]
        model_outputs[1][0,:].tolist(), # regs [1, 2, 96, 96]
        model_outputs[2][0,:].tolist(), # w_h_ [1, 2, 96, 96]
    ]
    with open( fn_output, 'w' ) as fp:
        json.dump( output_data, fp )


if __name__ == '__main__':
    cfg = get_cfg()
    num_classes = cfg.num_classes

    model_name = '%s_hc%s' % ( cfg.arch, cfg.head_conv )
    model, _ = load_network_arch( cfg.arch, cfg.num_classes, cfg.head_conv, pretrained=False, export_mode=True )
    model = load_model(model, cfg.model_path, is_nested=False, map_location='cpu')
    model.eval()

    img_batch = load_test_img_batch( cfg.img_size, batch_size=1 )
    model_inputs = torch.from_numpy( img_batch )
    torch_outputs = model( model_inputs )

    input_names = ["input.1"]
    output_names = [ 'output_hmap', 'output_regs', 'output_w_h_']

    torch.onnx.export(
        model,
        model_inputs,
        cfg.output_name,
        input_names=input_names,
        output_names=output_names,
        operator_export_type=OperatorExportTypes.ONNX
    )

    fn_output = '/tmp/test_image_outputs.%s.json' % model_name
    save_model_output_as_json( fn_output, torch_outputs[0] )


# pascal model for validation
# python scripts/export_onnx.py --arch mobilenetv2 --img_size 384 --model_path ckpt/pascal_mobilenetv2_384_dp/checkpoint.t7 --num_classes 20  --output_name /tmp/mobilenetv2_hc64.onnx
# python scripts/export_onnx.py --arch resnet_18 --img_size 384 --model_path ckpt/pascal_resnet18_384_dp/checkpoint.t7 --num_classes 20  --output_name /tmp/resnet_hc64.onnx
