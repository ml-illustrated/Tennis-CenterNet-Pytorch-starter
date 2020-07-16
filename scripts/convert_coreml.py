import sys, os
import numpy as np
import json, pickle

import coremltools
import onnx_coreml
from coremltools.proto import NeuralNetwork_pb2, FeatureTypes_pb2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from export_onnx import load_test_image
# from utils.post_process import _topk, hmap__bboxes, tennis_hmap__bboxes

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_onnx_model( fn_onnx ):
    import onnx
    
    # Load the ONNX GraphProto object. Graph is a standard Python protobuf object
    onnx_model = onnx.load(fn_onnx)
    
    onnx.checker.check_model(onnx_model)
    return onnx_model

    
if __name__ == '__main__':
    arch = sys.argv[1]
    hc = sys.argv[2]

    name = '%s_%s' % ( arch, hc )
    
    fn_onnx = '/tmp/%s.onnx' % ( name )
    onnx_model = load_onnx_model( fn_onnx )

    from datasets.pascal import VOC_MEAN, VOC_STD    

    # from Survival Guide
    inv_std = 1 / np.mean( VOC_STD )
    red_bias, green_bias, blue_bias = ( -VOC_MEAN[0]*inv_std, -VOC_MEAN[1]*inv_std, -VOC_MEAN[2]*inv_std )
    image_scale = inv_std / 255.0
    image_scale = 1.0 / (255.0 * 0.226) 
    
    use_img_input=True
    fn_mlmodel_path = '/tmp/%s.mlmodel' % ( name )
    if 1:
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
        )
        if use_img_input:
            convert_params.update( dict(
                image_input_names =  ['input.1' ],
                preprocessing_args = {
                    'red_bias': red_bias,
                    'green_bias': green_bias,
                    'blue_bias': blue_bias,
                    'image_scale': image_scale,
                },
            ) )
        
        mlmodel = onnx_coreml.convert(
            onnx_model,
            **convert_params,
        )
        #print(dir(mlmodel))
        spec = mlmodel.get_spec()
        # print(spec.description)        

        # https://machinethink.net/blog/coreml-image-mlmultiarray/
        if mlmodel != None:        
            input = spec.description.input[0]
            input.type.imageType.colorSpace = FeatureTypes_pb2.ImageFeatureType.RGB
            input.type.imageType.height = 384
            input.type.imageType.width = 384

            # print(spec.description)

            mlmodel = coremltools.models.MLModel(spec)

        if mlmodel != None:
            mlmodel.save( fn_mlmodel_path )



'''
USAGE:
python scripts/convert_coreml.py 'resnet' 'hc64'
python scripts/convert_coreml.py 'mobilenetv2' 'hc64'

'''


