import os
import sys
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.process import load_network_arch, load_model
# from utils.utils import load_model
from utils.image import get_border, get_affine_transform, affine_transform, color_aug, transform_preds

from utils.summary import create_logger
from utils.post_process import ctdet_decode

# from datasets.coco import COCO_MEAN, COCO_STD
from datasets.pascal import VOC_MEAN, VOC_STD

from utils.debugger import Debugger

# Training settings
def get_cfg():
  parser = argparse.ArgumentParser(description='centernet')

  parser.add_argument('--fn_image', type=str)

  parser.add_argument('--root_dir', type=str, default='./')

  parser.add_argument('--arch', type=str, default='resnet_18')

  parser.add_argument('--dataset', type=str, default='pascal')

  parser.add_argument('--img_size', type=int, default=384)
  parser.add_argument('--model_path', type=str, default='ckpt/pascal_resnet18_384_dp/checkpoint.t7' )

  parser.add_argument('--num_classes', type=int, default=20)
  parser.add_argument('--head_conv', type=int, default=64)
  
  parser.add_argument('--test_topk', type=int, default=100)

  cfg = parser.parse_args()

  # cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
  cfg.model_path = cfg.model_path 

  # cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]
  cfg.device = torch.device('cpu')
  return cfg

def main():
  cfg = get_cfg()
  max_per_image = 100
  num_classes = cfg.num_classes

  print('Loading model...')
  model_name = '%s_hc%s' % ( cfg.arch, cfg.head_conv )
  model, shift_buffer = load_network_arch( cfg.arch, cfg.num_classes, cfg.head_conv, pretrained=False )
  model = load_model(model, cfg.model_path, is_nested=False, map_location='cpu')
  
  model = model.to(cfg.device)
  model.eval()

  debugger = Debugger(dataset=cfg.dataset, ipynb=False,
                      theme='black')
  

  all_inputs = [ load_and_transform_image( cfg.fn_image, cfg.img_size ) ]

  results = {}
  with torch.no_grad():
      img_id, inputs = all_inputs[0]
      
      detections = []
      for scale in [ 1. ]:
        img_numpy = inputs[scale]['image']
        img = torch.from_numpy(img_numpy).to(cfg.device)
        output = model(img)[-1] # array of 3
        dets = ctdet_decode(*output, K=cfg.test_topk) # torch.Size([1, 100, 6])
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0] # (100,6)
        # debug img uses dets prior to post_process
        add_debug_image(debugger, img_numpy, dets, output, scale)        

        # print( 'meta: ', inputs[scale]['center'], inputs[scale]['scale'], inputs[scale]['fmap_w'], inputs[scale]['fmap_h'] )

        dets[:, :2] = transform_preds(dets[:, 0:2],
                                      inputs[scale]['center'],
                                      inputs[scale]['scale'],
                                      (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
        dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                       inputs[scale]['center'],
                                       inputs[scale]['scale'],
                                       (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))

        # print( 'dets post_proc: ', dets )
        # MNV3:     [[117.8218   132.52121  227.10435  351.23346    0.854211  14.      ]]
        # resnet18: [[115.41386, 133.93118, 230.14862, 356.79816, 0.90593797]]
        
        cls = dets[:, -1] # (100,)
        top_preds = {}
        for j in range(num_classes):
          inds = (cls == j)
          top_preds[j + 1] = dets[inds, :5].astype(np.float32)
          top_preds[j + 1][:, :4] /= scale

        detections.append(top_preds)

      bbox_and_scores = {}
      for j in range(1, num_classes + 1):
        bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
        # if len(dataset.test_scales) > 1:
        # soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
      scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, num_classes + 1)])

      if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
          keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
          bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

      results[img_id] = bbox_and_scores
      # print( 'bbox_and_scores: ', bbox_and_scores )

      # show_results(debugger, image, results)
      debugger.show_all_imgs(pause=True)


def add_debug_image( debugger, images, dets, output, scale=1):
    down_ratio = 4
    detection = dets.copy() # (K, 6)
    detection[:, :4] *= down_ratio

    for i in range(1):
      img = images[i]
      print( 'img shape:', img.shape ) # C x H x W
      img = img.transpose(1, 2, 0) # to H x x W C
      img = ((img * VOC_STD + VOC_MEAN) * 255).astype(np.uint8)
      # cv2.imshow('t', img )
      heatmap = output[0][i].detach().cpu().numpy() # idx 0 == 'hm'
      print( 'output: ', output[0].shape, output[1].shape, output[2].shape )
      pred = debugger.gen_colormap(heatmap, img.shape[:2])
      print( 'pred: ', pred.shape )
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for det in detection:
        if det[4] > 0.3: # ??????? self.opt.center_thresh:
          print('add_coco_box: ', det)
          debugger.add_coco_bbox(det[:4], det[-1],
                                 det[4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

# copied from coco.py
def load_and_transform_image( fn_image, net_img_size, test_flip=False ):
    padding = 31 # 127 for hourglass
    down_ratio = 4
    
    img_mean = np.array(VOC_MEAN, dtype=np.float32)[None, None, :]
    img_std = np.array(VOC_STD, dtype=np.float32)[None, None, :]
    
    image = cv2.imread(fn_image)
    height, width = image.shape[0:2]

    out = {}
    for scale in [ 1. ]:
      new_height = int(height * scale)
      new_width = int(width * scale)

      if True: # self.fix_size:
        img_height, img_width = net_img_size, net_img_size
        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        scaled_size = max(height, width) * 1.0
        scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
      else: # not fix_size
        img_height = (new_height | padding) + 1
        img_width = (new_width | padding) + 1
        center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        scaled_size = np.array([img_width, img_height], dtype=np.float32)
          
      img = cv2.resize(image, (new_width, new_height))
      trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
      img = cv2.warpAffine(img, trans_img, (img_width, img_height))

      img = img.astype(np.float32) / 255.
      img -= img_mean
      img /= img_std
      img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

      if test_flip:
        img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

      out[scale] = {'image': img,
                    'center': center,
                    'scale': scaled_size,
                    'fmap_h': img_height // down_ratio,
                    'fmap_w': img_width // down_ratio}

    return fn_image, out
      
if __name__ == '__main__':
  main()
  

'''
python scripts/demo.py --test_topk 1 --fn_image ../../pose/drop-shot.jpg 

'''
