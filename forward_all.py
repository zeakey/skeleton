# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import cv2
import scipy.io
import os, sys, shutil, argparse
from os.path import join, splitext, split
parser = argparse.ArgumentParser(description='Forward all testing images.')
parser.add_argument('--model', type=str, default='snapshot/fsds-skl_iter_30000.caffemodel')
parser.add_argument('--net', type=str, default='models/fsds_test.prototxt')
parser.add_argument('--output', type=str, default='softmax_fuse')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ms', type=str, default='False') # use multiscale
parser.add_argument('--save_all', type=str, default='False') # save all interior outputs
parser.add_argument('--scale', type=float, default=1) # resize image
parser.add_argument('--test_dir', type=str, default='data/SK-LARGE/images/test') # switch between testing set.
parser.add_argument('--save_dir', type=str, default=None)
args = parser.parse_args()
sys.path.insert(0, 'caffe/python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(args.gpu)
ms = args.ms
EPSILON = 1e-6
def str2bool(str1):
  if str1.lower() == 'true' or str1.lower() == '1':
    return True
  elif str1.lower() == 'false' or str1.lower() == '0':
    return False
  else:
    raise ValueError('Error!')

args.ms = str2bool(args.ms)
args.save_all = str2bool(args.save_all)

def getsk(x):
  x = np.squeeze(x)
  if x.ndim == 2:
    return x
  elif x.ndim == 3:
    return 1 - np.squeeze(x[0, :, :])
  else:
    raise Exception("Invalid blob ndim: %d"%x.ndim)

def expand_channel(img):
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    img = np.repeat(img, 3, 2)
  return img

def forward(img):
  assert img.ndim == 3, "image shape: %s"%str(img.shape)
  img -= np.array((104.00698793,116.66876762,122.67891434))
  img = img.transpose((2, 0, 1))
  net.blobs['data'].reshape(1, *img.shape)
  net.blobs['data'].data[...] = img
  return net.forward()
net = caffe.Net(args.net, args.model, caffe.TEST)
test_dir = args.test_dir # test images directory
if args.save_dir:
  save_dir = args.save_dir
else:
  save_dir = join('data/sk-results/', splitext(split(args.model)[1])[0]) # directory to save results
if args.ms:
  save_dir += "_multiscale"
if args.scale != 1:
  save_dir += '_scale'+str(args.scale)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
imgs = [i for i in os.listdir(test_dir) if '.jpg' in i]
nimgs = len(imgs)
if args.save_all:
  img = imgs[0]
  img = cv2.imread(join(test_dir, img)).astype(np.float32)
  o = forward(img)
  keys = o.keys()
  outputs = dict(zip(keys, [None]*len(keys)))
print "totally "+str(nimgs)+"images"
for i in range(nimgs):
  img = imgs[i]
  img = cv2.imread(join(test_dir, img))
  img = expand_channel(img)
  h, w, _ = img.shape
  skeleton = np.zeros([h, w], np.float32)
  if args.save_all:
    for k in keys:
      outputs[k] = np.zeros([h, w], np.float32)
  if args.ms:
    scales = np.array([0.25, 0.5, 1, 2])
    #scales = np.array([0.5, 1, 1.5])
  else:
    scales = np.array([1])
  scales = scales * args.scale
  for s in scales:
    h1, w1 = int(s*h), int(s*w)
    img1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    result = forward(img1)
    sk1 = getsk(result[args.output])
    skeleton += cv2.resize(sk1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    if args.save_all:
      for k in keys:
        sk1 = getsk(result[k])
        outputs[k] += cv2.resize(sk1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
  skeleton /= len(scales)
  fn, ext = splitext(imgs[i])
  if np.count_nonzero(skeleton) == 0:
    print("Empty detection at %s" % fn)
  scipy.misc.imsave(join(save_dir, fn+'.png'), skeleton / max(skeleton.max(), EPSILON))
  if args.save_all:
    for k in keys:
      scipy.misc.imsave(join(save_dir, fn+'_'+k+'.png'), outputs[k] / outputs[k].max())
  print "Saving to '" + join(save_dir, imgs[i][0:-4]) + "', Processing %d of %d..."%(i + 1, nimgs)
print "Copying model weights and net proto to %s..."%save_dir
shutil.copy(args.net, save_dir)
shutil.copy(args.model, save_dir)

