from __future__ import division
import numpy as np
import sys, os, argparse
from os.path import isfile, join, isdir
sys.path.insert(0, 'caffe/python')
sys.path.insert(0, 'model')
sys.path.insert(0, 'lib')
import caffe
parser = argparse.ArgumentParser(description='Training skeleton nets.')
parser.add_argument('--gpu', type=int, help='gpu ID', default=0)
parser.add_argument('--solver', type=str, help='solver', default='models/fsds_solver.prototxt')
parser.add_argument('--weights', type=str, help='base model', default='models/vgg16convs.caffemodel')
args = parser.parse_args()
assert isfile(args.weights) and isfile(args.solver)
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt
caffe.set_mode_gpu()
caffe.set_device(args.gpu)
if not isdir('snapshot'):
  os.makedirs('snapshot')
solver = caffe.SGDSolver(args.solver)
solver.net.copy_from(args.weights)
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)
for p in solver.net.params:
  param = solver.net.params[p]
  for i in range(len(param)):
    pmean = solver.net.params[p][i].data.mean()
    pstd = solver.net.params[p][i].data.std()
    if i == 0 and pmean == 0 and pstd == 0:
      print "WARNING! layer %s param[%d]: mean=%.5f, std=%.5f"%(p, i, pmean, pstd)
    else:
      print "layer %s param[%d]: mean=%.5f, std=%.5f"%(p, i, pmean, pstd)
raw_input("Press Enter to Continue...")
solver.solve()

