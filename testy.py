# setup numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10) # large images
plt.rcParams['image.interpolation'] = 'nearest' # show square pixels
plt.rcParams['image.cmap'] = 'gray' # use grayscale

# import sys, timeit and caffe
import sys
import timeit
sys.path.append('/home/amlaanb/Caffe/caffe/python')
import caffe

import os
if os.path.isfile('/home/amlaanb/Caffe/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
	print 'Caffenet found'
else:
	print 'Caffenet not found'

# set caffe to CPU mode
caffe.set_mode_cpu()

model_def = '/home/amlaanb/Caffe/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = '/home/amlaanb/Caffe/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

# load mean ImageNet image
mu = np.load('/home/amlaanb/Caffe/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
print 'mean-subtracted values: ', zip('BGR', mu)

# create transformer for input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# set input size
net.blobs['data'].reshape(50, 3, 227, 227)

# load and display image
image = caffe.io.load_image('/home/amlaanb/Caffe/caffe/examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.figure(0)
plt.imshow(image)
plt.show()

# copy image data into alloted memory
net.blobs['data'].data[...] = transformed_image

### perform classification (forward pass) and calculate time using timeit module
t1 = timeit.default_timer()
output = net.forward()
t2 = timeit.default_timer()

output_prob = output['prob'][0] # output probability vector for first image in batch

print 'Predicted class: ', output_prob.argmax()

# load ImageNet labels
labels_file = '/home/amlaanb/Caffe/caffe/data/ilsvrc12/synset_words.txt'

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'Output label: ', labels[output_prob.argmax()]

print 'CPU Forward Pass Time: ', t2 - t1

# for each layer, show the output shape (batch_size, channel_dim, height, width)
for layer_name, blob in net.blobs.iteritems():
	print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
	print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def vis_square(data):
	# normalize data for display
	data = (data - data.min()) / (data.max() - data.min())

	# force number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = (((0, n ** 2 - data.shape[0]),
		(0, 1), (0, 1))
		+ ((0, 0),) * (data.ndim - 3))
	data = np.pad(data, padding, mode='constant', constant_values=1)

	# tile the filters into image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

	plt.imshow(data);
	plt.axis('off');
	plt.show()

filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)

feat = net.blobs['pool5'].data[0]
vis_square(feat)