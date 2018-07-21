import argparse
import tensorflow as tf
import data
import numpy as np
from preprocessing import preprocess
from postprocessing import postprocess
import time
import model
from tqdm import tqdm
import os

from labels import labels

import matplotlib.pyplot as plt

def main(args):

	# get datasets
	dataset = data.get_dataset(args.dataset, args.split, image_size=args.image_size, data_dir=args.data_dir, is_training=True)

	im_x = preprocess(dataset.x, args.preprocessing_a, image_size=args.image_size, output_channels=args.num_channels)
	im_y = preprocess(dataset.y, args.preprocessing_b, image_size=args.image_size)

	# No need to use tf.train.batch
	im_x = tf.expand_dims(im_x, 0)
	im_y = tf.expand_dims(im_y, 0)

	# build models

	transformed_x = model.transformer(im_x, output_channels=dataset.num_classes, output_fn=None, scope='model/AtoB')
	transformed_y = model.transformer(im_y, output_channels=args.num_channels, scope='model/BtoA')

	cycled_x = model.transformer(transformed_x, output_channels=args.num_channels, scope='model/BtoA', reuse=True)
	cycled_y = model.transformer(transformed_y, output_channels=dataset.num_classes, output_fn=None, scope='model/AtoB', reuse=True)

	# Correct colors for outputting

	color_map = np.array(list(map(lambda x: x.color, labels[:dataset.num_classes]))).astype(np.float32)

	image_x = (im_x + 1.0) / 2.0
	image_transformed_y = (transformed_y + 1.0) / 2.0
	image_cycled_x = (cycled_x + 1.0) / 2.0

	segmentation_y = postprocess(tf.argmax(im_y, -1), 'segmentation_to_rgb', dataset.num_classes, color_map)
	segmentation_transformed_x = postprocess(tf.argmax(transformed_x, -1), 'segmentation_to_rgb', dataset.num_classes, color_map)
	segmentation_cycled_y = postprocess(tf.argmax(cycled_y, -1), 'segmentation_to_rgb', dataset.num_classes, color_map)

	

	saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))

	with tf.Session() as sess:
		# Tensorflow initializations
		sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
		tf.train.start_queue_runners(sess=sess)
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))

		for i in tqdm(range(args.num_batches)):
			x, y, x_t, y_t, x_c, y_c = sess.run([image_x, segmentation_y, segmentation_transformed_x, image_transformed_y, image_cycled_x, segmentation_cycled_y])

			plt.subplot(231)
			plt.imshow(x[0])
			plt.subplot(232)
			plt.imshow(x_t[0])
			plt.subplot(233)
			plt.imshow(x_c[0])
			plt.subplot(234)
			plt.imshow(y[0])
			plt.subplot(235)
			plt.imshow(y_t[0])
			plt.subplot(236)
			plt.imshow(y_c[0])
			plt.show()




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='kitti')
	parser.add_argument('--data_dir', type=str, default='/home/aaron/data/datasets/kitti')
	parser.add_argument('--split', type=str, default='test')
	parser.add_argument('--preprocessing_a', type=str, default='simple')
	parser.add_argument('--preprocessing_b', type=str, default=None)

	parser.add_argument('--image_size', type=int, default=64)
	parser.add_argument('--num_channels', type=int, default=3)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_batches', type=int, default=10)
	parser.add_argument('--shuffle', type=bool, default=False)
	parser.add_argument('--checkpoint_dir', type=str)

	args = parser.parse_args()

	main(args)