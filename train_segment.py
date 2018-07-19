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

def main(args):

	# get datasets
	dataset = data.get_dataset(args.dataset, args.split, image_size=args.image_size, data_dir=args.data_dir, is_training=True)

	im_x = preprocess(dataset.x, args.preprocessing_a, image_size=args.image_size, output_channels=args.num_channels)
	im_y = preprocess(dataset.y, args.preprocessing_b, image_size=args.image_size)

	im_batch_x, im_batch_y = data.create_batch([im_x, im_y], batch_size=args.batch_size, shuffle=args.shuffle)


	# build models

	transformed_x = model.transformer(im_batch_x, output_channels=dataset.num_classes, output_fn=None, scope='model/AtoB')
	transformed_y = model.transformer(im_batch_y, output_channels=args.num_channels, scope='model/BtoA')

	cycled_x = model.transformer(transformed_x, output_channels=args.num_channels, scope='model/BtoA', reuse=True)
	cycled_y = model.transformer(transformed_y, output_channels=dataset.num_classes, output_fn=None, scope='model/AtoB', reuse=True)

	# create loss functions

	cycle_loss_x = tf.losses.absolute_difference(im_batch_x, cycled_x, scope='cycle_loss_x')
	cycle_loss_y = tf.losses.softmax_cross_entropy(im_batch_y, cycled_y, scope='cycle_loss_y')

	transform_loss_xy = tf.losses.absolute_difference(im_batch_x, transformed_y, scope='transform_loss_xy')
	transform_loss_yx = tf.losses.softmax_cross_entropy(im_batch_y, transformed_x, scope='transform_loss_yx')

	total_loss = cycle_loss_x + cycle_loss_y + transform_loss_xy + transform_loss_yx

	optimizer = tf.train.AdamOptimizer(args.learning_rate, args.beta1, args.beta2, args.epsilon)

	inc_global_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
	tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, inc_global_step)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_tensor = optimizer.minimize(total_loss)

		# Set up train op to return loss
		with tf.control_dependencies([train_tensor]):
			train_op = tf.identity(total_loss, name='train_op')




	# set up logging

	# Gather initial summaries.
	summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

	# Add summaries for losses.
	for loss in tf.get_collection(tf.GraphKeys.LOSSES):
		summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

	# Add summaries for variables.
	for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		summaries.add(tf.summary.histogram(variable.op.name, variable))


	color_map = np.array(list(map(lambda x: x.color, labels[:dataset.num_classes]))).astype(np.float32)

	segmentation_y = postprocess(tf.argmax(im_batch_y, -1), 'segmentation_to_rgb', dataset.num_classes, color_map)
	segmentation_transformed_x = postprocess(tf.argmax(transformed_x, -1), 'segmentation_to_rgb', dataset.num_classes, color_map)
	segmentation_cycled_y = postprocess(tf.argmax(cycled_y, -1), 'segmentation_to_rgb', dataset.num_classes, color_map)

	summaries.add(tf.summary.image('x', im_batch_x))
	summaries.add(tf.summary.image('y', segmentation_y))
	summaries.add(tf.summary.image('transformed_x', segmentation_transformed_x))
	summaries.add(tf.summary.image('transformed_y', transformed_y))
	summaries.add(tf.summary.image('cycled_x', cycled_x))
	summaries.add(tf.summary.image('cycled_y', segmentation_cycled_y))



	# Merge all summaries together.
	summary_op = tf.summary.merge(list(summaries), name='summary_op')


	# create train loop

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
	checkpoint_path = os.path.join(args.output_dir, 'model.ckpt')
	writer = tf.summary.FileWriter(args.output_dir)

	with tf.Session() as sess:
		# Tensorflow initializations
		sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
		tf.train.start_queue_runners(sess=sess)
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		last_log_time = 0
		last_save_time = 0
		for i in tqdm(range(args.num_batches)):
			if last_log_time < time.time() - args.log_every_n_seconds:
				last_log_time = time.time()
				summary, loss_val, global_step = sess.run([summary_op, train_op, tf.train.get_global_step()])
				writer.add_summary(summary, global_step)
				writer.flush()
			else:
				loss_val, global_step = sess.run([train_op, tf.train.get_global_step()])

			if last_save_time < time.time() - args.save_every_n_seconds:
				last_save_time = time.time()
				saver.save(sess, checkpoint_path, global_step=global_step)

		saver.save(sess, checkpoint_path, global_step=args.num_batches)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='kitti')
	parser.add_argument('--data_dir', type=str, default='/home/aaron/data/datasets/kitti')
	parser.add_argument('--split', type=str, default='train')
	parser.add_argument('--preprocessing_a', type=str, default='simple')
	parser.add_argument('--preprocessing_b', type=str, default=None)

	parser.add_argument('--image_size', type=int, default=64)
	parser.add_argument('--num_channels', type=int, default=3)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_batches', type=int, default=100000)
	parser.add_argument('--shuffle', type=bool, default=True)
	parser.add_argument('--output_dir', type=str, default='output/%d' % int(time.time() * 1000))
	parser.add_argument('--log_every_n_seconds', type=int, default=30)
	parser.add_argument('--save_every_n_seconds', type=int, default=300)
	parser.add_argument('--learning_rate', type=float, default=1e-4)
	parser.add_argument('--beta1', type=float, default=0.9)
	parser.add_argument('--beta2', type=float, default=0.99)
	parser.add_argument('--epsilon', type=float, default=1e-8)

	args = parser.parse_args()

	main(args)