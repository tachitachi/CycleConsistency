import argparse
import tensorflow as tf
import data
import numpy as np
from preprocessing import preprocess
import time
import model
from tqdm import tqdm
import os

def main(args):

	# get datasets
	source_dataset = data.get_dataset('svhn', 'train')
	target_dataset = data.get_dataset('mnist', 'train')

	im_s = preprocess(source_dataset.x, 'simple', image_size=28, output_channels=1)
	label_s = source_dataset.y

	im_t = preprocess(target_dataset.x, 'simple', image_size=28, output_channels=1)
	label_t = target_dataset.y

	im_batch_s, label_batch_s, im_batch_t, label_batch_t = data.create_batch([im_s, label_s, im_t, label_t], batch_size=args.batch_size, shuffle=args.shuffle)


	# build models

	transformed_s = model.transformer(im_batch_s, scope='model/s_to_t')
	transformed_t = model.transformer(im_batch_t, scope='model/t_to_s')

	cycled_s = model.transformer(transformed_s, scope='model/t_to_s', reuse=True)
	cycled_t = model.transformer(transformed_t, scope='model/s_to_t', reuse=True)

	# create loss functions

	cycle_loss_s = tf.losses.absolute_difference(im_batch_s, cycled_s, scope='cycle_loss_s')
	cycle_loss_t = tf.losses.absolute_difference(im_batch_t, cycled_t, scope='cycle_loss_t')

	total_loss = cycle_loss_s + cycle_loss_t

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

	summaries.add(tf.summary.image('source', im_batch_s))
	summaries.add(tf.summary.image('target', im_batch_t))
	summaries.add(tf.summary.image('source_transformed', transformed_s))
	summaries.add(tf.summary.image('target_transformed', transformed_t))
	summaries.add(tf.summary.image('source_cycled', cycled_s))
	summaries.add(tf.summary.image('target_cycled', cycled_t))



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
			else:
				loss_val, global_step = sess.run([train_op, tf.train.get_global_step()])

			if last_save_time < time.time() - args.save_every_n_seconds:
				last_save_time = time.time()
				saver.save(sess, checkpoint_path, global_step=global_step)

		saver.save(sess, checkpoint_path, global_step=args.num_batches)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_batches', type=int, default=10000)
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