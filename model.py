import tensorflow as tf
import numpy as np

models = {}

def register_model(name):
    def decorator(fn):
        models[name] = fn
        return fn
    return decorator

def get_model(name, *args, **kwargs):
    model_fn = models[name]
    return model_fn(inputs, *args, **kwargs)


@register_model('transformer')
def transformer(inputs, ndf=64, leaky=False, scope='transformer', reuse=False):
	if leaky:
		activation_fn = tf.nn.leaky_relu
	else:
		activation_fn = tf.nn.relu

	with tf.variable_scope(scope, reuse=reuse):

		net = inputs

		net = tf.layers.conv2d(net, filters=ndf, kernel_size=7, strides=1, padding='SAME')
		net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)

		
		net = tf.layers.conv2d(net, filters=ndf * 2, kernel_size=3, strides=2, padding='SAME')
		net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)

		
		net = tf.layers.conv2d(net, filters=ndf * 4, kernel_size=3, strides=2, padding='SAME')
		skip1 = net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)

		
		net = tf.layers.conv2d(net, filters=ndf * 4, kernel_size=3, strides=1, padding='SAME')
		net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)

		
		net = tf.layers.conv2d(net, filters=ndf * 4, kernel_size=3, strides=1, padding='SAME')
		skip2 = net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)

		
		net = tf.layers.conv2d(skip1 + net, filters=ndf * 4, kernel_size=3, strides=1, padding='SAME')
		net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)

		
		net = tf.layers.conv2d(net, filters=ndf * 4, kernel_size=3, strides=1, padding='SAME')
		net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)


		net = tf.layers.conv2d_transpose(skip2 + net, filters=ndf * 2, kernel_size=3, strides=2, padding='SAME')
		net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)


		net = tf.layers.conv2d_transpose(net, filters=ndf, kernel_size=3, strides=2, padding='SAME')
		net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn)


		net = tf.layers.conv2d_transpose(net, filters=inputs.shape[-1], kernel_size=7, strides=1, padding='SAME')
		net = tf.nn.tanh(net)

		return net
