import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

tf.compat.v1.disable_eager_execution()

ds, info = tfds.load('mnist', split='train', with_info=True, as_supervised=True)
print(ds)

# 784 pixels and None - depends on batch size
real_imgs = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
z = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])


# our generator, accepts noise - z
def generator(z, reuse=None):
	# variable scope allows us to have subset of data we could reuse over layers
	with tf.compat.v1.variable_scope('gen', reuse=reuse):
		hidden1 = tf.compat.v1.layers.dense(inputs=z, units=128, activation=tf.nn.leaky_relu(features=z, alpha=0.2))
		hidden2 = tf.compat.v1.layers.dense(inputs=hidden1, activation=tf.nn.leaky_relu(features=hidden1, alpha=0.2))
		output = tf.compat.v1.layers.dense(inputs=hidden2, units=784, activation=tf.nn.tanh(features=hidden2))
		return output


# will try to tell if something is real or fake.
def discriminator(X, reuse=None):
	# variable scope allows us to have subset of data we could reuse over layers
	with tf.compat.v1.variable_scope('dis', reuse=reuse):
		hidden1 = tf.compat.v1.layers.dense(input=X,
		                                    units=128,
		                                    activation=tf.nn.leaky_relu(alpha=0.1))
		hidden2 = tf.compat.v1.layers.dense(input=z,
		                                    activation=tf.nn.leaky_relu(alpha=0.1))

		# one neuron -> real of fake
		logits = tf.compat.v1.layers.dense(input=1,
		                                   units=1,
		                                   activation=tf.nn.tanh)

		output = tf.nn.sigmoid(logits)
		return output


G = generator(z)

D_output_real, D_logits_real = discriminator(real_imgs)
D_output_fake, D_logits_fake = discriminator(G)
