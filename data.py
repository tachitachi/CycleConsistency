import tensorflow as tf
from scipy.io import loadmat
import util
import numpy as np
import gzip
import struct
import tarfile
import pickle
import os

# CIFAR-10 download link
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz


# SVHN download links
# http://ufldl.stanford.edu/housenumbers/train_32x32.mat
# http://ufldl.stanford.edu/housenumbers/test_32x32.mat
# http://ufldl.stanford.edu/housenumbers/extra_32x32.mat

datasets = {}

def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def get_dataset(name, *args, **kwargs):
    return datasets[name](*args, **kwargs)

@register_dataset('cifar10')
class cifar10(object):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    splits = ['train', 'test']
    batches = {
        'train': ['cifar-10-batches-py/data_batch_1', 'cifar-10-batches-py/data_batch_2', 
                  'cifar-10-batches-py/data_batch_3', 'cifar-10-batches-py/data_batch_4', 
                  'cifar-10-batches-py/data_batch_5'],
        'test': ['cifar-10-batches-py/test_batch'],
    }

    def __init__(self, split):
        assert(split in cifar10.splits)

        # check if downloads exist, and download otherwise
        file = util.download(cifar10.url)

        tar = tarfile.open(file, "r:gz")
        tar.extractall(os.path.dirname(file))
        tar.close()

        filenames = list(map(lambda x: os.path.join('data', x), cifar10.batches[split]))

        # parse mats and read into tf.data.Dataset
        self.x, self.y = self._load(filenames)

    def _load(self, filenames):
        # Must initialize tf.GraphKeys.TABLE_INITIALIZERS
        # sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))

        X_list = []
        y_list = []

        for file in filenames:
            with open(file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                #print(data)
                partial_X = np.asarray(data[b'data']).reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1).astype(np.float32) / 255
                partial_y = np.asarray(data[b'labels']).astype(np.int64)
                X_list.append(partial_X)
                y_list.append(partial_y)
        

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # load into tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        next_X, next_y = iterator.get_next()

        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

        return next_X, next_y


@register_dataset('svhn')
class SVHN(object):
    urls = {
        'train': 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
        'test': 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
    }

    def __init__(self, split):
        assert(split in SVHN.urls)

        # check if downloads exist, and download otherwise
        file = util.download(SVHN.urls[split])

        # parse mats and read into tf.data.Dataset
        self.x, self.y = self._load(file)

    def _load(self, file):
        # Must initialize tf.GraphKeys.TABLE_INITIALIZERS
        # sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))

        mat = loadmat(file)

        X = mat['X'].transpose(3, 0, 1, 2).astype(np.float32)  / 255 # [?, 32, 32, 3] [0, 1] float32

        y = mat['y'].squeeze().astype(np.int64) # [?, 1] [1, 10] int64
        y[y == 10] = 0

        # load into tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        next_X, next_y = iterator.get_next()

        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

        return next_X, next_y

@register_dataset('mnist')
class MNIST(object):
    urls = {
        'train': {
            'image': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'label': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            },
        'test': {
            'image': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'label': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            }
    }

    def __init__(self, split):
        assert(split in MNIST.urls)

        # check if downloads exist, and download otherwise
        image_file = util.download(MNIST.urls[split]['image'])
        label_file = util.download(MNIST.urls[split]['label'])

        # parse mats and read into tf.data.Dataset
        self.x, self.y = self._load(image_file, label_file)

    def _read_idx(self, filepath, num_dims):

        base_magic_num = 2048
        with gzip.GzipFile(filepath) as f:
            magic_num = struct.unpack('>I', f.read(4))[0]
            expected_magic_num = base_magic_num + num_dims
            if magic_num != expected_magic_num:
                raise ValueError('Incorrect MNIST magic number (expected '
                                 '{}, got {})'
                                 .format(expected_magic_num, magic_num))
            dims = struct.unpack('>' + 'I' * num_dims,
                                 f.read(4 * num_dims))

            buf = f.read(np.prod(dims))
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(*dims)
            return data


    def _load(self, image_file, label_file):
        # Must initialize tf.GraphKeys.TABLE_INITIALIZERS
        # sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))

        X = self._read_idx(image_file, 3).reshape([-1, 28, 28, 1]).astype(np.float32) / 255
        y = self._read_idx(label_file, 1).astype(np.int64)

        # load into tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        next_X, next_y = iterator.get_next()

        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

        return next_X, next_y


def create_batch(tensors, batch_size=32, shuffle=False, queue_size=10000, min_queue_size=5000, num_threads=1):
    # Must initialize tf.GraphKeys.QUEUE_RUNNERS
    # tf.train.start_queue_runners(sess=sess)
    if shuffle:
        return tf.train.shuffle_batch(tensors, batch_size=batch_size, capacity=queue_size, min_after_dequeue=min_queue_size, num_threads=num_threads)
    else:
        return tf.train.batch(tensors, batch_size=batch_size, capacity=queue_size, num_threads=num_threads)


if __name__ == '__main__':
    dataset = get_dataset('cifar10', 'train')