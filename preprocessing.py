import tensorflow as tf

preps = {}

def register_prep(name):
    def decorator(fn):
        preps[name] = fn
        return fn
    return decorator

def preprocess(inputs, name, *args, **kwargs):
    if name is None:
        return inputs
    preprocessing_fn = preps[name]
    return preprocessing_fn(inputs, *args, **kwargs)


@register_prep('simple')
def simple(inputs, image_size=None, output_channels=None, center=True):
    # inputs should be in the range [0, 1] with shape (batch, height, width, channels)

    im = inputs

    if output_channels == 3 and im.shape[-1] == 1:
        im = tf.tile(im, [1, 1, 1, 3])
    elif output_channels == 1 and im.shape[-1] == 3:
        im = tf.image.rgb_to_grayscale(im)

    if image_size is not None:
        im = tf.image.resize_images(im, (image_size, image_size))

    if center:
        im = im * 2.0 - 1.0

    return im