import tensorflow as tf

posts = {}

def register_post(name):
    def decorator(fn):
        posts[name] = fn
        return fn
    return decorator

def postprocess(inputs, name, *args, **kwargs):
    postprocessing_fn = posts[name]
    return postprocessing_fn(inputs, *args, **kwargs)


@register_post('segmentation_to_rgb')
def segmentation_to_rgb(inputs, num_classes, color_map, one_hot=False):
    # inputs should be in the range [0, 33] with shape (batch, height, width, 1)

    image_height = tf.shape(inputs)[0]
    image_width = tf.shape(inputs)[1]

    if not one_hot:
        flat_segmentation = tf.reshape(inputs, [-1])
        seg = tf.one_hot(flat_segmentation, num_classes)
    else:
        seg = tf.reshape(inputs, [-1])

    segmentation_rgb = tf.matmul(seg, color_map) / 255
    segmentation_rgb = tf.reshape(segmentation_rgb, [image_height, image_width, 3])

    return segmentation_rgb