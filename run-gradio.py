import gradio as gr
import numpy as np
import tensorflow as tf
from lucid.modelzoo import vision_models
from lucid.misc.io import show, load, save
from lucid.misc.tfutil import create_session
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
from lucid.optvis.objectives import wrap_objective
import operator
# from PIL import Image,

model = vision_models.InceptionV1()
model.load_graphdef()

print(tf.test.is_gpu_available())

style_layers = [
  'conv2d2',
  'mixed3a',
  'mixed4a',
  'mixed4b',
  'mixed4c',
]

content_layers = [
  'mixed3b',
]

def style_transfer_param(content_image, style_image, decorrelate=True, fft=True):
  style_transfer_input = param.image(*content_image.shape[:2], decorrelate=decorrelate, fft=fft)[0]
  content_input = tf.convert_to_tensor(content_image, dtype="float32")
  style_input = tf.convert_to_tensor(style_image, dtype="float32")
  return tf.stack([style_transfer_input, content_input, style_input])

# these constants help remember which image is at which batch dimension
TRANSFER_INDEX = 0
CONTENT_INDEX = 1
STYLE_INDEX = 2


def mean_L1(a, b):
  return tf.reduce_mean(tf.abs(a-b))

@wrap_objective
def activation_difference(layer_names, activation_loss_f=mean_L1, transform_f=None, difference_to=CONTENT_INDEX):
  def inner(T):
    # first we collect the (constant) activations of image we're computing the difference to
    image_activations = [T(layer_name)[difference_to] for layer_name in layer_names]
    if transform_f is not None:
      image_activations = [transform_f(act) for act in image_activations]

    # we also set get the activations of the optimized image which will change during optimization
    optimization_activations = [T(layer)[TRANSFER_INDEX] for layer in layer_names]
    if transform_f is not None:
      optimization_activations = [transform_f(act) for act in optimization_activations]

    # we use the supplied loss function to compute the actual losses
    losses = [activation_loss_f(a, b) for a, b in zip(image_activations, optimization_activations)]
    return tf.add_n(losses)

  return inner


def gram_matrix(array, normalize_magnitue=True):
  channels = tf.shape(array)[-1]
  array_flat = tf.reshape(array, [-1, channels])
  gram_matrix = tf.matmul(array_flat, array_flat, transpose_a=True)
  if normalize_magnitue:
    length = tf.shape(array_flat)[0]
    gram_matrix /= tf.cast(length, tf.float32)
  return gram_matrix


def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

# content_image = load("ali-abid.JPG")
# style_image = load("mona-lisa.jpg")[..., :3] # removes transparency channel
#
def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

def generate(content, style):
    # content, style = np.array(content, style)
    # style = Image.fromarray(np.uint8(style)).convert('RGB')

    style = style[..., :3]
    param_f = lambda: style_transfer_param(content, style)

    content_obj = 100 * activation_difference(content_layers,
                                              difference_to=CONTENT_INDEX)
    content_obj.description = "Content Loss"

    style_obj = activation_difference(style_layers, transform_f=gram_matrix,
                                      difference_to=STYLE_INDEX)
    style_obj.description = "Style Loss"

    objective = - content_obj - style_obj

    vis = \
    render.render_vis(model, objective, param_f=param_f, thresholds=[512],
                      verbose=False)[-1]
    # print(type(vis))
    # print(vis[0,:,:,:].shape)
    return vis[0,:,:,:]


inputs = [gr.inputs.Image(shape=(512, 512)), gr.inputs.Image(shape=(512,
                                                                    512), 
                                                             image_mode=None)]
outputs = gr.outputs.Image(label="Stylized Image")

gr.Interface(generate, inputs, outputs, title="Style Transfer (Lucid)").launch(
    inbrowser=True)