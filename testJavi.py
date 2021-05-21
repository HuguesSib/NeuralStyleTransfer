import os
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import functools

from tensorflow.python.keras.applications.densenet import preprocess_input

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
  
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(style_outputs, content_outputs, style_targets, content_targets):
    style_weight=1e-2
    content_weight=1e4

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[i]-gram_matrix(style_targets[i]))**2) 
                           for i in range(len(style_outputs))])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[i]-content_targets[i])**2) 
                             for i in range(len(content_outputs))])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(vgg, image, style_target, content_target, num_content_layers,num_style_layers):
    with tf.GradientTape() as tape:
        gram_style_outputs, content_outputs = create_outputs(vgg, image,num_content_layers,num_style_layers)
        loss = style_content_loss(gram_style_outputs, content_outputs, style_target, content_target)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

def create_outputs(vgg,image,num_content_layers,num_style_layers):
    preprocessed_image = tf.keras.applications.vgg19.preprocess_input(image*255.0)
    outputs = vgg(preprocessed_image)
    style_outputs = outputs[:num_style_layers]
    content_outputs = outputs[num_style_layers:]

    gram_style_outputs = []
    for s in style_outputs:
        gram_style_outputs.append(gram_matrix(s))

    return gram_style_outputs, content_outputs


## Main code

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_image = load_img(content_path)
style_image = load_img(style_path)

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Style content model
vgg = vgg_layers(style_layers + content_layers)

preprocessed_content_image = tf.keras.applications.vgg19.preprocess_input(content_image*255)
preprocessed_style_image = tf.keras.applications.vgg19.preprocess_input(style_image*255)
content_target = vgg(preprocessed_content_image)
content_target = content_target[num_style_layers:]
style_target = vgg(preprocessed_style_image)
style_target = style_target[:num_style_layers]

image = tf.Variable(content_image)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

train_step(vgg,image, style_target, content_target, num_content_layers,num_style_layers)
