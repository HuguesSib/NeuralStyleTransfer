# Imports
import os
from site import main
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from tensorflow.python.keras.applications.densenet import preprocess_input
from tensorflow.python.ops.image_ops_impl import ResizeMethod

# Functions definition

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, height=None, width=None):
    # Load the image
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if height is None and width is None:
        height = img.shape[0]
        width = img.shape[1]
        new_shape = tf.cast((height, width), tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img, height, width
    else:
        new_shape = tf.cast((height, width), tf.int32)
        img = tf.image.resize(img, new_shape, method=ResizeMethod.GAUSSIAN)
        img = img[tf.newaxis, :]

        return img


def create_noise_image(width, height):
    img = np.random.uniform(0, 1, (width, height, 3))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.reshape(img, (1, width, height, 3))

    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_model(layer_names):
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    #Retrieve input and output layers of the model
    inputs = [vgg.input]
    outputs = []
    for name in layer_names:
        outputs.append(vgg.get_layer(name).output)
    # Model groups layers into an object with training and inference features
    return  tf.keras.Model(inputs, outputs)


def gram_matrix(input_tensor):
     gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
     xshape = input_tensor.get_shape().as_list()
     return gram / (xshape[1]*xshape[2])


def style_content_loss(style_outputs, content_outputs, style_targets, content_targets):
    #ratio = 1e-4
    content_weight = 1e4 # MODIFY THIS VALUE. ALPHA
    style_weight = 1e-2#ratio*content_weight  # MODIFY THIS VALUE. BETA
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[i] - gram_matrix(style_targets[i])) ** 2)
                        for i in range(len(style_outputs))])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[i] - content_targets[i]) ** 2)
                        for i in range(len(content_outputs))])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


@tf.function()
def train_step(vgg, image, style_target, content_target, num_content_layers, num_style_layers):
    high_freq_weight = 50 # MODIFY THIS VALUE 
    with tf.GradientTape() as tape:
        gram_style_outputs, content_outputs = create_outputs(vgg, image, num_style_layers)
        loss = style_content_loss(gram_style_outputs, content_outputs, style_target, content_target)
        loss += high_freq_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


def create_outputs(vgg, image, num_style_layers):
    preprocessed_image = tf.keras.applications.vgg19.preprocess_input(image * 255.0)
    outputs = vgg(preprocessed_image)
    style_outputs = outputs[:num_style_layers]
    content_outputs = outputs[num_style_layers:]

    gram_style_outputs = []
    for s in style_outputs:
        gram_style_outputs.append(gram_matrix(s))

    return gram_style_outputs, content_outputs


if __name__ == '__main__':
  content_path = tf.keras.utils.get_file('animal.jpg','https://img.traveltriangle.com/blog/wp-content/uploads/2018/11/Cover-Sweden-Wildlifeepb0310.jpg')
  style_path = tf.keras.utils.get_file('fpa.jpg','https://www.artranked.com/images/3d/3d81e73a76a3040df5907503938e0d8a.jpg')
  extra_path = tf.keras.utils.get_file('lion.jpg', 'https://www.krugerpark.co.za/images/black-maned-lion-shem-compion-590x390.jpg')
  
  content_image, h, w = load_img(content_path)
  style_image = load_img(style_path, h, w)
  noise_image = create_noise_image(h, w)
  #noise_image = load_img(style_path, h, w)
  #noise_image = load_img(noise_image, h,w)
  plt.subplot(1,2,1)
  imshow(content_image, 'Content Image')
  plt.subplot(1,2,2)
  imshow(style_image, 'Style Image')
  plt.show()

  #Get the name of the VGG19 by loading a dummy model
  # vgg_test = tf.keras.applications.VGG19(include_top=False)
  # for i in vgg_test.layers:
  # print(i.name)
  content_layers = ['block5_conv2']
  style_layers = ['block1_conv1', 'block2_conv1','block3_conv1','block4_conv1','block5_conv1'] 

  num_content_layers = len(content_layers)
  num_style_layers = len(style_layers)

  # Style and content model
  vgg = vgg_model(style_layers + content_layers)

  preprocessed_content_image = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
  preprocessed_style_image = tf.keras.applications.vgg19.preprocess_input(style_image * 255)
  content_target = vgg(preprocessed_content_image)
  content_target = content_target[num_style_layers:]
  style_target = vgg(preprocessed_style_image)
  style_target = style_target[:num_style_layers]

  image = tf.Variable(noise_image)

  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  epochs = 1000

  for i in range(epochs):
    train_step(vgg, image, style_target, content_target, num_content_layers, num_style_layers)
    if i % 100 == 0:
      print("Epoch ", i+1, "/", epochs)
      
  PIL_image = tensor_to_image(image)
  PIL_image.save('generated_image.jpg')
  #PIL_image.show()

  plt.subplot(1,3,1)
  imshow(content_image, 'Content Image')

  plt.subplot(1,3,2)
  imshow(style_image, 'Style Image')
  
  plt.subplot(1,3,3)
  generated_image = np.array(PIL_image)
  imshow(generated_image, 'Generated Image')
  plt.show()   
