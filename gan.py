import keras, os, io, math
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.datasets import mnist
from keras.layers import Conv2D, Conv2DTranspose, Input, Dense, MaxPool2D
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from IPython.display import clear_output
from tqdm import tqdm
from tensorflow.python.eager import context
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

def make_tensorboard_images(generator, noise, image_shape):
    """
    Create tensorboard-conform images from noise using generator.
    Based on https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width = image_shape[:2]
    if len(image_shape)>2:
        channels = image_shape[2]
    else:
        channels = 1
    
    images = generator.predict(noise)
    images = ((images + 1)/2*255).astype(np.uint8)
    batch_size = images.shape[0]
    
    if channels>1:
        images = images.reshape(batch_size, height, width, channels)
    else:
        images = images.reshape(batch_size, height, width)
    
    tensorboard_images = []

    for image in images.copy():
        tensorboard_images.append(convert_to_tensorboard_image(image))
    
    return tensorboard_images, images

def convert_to_tensorboard_image(image):
    height, width = image.shape[:2]
    if len(image.shape)>2:
        channels = image.shape[2]
    else:
        channels = 1
    image = Image.fromarray(image)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channels,
                            encoded_image_string=image_string)

def write_log_to_tensorboard(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def write_image_to_tensorboard(callback, names, images, batch_no):
    for name, image in zip(names, images):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=image)])
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def make_summary_image(images):
    n_images = len(images)
    height, width = images[0].shape[:2]
    margin = 10
    
    n_cols = 5
    n_rows = math.ceil(n_images / n_cols)

    summary_plot = np.zeros(((height+margin)*n_rows, (width+margin)*n_cols ))
    image_counter = 0

    for i in range(n_rows):
        for j in range(n_cols):
            summary_plot[(height+margin)*i:(height+margin)*i+height, (width+margin)*j:(width+margin)*j+width] = images[image_counter]
            image_counter += 1
    
    return convert_to_tensorboard_image(summary_plot.astype(np.uint8))

def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)

class PlotCallback(Callback):
    def on_train_begin(self, logs={}):
      self.x = []
      self.xmax = 10
      self.losses = []
      self.val_losses = []

      self.fig = plt.figure()

      self.logs = []
  
    def on_epoch_end(self, epoch, logs=None):
      self.logs.append(logs)
      self.x.append(epoch)
      self.losses.append(logs.get('loss'))
      self.val_losses.append(logs.get('val_loss'))
      if epoch + 2 > self.xmax:
        self.xmax += 10
      
      clear_output(wait=True)
      plt.plot(self.x, self.losses, label="loss")
      plt.plot(self.x, self.val_losses, 'o', label="val_loss")
      plt.legend()
      #plt.ylim([-0.05,max(self.losses)*1.2])
      plt.xlim([0,self.xmax])
      plt.show()

def load_data():
    """
    load images and labels and rescale them to [-1,1] and flatten them
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = 2*x_train.astype(np.float32)/255 - 1
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

def create_generator():
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=784, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return generator

def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return discriminator

def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def training(epochs=1, batch_size=128):
    #Loading the data
    (x_train, y_train, x_test, y_test) = load_data()
    batch_count = x_train.shape[0] // batch_size
    test_batch_size = 20
    test_noise = np.random.normal(0, 1, [test_batch_size, 100])

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)
    
    callback = keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(datetime.now()))
    callback.set_model(gan)

    batch_number = 0

    for epoch_number in range(epochs):
        np.random.shuffle(x_train)
        print("Epoch {}".format(epoch_number))
        for i in tqdm(range(batch_count)):
            # generate random noise as an input to initialize the generator
            noise = np.random.normal(0, 1, [batch_size, 100])

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            image_batch = x_train[i*batch_size:(i+1)*batch_size]

            # Labels for generated and real data
            y_dis_real = np.zeros(batch_size)
            y_dis_real[:batch_size] = 0.9
            y_dis_generated = np.zeros(batch_size)
            

            # Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(image_batch, y_dis_real)
            d_loss_generated = discriminator.train_on_batch(generated_images, y_dis_generated)
            d_loss = (d_loss_real + d_loss_generated)/2
            d_probs = discriminator.predict(image_batch)

            # Tricking the noised input of the Generator as real data
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator 
            # and training the chained GAN model with Discriminator’s weights freezed.
            gan_loss = gan.train_on_batch(noise, y_gen)
            
            gan_probs = gan.predict(noise)

            write_log_to_tensorboard(callback, ['gan_loss'], [gan_loss], batch_number)
            write_log_to_tensorboard(callback, ['d_loss'], [d_loss], batch_number)
            write_log_to_tensorboard(callback, ['gan_probs'], [np.average(gan_probs)], batch_number)
            write_log_to_tensorboard(callback, ['d_probs'], [np.average(d_probs)], batch_number)

            batch_number += 1
        
        test_images_for_tensorboard, test_images = make_tensorboard_images(generator, test_noise, (28,28,1))
        summary_image = make_summary_image(test_images)
        write_image_to_tensorboard(callback, ['Generator images'], [summary_image], epoch_number)
        
        write_image_to_tensorboard(callback, ['Generator image {}'.format(i) for i, _ in enumerate(test_noise)], 
                                   test_images_for_tensorboard, epoch_number)
        
        plot_generated_images(epoch_number, generator)

training(400)