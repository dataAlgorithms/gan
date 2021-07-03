#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[14]:


import os
curr_dir = os.getcwd() + '/New'
os.makedirs(curr_dir,exist_ok=True)


# In[ ]:





# In[15]:


import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


# ## GAN Model Synthesizers

# In[16]:


def define_gan(g_model, dis_model):
    model = keras.models.Sequential()
    dis_model.trainable = False
    model.add(g_model)
    model.add(dis_model)
    opt = keras.optimizers.Adam(learning_rate= 0.0002,
                                beta_1= 0.5)
    
    model.compile(loss= 'binary_crossentropy',
                  optimizer= opt)
    return model


# ## Discriminator and Generator Model Synthesizers

# In[17]:


def define_generator(latent_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units= 128 * 7 * 7,
                                 input_dim= latent_dim))
    model.add(keras.layers.Reshape((7, 7, 128)))
    
    model.add(keras.layers.Conv2DTranspose(filters= 128,
                                           kernel_size= (4,4),
                                           strides= (2,2),
                                           padding= 'same'))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Conv2DTranspose(filters= 128,
                                           kernel_size= (4,4),
                                           strides= (2,2),
                                           padding= 'same'))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Conv2D(filters= 1,
                                  kernel_size= (7,7),
                                  activation= 'sigmoid',
                                  padding= 'same'))
    return model


# In[ ]:


def define_discriminator(input_shape= (28,28,1)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters= 64,
                                  strides= (2,2),
                                  kernel_size= (3, 3),
                                  padding= 'same',
                                  input_shape= input_shape))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Conv2D(filters= 64,
                                  strides= (2,2),
                                  kernel_size= (3, 3),
                                  padding= 'same',
                                  input_shape= input_shape))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units= 1,
                                 activation= 'sigmoid'))
    opt = keras.optimizers.Adam(learning_rate= 0.0002, beta_1= 0.5)
    model.compile(loss= 'binary_crossentropy', optimizer= opt)
    return model


# ## Generating Real Exampels

# In[ ]:


def load_mnist_data():
    (X_train, _), (_, _) = keras.datasets.mnist.load_data()
    X_train = np.expand_dims(X_train, axis= -1).astype('float32') / 255.0
    return X_train


# In[ ]:


def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


# ## Generating Fake Examples

# In[ ]:


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input


# In[ ]:


def generate_fake_samples(g_model, latent_dim, n_samples):
    X_input = generate_latent_points(latent_dim= latent_dim,
                               n_samples= n_samples)
    X = g_model.predict(X_input)
    y = np.zeros((n_samples, 1))
    return X, y


# ## Summarizing the model

# In[ ]:


def summarize_model(epoch, g_model, d_model, latent_dim, dataset, n_samples= 100):
    X_real, y_real = generate_real_samples(dataset= dataset, n_samples= n_samples)
    X_fake, y_fake = generate_fake_samples(g_model= g_model,
                                           latent_dim= latent_dim,
                                           n_samples= n_samples)
    
    acc_real = d_model.evaluate(X_real, y_real, verbose= 0)
    acc_fake = d_model.evaluate(X_fake, y_fake, verbose= 0)
    print(f'Epoch: {epoch + 1}, Accuracy on real data: {acc_real}, Accuracy on generated data: {acc_fake}')
    save_plot(X_fake, epoch= epoch, n=10)
    model_name = f'./New/generator_model_{epoch + 1}.h5'
    g_model.save(model_name)


# ## Plotting the image

# In[ ]:


def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap= 'gray')
    filename = f'./New/generated_plot_epoch{epoch + 1}.png'
    plt.savefig(filename)
    plt.close()


# ## GAN Model Trainer

# In[ ]:


def train_gan(gan_model, g_model, d_model, dataset, latent_dim, epochs= 100, batch_size= 256):
    half_batch = int(batch_size / 2)
    batch_per_epoch = int(dataset.shape[0]/batch_size)
    for i in range(epochs):
        for j in range(batch_per_epoch):
            # Generating real and fake examples
            X_real, y_real = generate_real_samples(dataset= dataset, n_samples= half_batch)
            X_fake, y_fake = generate_fake_samples(g_model= g_model,
                                                   latent_dim= latent_dim,
                                                   n_samples= half_batch)
            # Stacking the training datas
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # Training the discriminator mode
            d_loss = d_model.train_on_batch(X, y)
            
            # Generating image from latent space
            x_input = generate_latent_points(latent_dim= latent_dim,
                                             n_samples= batch_size)
            
            X_gan = generate_latent_points(latent_dim= latent_dim,
                                           n_samples= batch_size)
            
            y_gan = np.ones((batch_size, 1))
            
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f'Epoch: {i + 1}, batch: {j}/{batch_per_epoch},dloss: {d_loss}, gloss: {g_loss}')
            
        # Saving the model every once in a while
        summarize_model(epoch= i,
                        g_model= g_model,
                        d_model= d_model,
                        dataset= dataset,
                        latent_dim= latent_dim)


# ## Evaluating the model

# In[ ]:


latent_dim = 100
dataset = load_mnist_data()
g_model = define_generator(latent_dim= latent_dim)
d_model = define_discriminator()
gan_model = define_gan(g_model= g_model, dis_model= d_model)

# Training the GAN for MNIST!!
train_gan(gan_model= gan_model,
          g_model= g_model,
          d_model= d_model,
          dataset= dataset,
          latent_dim= latent_dim)

