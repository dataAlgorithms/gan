#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


# # Model Synthesizers

# In[2]:


def define_GAN(g_model, d_model):
    d_model.trainable = False
    model = keras.models.Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = keras.optimizers.Adam(learning_rate= 0.0002,
                                beta_1= 0.5)
    model.compile(loss= 'binary_crossentropy', optimizer= opt)
    return model


# In[3]:


def define_discriminator(input_shape= (32,32,3)):
    model = keras.models.Sequential()
    # Input is a 32*32*3 image
    model.add(keras.layers.Conv2D(filters= 64,
                                  kernel_size= (3,3),
                                  padding= 'same',
                                  input_shape= input_shape))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Dropout(0.4))
    
    model.add(keras.layers.Conv2D(filters= 64,
                                  kernel_size= (3,3),
                                  strides= (2,2),
                                  padding= 'same'))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Dropout(0.4))
    
    # Input is a 16*16*3 image
    model.add(keras.layers.Conv2D(filters= 128,
                                  kernel_size= (4,4),
                                  strides= (2,2),
                                  padding= 'same'))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Dropout(0.4))
    
    # Input is a 8*8*3 image
    model.add(keras.layers.Conv2D(filters= 256,
                                  kernel_size= (4,4),
                                  strides= (2,2),
                                  padding= 'same'))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Dropout(0.4))
    
    # Input is now 4*4*3
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(units= 1,
                                 activation= 'sigmoid'))
    opt = keras.optimizers.Adam(learning_rate= 0.0002, beta_1= 0.5)
    model.compile(loss= 'binary_crossentropy', optimizer= opt, metrics= ['accuracy'])
    
    return model


# In[4]:


def define_generator(latent_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units= 256 * 4 * 4, input_dim= latent_dim))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Reshape((4, 4, 256)))
    # 8 * 8 now
    model.add(keras.layers.Conv2DTranspose(filters= 128,
                                           kernel_size= (4,4),
                                           padding= 'same',
                                           strides= (2,2)))
    model.add(keras.layers.LeakyReLU(0.2))
    # 16 * 16 now
    model.add(keras.layers.Conv2DTranspose(filters= 128,
                                           kernel_size= (4,4),
                                           padding= 'same',
                                           strides= (2,2)))
    model.add(keras.layers.LeakyReLU(0.2))
    # 32 * 32 now
    model.add(keras.layers.Conv2DTranspose(filters= 128,
                                           kernel_size= (4,4),
                                           padding= 'same',
                                           strides= (2,2)))
    model.add(keras.layers.LeakyReLU(0.2))
    model.add(keras.layers.Conv2D(filters= 3,
                                  kernel_size= (3,3),
                                  padding= 'same',
                                  activation= 'tanh'))
    
    return model


# # Generating points

# In[ ]:


def load_data():
    (X_train, _), (_, _) = keras.datasets.cifar10.load_data()
    return X_train


# In[ ]:


def generate_latent_points(latent_dim, n_samples):
    X = np.random.randn(latent_dim * n_samples)
    X = X.reshape((n_samples, latent_dim))
    return X


# In[ ]:


def generate_real_sample(n_samples):
    data = load_data()
    ix = np.random.randint(0,data.shape[0], n_samples)
    X = data[ix]
    X = X.reshape((n_samples, 32, 32, 3)).astype('float32')
    X = (X - 127.5) / 127.5
    y = np.ones((n_samples, 1))
    return X, y


# In[ ]:


def generate_fake_sample(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim= latent_dim,
                                     n_samples= n_samples)
    
    X = g_model.predict(x_input)
    y = np.ones((n_samples, 1))
    return X, y


# # Summarizing and plotting the model

# In[ ]:


def save_plot(x_input, epoch, n=7):
    x_input = (x_input + 1.0) / 2.0
    filename = f'generated_{epoch + 1}.png'
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.imshow(x_input[i,:,:,:])
        plt.axis('off')
    plt.savefig(filename)
    plt.close()


# In[ ]:


def summarize_the_model(g_model, d_model, epoch, latent_dim, n_samples):
    X_real, y_real = generate_real_sample(n_samples= n_samples)
    X_fake, y_fake = generate_fake_sample(g_model= g_model,
                                          latent_dim= latent_dim,
                                          n_samples= n_samples)
    print(f'Accuracy on real data: {d_model.evaluate(X_real, y_real, verbose= 0)}')
    print(f'Accuracy on fake data: {d_model.evaluate(X_fake, y_fake, verbose= 0)}')
    filename = f'model_e_{epoch + 1}.h5'
    save_plot(x_input= X_fake,
              epoch= epoch)
    
    g_model.save(filename)


# # GAN trainer

# In[ ]:


def train_GAN(gan_model, g_model, d_model, dataset_len, latent_dim, iters= 100, batch_size= 256):
    half_batch = int(batch_size / 2)
    batch_per_epoch = int(dataset_len / batch_size)
    for i in range(iters):
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_sample(n_samples= half_batch)
            X_fake, y_fake = generate_fake_sample(g_model= g_model,
                                                  latent_dim= latent_dim,
                                                  n_samples= half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            d_model.train_on_batch(X, y)
            
            x_gan = generate_latent_points(latent_dim= latent_dim,
                                             n_samples= batch_size)
            y_gan = np.ones((batch_size, 1))
            gan_model.train_on_batch(x_gan, y_gan)
            if not j%10:
                print(f'Epoch: {i+1}, Batches/Epoch: {j+1}/{batch_per_epoch}')
                summarize_the_model(g_model= g_model,
                                    d_model= d_model,
                                    epoch= i,
                                    latent_dim= latent_dim,
                                    n_samples= batch_size)


# # Training the model

# In[ ]:


latent_dim = 100
g_model = define_generator(latent_dim= latent_dim)
d_model = define_discriminator()
gan_model = define_GAN(d_model= d_model,
                       g_model= g_model)
train_GAN(gan_model= gan_model,
          g_model= g_model,
          d_model= d_model,
          dataset_len= load_data().shape[0],
          latent_dim= latent_dim)

