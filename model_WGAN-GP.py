# AUTHOR: KRISH KABRA
# Acknowledgment must be given to drewszurko: 
# github.com/drewszurko/tensorflow-WGAN-GP/blob/797e7c7c8c5861f3f55387635319972d9d224a8f/ops.py#L102

from tensorflow.keras import layers
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K 
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output 
import time

class WGANGP(): 
  def __init__(self, noise_dim=100,dropout=0.2):   
    # setup config variables eg. noise_dim, hyperparams, verbose, plotting etc. 
    self.noise_dim = noise_dim
    self.dropout = dropout
    self.eeg_img_shape = (50,200,5)
        
    # setup history dictionary
    self.history = {}

    # build discriminator and generator models
    self.generator = self.build_generator()
    self.discriminator = self.build_discriminator()

  def build_generator(self):
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*11*512, use_bias=False, input_shape=(self.noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((4, 11, 512)))
    assert model.output_shape == (None, 4, 11, 512) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (5, 4), strides=(2, 2), padding='valid', use_bias=False))
    assert model.output_shape == (None, 11, 24, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (5, 4), strides=(2, 2), padding='valid', use_bias=False))
    assert model.output_shape == (None, 25, 50, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 100, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(5, (5, 5), strides=(1, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 50, 200, 5)

    return model

  def build_discriminator(self):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.eeg_img_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(self.dropout))
    assert model.output_shape == (None,25,100,64) 

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(self.dropout)) # output = (25, 50, 128)
    assert model.output_shape == (None,25,50,128) 

    model.add(layers.Flatten())
    model.add(layers.Dropout(self.dropout))
    model.add(layers.Dense(1))
    assert model.output_shape == (None,1) 
    
    return model
  
  # generate fake data after training! 
  def generate_fake_data(self,N=100): 
    noise = tf.random.normal([N, self.noise_dim]).numpy()
    return generator(noise, training=False).numpy(), noise 
    
  # loss functions
  def disc_loss(self, fake_logits, real_logits):
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

  def gen_loss(self,fake_logits): 
    return - tf.reduce_mean(fake_logits)
  
  # gradient penalty term for discriminator 
  def gradient_penalty(self, discriminator, real_imgs, gen_imgs): 
    eps = tf.random.uniform([real_imgs.shape[0], 1, 1, 1], 0., 1.)
    inter = real_imgs + (eps * (real_imgs - gen_imgs))
    with tf.GradientTape() as tape: 
      tape.watch(inter)
      pred = discriminator(inter)
    
    grad = tape.gradient(pred,inter)[0]
    grad_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grad)))
    
    return tf.reduce_mean(grad_l2_norm) 

  # training functions
  # Notice the use of `tf.function`
  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self,images):
    
    # loss variables to return
    disc_loss, disc_grads = 0,0
    
    # ---------------------
    #  Train Discriminator
    # ---------------------

    # train discriminator over several iterations 
    for _ in range(self.discriminator_iters):
      # setup gradient tools -- GradientTape automatically watches all trainable variables 
      with tf.GradientTape() as disc_tape:
        # forward prop
        noise = tf.random.normal([images.shape[0], self.noise_dim]) 
        gen_imgs = self.generator(noise, training=True)
        fake_logits = self.discriminator(gen_imgs, training=True)
        real_logits = self.discriminator(images, training=True)
        
        # calculate loss
        loss = self.disc_loss(fake_logits,real_logits)
        gp = self.gradient_penalty(partial(self.discriminator, training=True), images, gen_imgs)
        loss += self.gp_weight * gp 

      # back prop      
      disc_grads = disc_tape.gradient(loss, self.discriminator.trainable_variables)
      self.discriminator_optimizer.apply_gradients(zip(disc_grads,self.discriminator.trainable_variables))
      
      # save some variables for history 
      disc_loss += loss
      disc_grads += disc_grads

    # ---------------------
    #  Train Generator 
    # ---------------------
    noise = tf.random.normal([images.shape[0], self.noise_dim])
    with tf.GradientTape() as gen_tape:
      gen_imgs = self.generator(noise, training=True)
      fake_logits = self.discriminator(gen_imgs, training=True)
      gen_loss = self.gen_loss(fake_logits)

    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    self.generator_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

    return disc_loss, disc_grads[0], gen_loss, gen_grads[0]

  # training loop
  def train(self, train_dataset, generator_optimizer, discriminator_optimizer, 
            epochs=25, batch_size=128, discriminator_iters=5,
            gp_weight=0, plot=False, save_plots=False):
    '''
    Training loop
    INPUTS: 
    dataset - EEG training dataset as numpy array with shape=(trials,eeg,freq_bins,time_bins)
    '''

    # set up data for training
    dataset = tf.data.Dataset.from_tensor_slices(train_dataset.astype('float32')).shuffle(train_dataset.shape[0]).batch(batch_size)
    N_batch = np.ceil(train_dataset.shape[0]/float(batch_size))

    # save optimizers
    self.generator_optimizer = generator_optimizer
    self.discriminator_optimizer = discriminator_optimizer

    # save training variables 
    self.discriminator_iters = discriminator_iters
    self.gp_weight = gp_weight

    # setup history variables 
    history = self.history
    history['grads'], history['loss']= {}, {}
    gen_loss_history, disc_loss_history = [],[]
    gen_grads_history, disc_grads_history= [],[]
    
    # start training loop
    for epoch in range(epochs):
      start = time.time()
      
      # refresh loss for every epoch 
      gen_loss, disc_loss, disc_grads, gen_grads = 0, 0, 0, 0
     
      with tqdm(total=N_batch, position=0, leave=True) as pbar:
        for image_batch in dataset:

          # train step           
          disc_loss_batch, disc_grads_batch, gen_loss_batch, gen_grads_batch = self.train_step(image_batch)   
          
          # convert variables to usable format
          disc_loss_batch = tf.reduce_mean(disc_loss_batch).numpy()/float(self.discriminator_iters)
          disc_grads_batch = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(disc_grads_batch)))).numpy()/float(self.discriminator_iters)
          gen_loss_batch = tf.reduce_mean(gen_loss_batch).numpy()
          gen_grads_batch = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(gen_grads_batch)))).numpy()
          
          # store history 
          gen_loss += gen_loss_batch/float(N_batch)
          disc_loss += disc_loss_batch/float(N_batch)
          gen_grads += gen_grads_batch/float(N_batch)
          disc_grads += disc_grads_batch/float(N_batch)

          pbar.update()
      pbar.close()
      
      # store history 
      gen_loss_history.append(gen_loss)
      disc_loss_history.append(disc_loss)
      gen_grads_history.append(gen_grads)
      disc_grads_history.append(disc_grads) 
      

      print ('Epoch #: {}/{}, Time taken: {} secs,\n Grads: disc= {}, gen= {},\n Losses: disc= {}, gen= {}'\
             .format(epoch+1,epochs,time.time()-start, disc_grads, gen_grads, disc_loss, gen_loss))

      if plot and epoch % 20 == 0: 
        # fake image example
        generated_image, _ = self.generate_fake_data(N=1)       
        # real image example  
        trial_ind, eeg = 0, 0
        real_image = np.expand_dims(train_dataset[trial_ind], axis=0)
        
        # visualize fake and real data examples
        plt.figure()
        plt.subplot(121)
        plt.imshow(generated_image[0, :, :, eeg], aspect='auto')
        plt.colorbar()
        plt.title('Fake decision, eeg {}:\n {}'.format(eeg, self.discriminator(generated_image).numpy()))
        plt.subplot(122)
        plt.imshow(real_image[0,:,:,eeg], aspect='auto')
        plt.title('Real decision, trial {}, eeg {}:\n {}'.format(trial_ind, eeg, self.discriminator(real_image).numpy()))
        plt.colorbar()
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        # plot discriminator classification
        gen_imgs, _ = self.generate_fake_data(N=train_dataset.shape[0])
        fake_predictions = self.discriminator(gen_imgs)
        real_predictions = self.discriminator(train_dataset)
        plt.figure()
        plt.plot(real_predictions.numpy(),'bo')
        plt.plot(fake_predictions.numpy(),'ro')
        plt.legend(['Real', 'Fake'])
        plt.show()

    # Generate after the final epoch
    clear_output(wait=True)
    
    plt.figure()
    plt.plot(gen_loss_history, 'r')
    plt.plot(disc_loss_history, 'b')
    plt.title('Loss history')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Generator', 'Discriminator'])
    plt.show()

    plt.figure()
    plt.plot(gen_grads_history, 'r')
    plt.plot(disc_grads_history, 'b')
    plt.title('Gradient history')
    plt.xlabel('Epochs')
    plt.ylabel('Gradients (L2 norm)')
    plt.legend(['Generator', 'Discriminator'])
    plt.show()

    history['grads']['gen'], history['grads']['disc'] = gen_grads_history, disc_grads_history
    history['loss']['gen'], history['loss']['disc'] = gen_loss_history, disc_loss_history   
    
    self.history = history
    
    return history    