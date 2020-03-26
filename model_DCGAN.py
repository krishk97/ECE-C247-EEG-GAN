# AUTHOR: KRISH KABRA
# Acknowledge must be given to: 
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

# generator and discriminator designed for CWT inputs 
# X = (N_trials, freq_bins=50, time_bins=200, N_eegs = 1)

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy 
from tensorflow.keras import backend as K
from tqdm import tqdm_notebook
from IPython.display import clear_output 
import time

class DCGAN(): 
  def __init__(self,gen_optimizer, disc_optimizer, noise_dim=100,dropout=0):   
    
    # setup config variables eg. noise_dim, hyperparams, verbose, plotting etc. 
    self.noise_dim = noise_dim
    self.dropout = dropout
    self.eeg_img_shape = (50,200,1)

    # Build and compile the discriminator
    self.discriminator = self.build_discriminator()
    # Ensure discriminator is trainable 
    self.discriminator.compile(loss='binary_crossentropy',
            optimizer= disc_optimizer,
            metrics=['accuracy'])

    # Build the generator
    self.generator = self.build_generator()

    # The generator takes noise as input and generates eeg img
    self.combined = self.build_GAN()
    self.combined.compile(loss='binary_crossentropy',
                          optimizer=gen_optimizer)   

    # history variables
    self.loss_history, self.acc_history, self.grads_history = {}, {}, {}

  def build_generator(self):
    model = Sequential()
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

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 50, 200, 1)

    noise = layers.Input(shape=(self.noise_dim,))
    img = model(noise)
    
    return Model(noise, img)

  def build_discriminator(self):
    model = Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.eeg_img_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(self.dropout)) 
    assert model.output_shape == (None,25,100,64) 

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(self.dropout)) 
    assert model.output_shape == (None,25,50,128) 

    # output decision for image -- 1=fake, 0=real
    model.add(layers.Flatten())
    model.add(layers.Dropout(self.dropout))
    model.add(layers.Dense(1,activation='sigmoid'))
    assert model.output_shape == (None,1) 
  
    img = layers.Input(shape=self.eeg_img_shape)
    validity = model(img)
    
    return Model(img, validity)
  
  def build_GAN(self): 
    # Generator takes noise and outputs generated eeg img
    z = layers.Input(shape=(self.noise_dim,))
    generated_eeg = self.generator(z)
    
    # For the combined model we will only train the generator
    discriminator = self.discriminator
    discriminator.trainable = False

    # The discriminator takes generated eeg img as input and determines validity
    validity = discriminator(generated_eeg)

    return Model(z,validity)

  # generate fake data! 
  def generate_fake_data(self,N=100): 
    noise = np.random.normal(0, 1, (N, self.noise_dim))
    gen_imgs = self.generator.predict(noise)
    return gen_imgs, noise
  
  # training loop
  def train(self, train_dataset, epochs=25, batch_size=128,discriminator_iters=1,label_smoothing=0,plot=False):
    '''
    Training loop
    INPUTS: 
    train_dataset - EEG training dataset as numpy array with shape=(trials,eeg,freq_bins,time_bins)
              Assumed dataset has already been normalized! 
    epochs - 
    batch_size - 
    plot - 
    '''
    # init loss history params 
    loss_history, acc_history, grads_history = self.loss_history, self.acc_history, self.grads_history
    gen_grads_history, disc_grads_history, real_grads_history, fake_grads_history = [], [], [], []
    gen_loss_history, disc_loss_history, real_loss_history, fake_loss_history = [], [], [], []
    gen_acc_history, disc_acc_history, real_acc_history, fake_acc_history = [], [], [], []

    # init training dataset that can be shuffled
    X_train = train_dataset.astype('float32') 
  
    for epoch in range(epochs):
      start = time.time()
      
      # shuffle training dataset 
      np.random.shuffle(X_train)

      # batch useful variables 
      num_batches = int(np.ceil(X_train.shape[0] / float(batch_size)))
      
      # grad, loss and acc parameters
      grads_real_l2_norm, grads_fake_l2_norm, grads_disc_l2_norm, grads_gen_l2_norm = 0,0,0,0
      d_loss, d_loss_real, d_loss_fake, g_loss = 0,0,0,0
      d_acc, d_acc_real, d_acc_fake, g_acc = 0,0,0,0

      for batch in tqdm_notebook(range(num_batches)):   
        
        # final batch 
        if batch==num_batches-1: 
          imgs = X_train[batch*batch_size:]
        else: 
          imgs = X_train[batch*batch_size:(batch+1)*batch_size]
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        assert discriminator_iters > 0, 'Number of discriminator must be positive integer'
        for _ in range(discriminator_iters): 
          # Generate batch of fake eeg data for discriminator to train on
          gen_imgs, noise = self.generate_fake_data(N=imgs.shape[0])

          # label smoothing 
          fake = np.zeros((imgs.shape[0],1)) + 0.5 * label_smoothing
          valid = np.ones((imgs.shape[0],1)) * (1.0 - label_smoothing) + 0.5 * label_smoothing

          # Train the discriminator (real classified as ones and generated as zeros)
          d_loss_real_batch, d_acc_real_batch = self.discriminator.train_on_batch(imgs, valid)
          d_loss_fake_batch, d_acc_fake_batch = self.discriminator.train_on_batch(gen_imgs, fake)
          
          # get discriminator gradients at input w/ real and fake imgs 
          inp_real = tf.Variable(imgs,dtype='float32')
          with tf.GradientTape() as tape:
            pred_real = self.discriminator(inp_real)
          grads_real = tape.gradient(pred_real, inp_real).numpy()
        
          inp_fake = tf.Variable(gen_imgs,dtype='float32')
          with tf.GradientTape() as tape: 
            pred_fake = self.discriminator(inp_fake)
        
          grads_fake = tape.gradient(pred_fake, inp_fake).numpy()   
          
          # update grad, loss and acc tracking
          grads_real_l2_norm += np.sqrt(np.sum(np.square(grads_real)))/(float(num_batches)*discriminator_iters)
          grads_fake_l2_norm += np.sqrt(np.sum(np.square(grads_fake)))/(float(num_batches)*discriminator_iters)
          grads_disc_l2_norm += 0.5 * (grads_fake_l2_norm + grads_real_l2_norm)/(float(num_batches)*discriminator_iters)
          d_loss_real += d_loss_real_batch/(float(num_batches)*discriminator_iters)
          d_acc_real += d_acc_real_batch/(float(num_batches)*discriminator_iters)
          d_loss_fake += d_loss_fake_batch/(float(num_batches)*discriminator_iters)
          d_acc_fake += d_acc_fake_batch/(float(num_batches)*discriminator_iters)
          d_loss_batch = 0.5 * (d_loss_real_batch + d_loss_fake_batch)
          d_acc_batch = 0.5 * (d_acc_real_batch + d_acc_fake_batch)
          d_loss += d_loss_batch/(float(num_batches)*discriminator_iters)
          d_acc += d_acc_batch/(float(num_batches)*discriminator_iters)
            
                
        # ---------------------
        #  Train Generator 
        # ---------------------
        # Generate 2*batch of fake eeg data for generator to train on
        gen_imgs, noise = self.generate_fake_data(N=2*imgs.shape[0])
        valid = np.ones((2*imgs.shape[0],1))
        
        # Train the generator (wants discriminator to mistake images as real)      
        g_loss_batch = self.combined.train_on_batch(noise, valid)
        # Manually calculate accuracy to avoid dropout layer 
        g_acc_batch = np.average(np.round(self.combined.predict(noise)))  
        
        # get generator gradients at input
        inp_noise = tf.Variable(np.random.normal(0, 1, (imgs.shape[0], self.noise_dim)),dtype='float32')
        with tf.GradientTape() as tape:
          pred = self.combined(inp_noise)
        
        grads = tape.gradient(pred, inp_noise).numpy()     
        
        # update grad, loss and acc tracking
        grads_gen_l2_norm += np.sqrt(np.sum(np.square(grads)))/float(num_batches)
        g_loss += g_loss_batch/float(num_batches)
        g_acc += g_acc_batch/float(num_batches)
        

        # ---------------------
        # Debugging 
        # ---------------------
        
        # print('Combined GAN batch acc: {}%'.format(100*np.average(np.round(self.combined.predict(noise)))))

        # print('Disc grads: real= {}, fake={}, avg= {}'.format(grads_real_l2_norm,grads_fake_l2_norm,grads_disc_l2_norm))
              
        # print('Gen grads: {}'.format(grads_gen_l2_norm))        
        
      # Save the grad, loss and accuracy histories
      gen_grads_history.append(grads_gen_l2_norm) 
      disc_grads_history.append(grads_disc_l2_norm) 
      real_grads_history.append(grads_real_l2_norm) 
      fake_grads_history.append(grads_fake_l2_norm) 
      gen_loss_history.append(g_loss) 
      disc_loss_history.append(d_loss) 
      real_loss_history.append(d_loss_real) 
      fake_loss_history.append(d_loss_fake) 
      gen_acc_history.append(g_acc)
      disc_acc_history.append(d_acc) 
      real_acc_history.append(d_acc_real) 
      fake_acc_history.append(d_acc_fake) 

      # Plot the progress
      print ('Epoch #: {}/{}, time taken: {} secs \n'.format(epoch+1,epochs,time.time()-start))
      print('Disc:  loss= {}, acc w/ dropout= {}%, grads= {} \n'.format(d_loss,100*d_acc,grads_disc_l2_norm))
      print('Disc Fake:  loss= {}, acc w/ dropout= {}%, grads= {} \n'.format(d_loss_fake,100*d_acc_fake,grads_fake_l2_norm))
      print('Disc Real:  loss= {}, acc w/ dropout= {}%, grads= {} \n'.format(d_loss_real,100*d_acc_real,grads_real_l2_norm))
      print('Gen:  loss= {}, acc w/o dropout= {}%, grads= {} \n'.format(g_loss,100*g_acc,grads_gen_l2_norm))     
      
      if plot: 
        # fake image example
        generated_image,_ = self.generate_fake_data(N=1)       
        # real image example  
        trial_ind, eeg = 0, 0
        real_image = np.expand_dims(train_dataset[trial_ind], axis=0)
        
        # visualize fake and real data examples
        plt.figure()
        plt.subplot(121)
        plt.imshow(generated_image[0, :, :, eeg], aspect='auto')
        plt.colorbar()
        plt.title('Fake decision, eeg {}:\n {}'.format(eeg, self.discriminator.predict(generated_image)))
        plt.subplot(122)
        plt.imshow(real_image[0,:,:,eeg], aspect='auto')
        plt.title('Real decision, trial {}, eeg {}:\n {}'.format(trial_ind, eeg, self.discriminator.predict(real_image)))
        plt.colorbar()
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        # plot discriminator classification
        fake_predictions = self.discriminator.predict(self.generate_fake_data(N=train_dataset.shape[0]))
        real_predictions = self.discriminator.predict(train_dataset)
        plt.figure()
        plt.plot(real_predictions,'bo')
        plt.plot(fake_predictions,'ro')
        plt.legend(['Real', 'Fake'])
        plt.show()

    # Generate after the final epoch
    clear_output(wait=True)
    
    # plot loss history
    plt.figure()
    plt.plot(gen_loss_history, 'r')
    plt.plot(disc_loss_history, 'b')
    plt.plot(real_loss_history, 'g')
    plt.plot(fake_loss_history, 'k')
    plt.title('Loss history')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Generator', 'Discriminator', 'Real', 'Fake'])

    # plot accuracy history
    plt.figure()
    plt.plot(100*gen_acc_history, 'r')
    plt.plot(100*disc_acc_history, 'b')
    plt.plot(100*real_acc_history, 'g')
    plt.plot(100*fake_acc_history, 'k')
    plt.title('Accuracy history')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(['Generator', 'Discriminator', 'Real', 'Fake'])

    # plot grads history
    plt.figure()
    plt.plot(gen_grads_history, 'r')
    plt.plot(disc_grads_history, 'b')
    plt.plot(real_grads_history, 'g')
    plt.plot(fake_grads_history, 'k')
    plt.title('L2-norm of Gradients at input history')
    plt.xlabel('Epochs')
    plt.ylabel('L2-norm of Gradients')
    plt.legend(['Generator', 'Discriminator', 'Real', 'Fake'])
    
    grads_history['Gen'], grads_history['Disc'] = gen_grads_history, disc_grads_history
    grads_history['Real'], grads_history['Fake'] = real_grads_history, fake_grads_history

    loss_history['Gen'], loss_history['Disc'] = gen_loss_history, disc_loss_history
    loss_history['Real'], loss_history['Fake'] = real_loss_history, fake_loss_history
    
    acc_history['Gen'], acc_history['Disc'] = gen_acc_history, disc_acc_history
    acc_history['Real'], acc_history['Fake'] = real_acc_history, fake_acc_history
    

    self.loss_history, self.acc_history = loss_history, acc_history
    
    return loss_history, acc_history, grads_history
