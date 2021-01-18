from PIL import Image, ImageFilter
import skimage as ski
import numpy as np
import cupy as cp
import pickle
import os
import tensorflow as tf


def get_batches(x, y, k): 
    p = np.random.permutation(x.shape[0])
    x, y = x[p], y[p]

    print (x[0])
    print(y[0])

    n = x.shape[0] % k
    n = k - n
    x = np.append(x, x[:n], axis=0)
    y = np.append(y, y[:n], axis=0)

    return x, y

def get_image_arrays(path, breeds):
    images_path = path
    
    images = []
    labels = []
    for folder in os.listdir(images_path):
        breed = folder[10:]
        breed_path = images_path + folder + '\\'
        for filename in os.listdir(breed_path):
            image_path = breed_path + filename
            image = Image.open(image_path).convert('RGB')
            images.append(np.array(image))
            labels.append(breeds[breed])

    return images, np.array(labels)

def get_ready_batches(path, extra):
    images = cp.load(path + 'images-224-%s.npy' % (extra))
    labels = cp.load(path + 'labels-224-%s.npy' % (extra))

    images = images.transpose(0, 3, 1, 2)

    for i in range(50):    
        p = cp.random.permutation(labels.shape[0])
        images = images[p]
        labels = labels[p]

    images_and_extra = cp.concatenate((images, images[:156]), axis=0)
    labels_and_extra = cp.concatenate((labels, labels[:156]), axis=0)

    cp.save(dataset_path + "images-extra-224-%s.npy" % (extra), images_and_extra)
    cp.save(dataset_path + "labels-extra-224-%s.npy" % (extra), labels_and_extra)

def resize_images(dataset_path, size):
    images_path = dataset_path + 'images-224\\'
    images_resized_path = dataset_path + 'images-112\\'

    os.mkdir(images_resized_path)

    for folder in os.listdir(images_path):
        breed_path = images_path + folder + '\\'
        resized_breed_path = images_resized_path + folder + '\\'
        os.mkdir(resized_breed_path)
        for photo in os.listdir(breed_path):
            image = Image.open(breed_path + photo)
            image = image.resize((size, size))
            image.save(resized_breed_path + photo)

def tranpose_images(dataset_path):
    images_path = dataset_path + 'images-224\\'
    images_resized_path = dataset_path + 'images-224-flipped\\'

    os.mkdir(images_resized_path)

    for folder in os.listdir(images_path):
        breed_path = images_path + folder + '\\'
        resized_breed_path = images_resized_path + folder + '\\'
        os.mkdir(resized_breed_path)
        for photo in os.listdir(breed_path):
            image = Image.open(breed_path + photo)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image.save(resized_breed_path + photo)

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 1e-5
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape((row,col,ch))

      gauss = gauss * 1e-9
      
      print(gauss.shape)

      Image.fromarray(gauss, mode='RGB').show()
      #Image.fromarray(image, mode='RGB').show()


      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

def add_noise(dataset_path, noise):
    images_path = dataset_path + 'images-224-flipped\\'
    images_resized_path = dataset_path + 'images-224-%s-noise-flipped\\' % (noise)

    os.mkdir(images_resized_path)

    for folder in os.listdir(images_path):
        breed_path = images_path + folder + '\\'
        resized_breed_path = images_resized_path + folder + '\\'
        os.mkdir(resized_breed_path)
        for photo in os.listdir(breed_path):
            image = Image.open(breed_path + photo)
            image = np.array(image)
            image = noisy(noise, image)
            image = Image.fromarray(image, mode='RGB')
            image.save(resized_breed_path + photo)

def remove_part_of_image(dataset_path):
    images_path = dataset_path + 'images-224-flipped\\'
    images_resized_path = dataset_path + 'images-224-flipped-removed\\'

    os.mkdir(images_resized_path)

    for folder in os.listdir(images_path):
        breed_path = images_path + folder + '\\'
        resized_breed_path = images_resized_path + folder + '\\'
        os.mkdir(resized_breed_path)
        for photo in os.listdir(breed_path):
            image = Image.open(breed_path + photo)
            image = np.array(image)
            x = np.random.randint(29, 224)
            y = np.random.randint(29, 224)
            if x > 30:
                px = x - 30
            else:
                px = x + 30
            if y > 30:
                py = y - 30
            else:
                py = y + 30

            
            image[px:x, py:y] = (0, 0, 0)
            
            image = Image.fromarray(image)
            
            image.save(resized_breed_path + photo)



dataset_path = "C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\"

with open(dataset_path + 'breed-dict.pickle', 'rb') as f:
    breeds = pickle.load(f)

