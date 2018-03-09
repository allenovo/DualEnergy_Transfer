import os
import math
import ipdb
import dicom
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from dicom.filereader import InvalidDicomError

from scipy import ndimage as nd
import scipy.ndimage.filters as fi


def batch_norm(x, epsilon=1e-5, momentum = 0.9):
    FLAGS = tf.app.flags.FLAGS
    train = FLAGS.train
    return x if not FLAGS.bn else \
           tf.contrib.layers.batch_norm(x,
                      decay=momentum, 
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=train)


def montage(imgs, shape):
    assert len(imgs) == np.prod(shape)
    w, h = imgs[0].shape[0], imgs[0].shape[1]
    montage_img = np.zeros((h*shape[0], w*shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            img = imgs[i*shape[1]+j]
            montage_img[i*h:i*h+img.shape[0],j*w:j*w+img.shape[1]] = img

    return montage_img


def normalize(imgs, imgs_max=None):
    if imgs_max is None:
        imgs_max = np.reshape(np.amax(imgs, axis=(1,2)), [-1,1,1,1]).astype(float)
        return (imgs / imgs_max - 0.5) * 2, imgs_max
    else:
        imgs = imgs / imgs_max
        imgs[imgs > 1] = 1
        return (imgs - 0.5) * 2


def denormalize(imgs, imgs_max):
    return (imgs + 1) / 2 * imgs_max


def gaussian_filer_batch(im_batch, sigma=50):
    im_list = range(im_batch.shape[0])
    for i in range(len(im_list)):
        im_list[i] = fi.gaussian_filter(im_batch[i], sigma=sigma, mode='constant', truncate=2.0)

    return np.stack(im_list, axis=0)


def normalize_to_jpeg(img):
    img_max = np.amax(img).astype(float)
    img_min = np.amin(img).astype(float)
    return (img - img_min) / (img_max - img_min) * 255


def load_Img(Path_Data, img_size):
    # Load the Bone image & High kVp image
    for dirName, subdirList, fileList in sorted(os.walk(Path_Data)):
        IB_stack = np.zeros((subdirList.__len__(), img_size, img_size), dtype=np.uint16)
        IH_stack = np.zeros((subdirList.__len__(), img_size, img_size), dtype=np.uint16)

        i = 0
        for subdirName in subdirList:
            print ('Loading and Processing Case: %s' % (subdirName))

            try:
                IB_raw = dicom.read_file(Path_Data + '/' + subdirName + '/' + 'IB.dcm')
                IB = IB_raw.pixel_array
            except:
                IB_raw = Image.open(Path_Data + '/' + subdirName + '/' + 'IB.dcm')
                IB = np.array(IB_raw)

            try:
                IH_raw = dicom.read_file(Path_Data + '/' + subdirName + '/' + 'IH1.dcm')
                IH = IH_raw.pixel_array
            except:
                IH_raw = Image.open(Path_Data + '/' + subdirName + '/' + 'IH1.dcm')
                IH = np.array(IH_raw)

            # If IH and IB shape doesn't match, crop to the smaller size
            if IB.shape != IH.shape and abs(IB.shape[0] - IH.shape[0]) < 5 and abs(IB.shape[1] - IH.shape[1]) < 5:
                h_IB = IB.shape[0]
                w_IB = IB.shape[1]
                h_IH = IH.shape[0]
                w_IH = IH.shape[1]

                h = min(h_IB, h_IH)
                w = min(w_IB, w_IH)

                IB = IB[0:h, 0:w]
                IH = IH[0:h, 0:w]

            # If the image size is not 2000x2000, resize to 2000x2000
            if IB.shape[0] != img_size or IB.shape[1] != img_size:
                IB = IB[80:-20, 6:-6]
                IH = IH[80:-20, 6:-6]

                IB = nd.interpolation.zoom(IB, zoom=(img_size / float(IB.shape[0]), img_size / float(IB.shape[1])),
                                           mode='reflect', order=5)
                IH = nd.interpolation.zoom(IH, zoom=(img_size / float(IH.shape[0]), img_size / float(IH.shape[1])),
                                           mode='reflect', order=5)

            # fig = plt.figure()
            # a = fig.add_subplot(1, 2, 1)
            # imgplot = plt.imshow(IH, cmap='gray')
            # a.set_title('High kVp image')
            #
            # a = fig.add_subplot(1, 2, 2)
            # imgplot = plt.imshow(IB, cmap='gray')
            # a.set_title('Bone image')
            #
            # plt.suptitle(subdirName)
            # plt.show()

            IB_stack[i, :, :] = IB
            IH_stack[i, :, :] = IH
            i = i + 1

        return IB_stack, IH_stack


## Data augmentation with Rotation (-20 to 20 degrees, 5 intervel) & Cropping(-50 to 50 pixels, 20 intervel)
def augment_Img(IB_stack, IH_stack, img_size):
    assert (IB_stack.shape[0] == IH_stack.shape[0])
    IB_aug = []
    IH_aug = []

    num_aug = 20

    for n in range(num_aug):
        print('Augmentation process iteration #%s' % (n + 1))

        for i in range(IB_stack.shape[0]):
            print ('Augmenting %s' % (i))
            IB = IB_stack[i, :, :]
            IH = IH_stack[i, :, :]

            # random sample the Crop for 4 sides
            crop_l = random.randint(1, int(50 * float(img_size)/2000))
            crop_r = random.randint(1, int(50 * float(img_size)/2000))
            crop_up = random.randint(1, int(100 * float(img_size)/2000))
            crop_bot = random.randint(1, int(100 * float(img_size)/2000))

            IB_crop = IB[crop_l:-crop_r, crop_up:-crop_bot]
            IB_crop = nd.interpolation.zoom(IB_crop,
                                            zoom=(img_size / float(IB_crop.shape[0]), img_size / float(IB_crop.shape[1])),
                                            mode='reflect', order=5)
            IH_crop = IH[crop_l:-crop_r, crop_up:-crop_bot]
            IH_crop = nd.interpolation.zoom(IH_crop,
                                            zoom=(img_size / float(IH_crop.shape[0]), img_size / float(IH_crop.shape[1])),
                                            mode='reflect', order=5)

            # random sample the Rotation
            ang = random.randint(-15, 15)

            IB_crop_rot = nd.interpolation.rotate(IB_crop, ang, mode='reflect', order=5, reshape=False)
            IH_crop_rot = nd.interpolation.rotate(IH_crop, ang, mode='reflect', order=5, reshape=False)

            # store in IB_aug & IH_aug for output
            IB_aug.append(np.reshape(IB_crop_rot, (1, img_size, img_size)))
            IH_aug.append(np.reshape(IH_crop_rot, (1, img_size, img_size)))

    IB_aug.append(IB_stack)
    IH_aug.append(IH_stack)
    IB_aug = np.concatenate(IB_aug, axis=0)
    IH_aug = np.concatenate(IH_aug, axis=0)

    # fig = plt.figure()
    # a = fig.add_subplot(1, 2, 1)
    # imgplot = plt.imshow(IH_aug[1, :, :], cmap='gray')
    # a.set_title('High kVp image')
    #
    # a = fig.add_subplot(1, 2, 2)
    # imgplot = plt.imshow(IB_aug[1, :, :], cmap='gray')
    # a.set_title('Bone image')
    #
    # plt.show()

    return IB_aug, IH_aug


def load_from_hdf5(hf_H_path, hf_B_path, is_norm=False):
    hf_B = h5py.File(hf_B_path, 'r')
    Ib_data = hf_B.get('IB')
    hf_H = h5py.File(hf_H_path, 'r')
    Ih_data = hf_H.get('IH')

    # Ih_data_new = np.zeros([Ih_data.shape[0], 256, 256])
    # Ib_data_new = np.zeros([Ib_data.shape[0], 256, 256])
    # for i in range(Ih_data.shape[0]):
    #     Ih_data_new[i] = nd.interpolation.zoom(Ih_data[i],
    #                             zoom=0.25,
    #                             mode='reflect', order=5)
    #     Ib_data_new[i] = nd.interpolation.zoom(Ib_data[i],
    #                             zoom=0.25,
    #                             mode='reflect', order=5)
    # Ih_data = Ih_data_new
    # Ib_data = Ib_data_new
    if is_norm:
        Ib_max = hf_B.get('IB_max')
        Ih_max = hf_H.get('IH_max')
        return Ih_data, Ih_max, Ib_data, Ib_max
    else:
        Ih_data = np.expand_dims(Ih_data, axis=3)
        Ib_data = np.expand_dims(Ib_data, axis=3)
        return Ih_data, Ib_data


## main program
if __name__ == '__main__':
    # augment train data
    Path_Data_Train = os.getcwd() + '/Data/Train'
    Path_Data_Validation = os.getcwd() + '/Data/Validation'
    Path_Data_Test = os.getcwd() + '/Data/Test'

    IB_stack_Train, IH_stack_Train = load_Img(Path_Data_Train, img_size=1024)
    IB_stack_Validation, IH_stack_Validation = load_Img(Path_Data_Validation, img_size=1024)
    IB_stack_Test, IH_stack_Test = load_Img(Path_Data_Test, img_size=1024)
    IB_aug_stack_Train, IH_aug_stack_Train = augment_Img(IB_stack_Train, IH_stack_Train, img_size=1024)

    # save into .h5 file
    # save Train_aug
    hf_B = h5py.File('IB_train3.h5', 'w')
    hf_B.create_dataset('IB', data=IB_aug_stack_Train)
    hf_B.close()

    hf_B = h5py.File('IH_train3.h5', 'w')
    hf_B.create_dataset('IH', data=IH_aug_stack_Train)
    hf_B.close()

    # save Validation
    hf_B = h5py.File('IB_validation3.h5', 'w')
    hf_B.create_dataset('IB', data=IB_stack_Validation)
    hf_B.close()

    hf_B = h5py.File('IH_validation3.h5', 'w')
    hf_B.create_dataset('IH', data=IH_stack_Validation)
    hf_B.close()

    # save Test
    hf_B = h5py.File('IB_test3.h5', 'w')
    hf_B.create_dataset('IB', data=IB_stack_Test)
    hf_B.close()

    hf_B = h5py.File('IH_test3.h5', 'w')
    hf_B.create_dataset('IH', data=IH_stack_Test)
    hf_B.close()

    # # augment data
    # Path_Data = os.getcwd() + '/Data'
    # IB_stack, IH_stack = load_Img(Path_Data, img_size=1024)
    # N = IB_stack.shape[-1]
    # N_train = int(np.floor(N * 0.9))
    # N_test  = N - N_train
    # np.random.permutation(N)
    # # IB_aug_stack, IH_aug_stack = augment_Img(IB_stack[:,:,:N_train], IH_stack[:,:,:N_train], img_size=1024)

    # # save into .h5 file
    # hf_B = h5py.File('IB_train2.h5', 'w')
    # hf_B.create_dataset('IB', data=np.transpose(IB_stack[:,:,:N_train], [2,0,1]))
    # hf_B.close()

    # hf_B = h5py.File('IH_train2.h5', 'w')
    # hf_B.create_dataset('IH', data=np.transpose(IH_stack[:,:,:N_train], [2,0,1]))
    # hf_B.close()

    # hf_B = h5py.File('IB_test2.h5', 'w')
    # hf_B.create_dataset('IB', data=np.transpose(IB_stack[:,:,N_train:], [2,0,1]))
    # hf_B.close()

    # hf_B = h5py.File('IH_test2.h5', 'w')
    # hf_B.create_dataset('IH', data=np.transpose(IH_stack[:,:,N_train:], [2,0,1]))
    # hf_B.close()


    # load IB.h5 and IH.h5, check img
    # hf_B = h5py.File('IB.h5', 'r')
    # n_B = hf_B.get('IB')
    # hf_H = h5py.File('IH.h5', 'r')
    # n_H = hf_H.get('IH')
    #
    # fig = plt.figure()
    # a = fig.add_subplot(1, 2, 1)
    # imgplot = plt.imshow(n_H[500, :, :], cmap='gray')
    # a.set_title('High kVp image')
    #
    # a = fig.add_subplot(1, 2, 2)
    # imgplot = plt.imshow(n_B[500, :, :], cmap='gray')
    # a.set_title('Bone image')
    #
    # plt.show()
    #
    # a = 1
