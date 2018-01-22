import os
import math
import numpy as np
import dicom
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import random
import h5py


def montage(imgs, shape):
    assert len(imgs) == np.prod(shape)
    w, h = imgs[0].shape[0], imgs[0].shape[1]
    montage_img = np.zeros((h*shape[0], w*shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            img = imgs[i*shape[1]+j]
            montage_img[i*h:(i+1)*h,j*w:(j+1)*w] = img

    return montage_img


def normalize(imgs):
    # TODO!
    pass


def denormalize(imgs):
    # TODO!
    pass

## Load the Bone image & High kVp image
def load_Img(Path_Data, img_size):
    for dirName, subdirList, fileList in sorted(os.walk(Path_Data)):
        IB_stack = np.zeros((img_size, img_size, subdirList.__len__()), dtype=np.uint16)
        IH_stack = np.zeros((img_size, img_size, subdirList.__len__()), dtype=np.uint16)

        i = 0
        for subdirName in subdirList:
            print ('Loading and Processing Case: %s' % (subdirName))

            IB_raw = dicom.read_file(Path_Data + '/' + subdirName + '/' + 'IB.dcm')
            IH_raw = dicom.read_file(Path_Data + '/' + subdirName + '/' + 'IH1.dcm')

            IB = IB_raw.pixel_array
            IH = IH_raw.pixel_array

            # If IH and IB shape doesn't match, crop to the smaller size
            if IB.shape != IH.shape and abs(IB.shape[0] - IH.shape[0]) < 5 and abs(IB.shape[1] - IH.shape[1]) < 5:
                h_IB = IB[0]
                w_IB = IB[1]
                h_IH = IH[0]
                w_IH = IH[1]

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

            IB_stack[:, :, i] = IB
            IH_stack[:, :, i] = IH
            i = i + 1

        return IB_stack, IH_stack


## Data augmentation with Rotation (-20 to 20 degrees, 5 intervel) & Cropping(-50 to 50 pixels, 20 intervel)
def augment_Img(IB_stack, IH_stack, img_size):
    assert (IB_stack.shape[2] == IH_stack.shape[2])
    IB_aug = []
    IH_aug = []

    num_aug = 40

    for n in range(num_aug):
        print('Augmentation process iteration #%s' % (n + 1))

        for i in range(IB_stack.shape[2]):
            print ('Augmenting %s' % (i))
            IB = IB_stack[:, :, i]
            IH = IH_stack[:, :, i]

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

    IB_aug.append(np.transpose(IB_stack, [2, 0, 1]))
    IH_aug.append(np.transpose(IH_stack, [2, 0, 1]))
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

## main program
if __name__ == '__main__':
    # augment data
    Path_Data = '/Users/Boris/Desktop/CycleGAN-DECC/Data'
    IB_stack, IH_stack = load_Img(Path_Data, img_size=1024)
    IB_aug_stack, IH_aug_stack = augment_Img(IB_stack, IH_stack, img_size=1024)

    # save into .h5 file
    hf_B = h5py.File('IB.h5', 'w')
    hf_B.create_dataset('IB', data=IB_aug_stack)
    hf_B.close()

    hf_B = h5py.File('IH.h5', 'w')
    hf_B.create_dataset('IH', data=IH_aug_stack)
    hf_B.close()


    # # load IB.h5 and IH.h5, check img
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
