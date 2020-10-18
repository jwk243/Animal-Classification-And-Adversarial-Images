# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

import scipy
from scipy import ndimage, spatial

import skimage
from skimage.transform import rotate

# END IMPORTS

#########################################################
###              BASELINE MODEL
#########################################################

class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.fc = nn.Linear(24 * 8 * 8, 128)
        self.cls = nn.Linear(128, 16)

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 24 * 8 * 8)
        x = F.relu(self.fc(x))
        x = self.cls(x)
        # TODO-BLOCK-END
        return x

def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    # optimizer.zero_grad()
    outputs = net(inputs)
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    net.zero_grad()
    loss = criterion(outputs, labels.squeeze())
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)
    
    total_images = labels.data.numpy().size
    num_correct = torch.sum(predicted == labels.data.reshape(-1))
    running_loss = loss.item()

    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
###               DATA AUGMENTATION
#########################################################

class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        x_shift = random.randrange(-self.max_shift, self.max_shift)
        y_shift = random.randrange(-self.max_shift, self.max_shift)

        image = scipy.ndimage.shift(image, (0, y_shift, x_shift), mode = 'constant')
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        contrast = random.uniform(self.min_contrast, self.max_contrast)
        image[0] = np.multiply(image[0], contrast) + np.multiply((1-contrast), np.mean(image[0]))
        image[1] = np.multiply(image[1], contrast) + np.multiply((1-contrast), np.mean(image[1]))
        image[2] = np.multiply(image[2], contrast) + np.multiply((1-contrast), np.mean(image[2]))
        np.clip(image, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W  = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        angle = random.randrange(-self.max_angle, self.max_angle)

        image = scipy.ndimage.rotate(image, angle, (1,2), reshape = False, mode= 'constant', prefilter=True, order=1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        flip = random.random()
        if flip<=self.p:
            red = image[0]
            green = image[1]
            blue = image[2]
            
            red = np.fliplr(red)
            blue = np.fliplr(blue)
            green = np.fliplr(green)

            image = np.asarray([red,green,blue])

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class GreyScale(object):
    """
    Randomly greyscales an image.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be turned to greyscale
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        flip = random.random()
        if flip<=self.p:
            # grey = np.dot(image[...,:3], [0.299, 0.587, 0.144])
            gray = [[0] * len(image[0][0]) for _ in range(len(image[0]))]
            for i in range(len(image)):
                for y in range(len(image[i])):
                    for x in range(len(image[i][y])):
                        if(i == 0):
                            gray[y][x] += (299/1000) * image[i][y][x]
                        elif(i == 1):
                            gray[y][x] += (587/1000) * image[i][y][x]
                        elif(i == 2):
                            gray[y][x] += (114/1000) * image[i][y][x]
            gray = np.array(gray)

            image = np.asarray([gray,gray,gray])

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class VerticalFlip(object):
    """
    Randomly flips image vertically.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped vertically with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        flip = random.random()
        if flip<=self.p:
            red = image[0]
            green = image[1]
            blue = image[2]
            
            red = np.flipud(red)
            blue = np.flipud(blue)
            green = np.flipud(green)

            image = np.asarray([red,green,blue])

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class CropAndResize(object):
    """
    Randomly flips crops and resizes an image.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be cropped and resized
        
        c          int; maximum amount to crop image
    """
    def __init__(self, p=0.5, x = 10, y= 10):
        self.p = p
        self.x = x
        self.y = y 

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped vertically with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        flip = random.random()
        if flip<=self.p:
            x_shift = random.randrange(0, self.x)
            y_shift = random.randrange(0, self.y)

            startx = W//2-((W - x_shift)//2)
            starty = H//2-((H - y_shift)//2)    
            crop_image = image[ :, starty:starty+(H - y_shift),startx:startx+(W - x_shift)]


            image[0] = cv2.resize(crop_image[0].astype('float32'), dsize=(H,W), interpolation=cv2.INTER_CUBIC)
            image[1] = cv2.resize(crop_image[1].astype('float32'), dsize=(H,W), interpolation=cv2.INTER_CUBIC)
            image[2] = cv2.resize(crop_image[2].astype('float32'), dsize=(H,W), interpolation=cv2.INTER_CUBIC)

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class ScaleDownAndPad(object):
    """
    Randomly scales down and pads an image.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be scaled and padded
        
        c          int; new CxC size
    """
    def __init__(self, p=0.5, c=10):
        self.p = p
        self.c = c

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped vertically with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        flip = random.random()
        if flip<=self.p:
            c_shift = random.randint(0, self.c)
            if(c_shift%2==1):
                c_shift = c_shift + 1
            
            new_w = W - c_shift
            new_h = H - c_shift

            resize_image = [[ ['#' for col in range(3)] for col in range(self.c)] for row in range(self.c)] 
            resize_image[0] = cv2.resize(image[0].astype('float32'), dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)
            resize_image[1] = cv2.resize(image[1].astype('float32'), dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)
            resize_image[2] = cv2.resize(image[2].astype('float32'), dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)

            top = bottom = c_shift//2
            left = right = c_shift//2


            image[0] = cv2.copyMakeBorder(resize_image[0], top, bottom, left, right, cv2.BORDER_CONSTANT)
            image[1] = cv2.copyMakeBorder(resize_image[1], top, bottom, left, right, cv2.BORDER_CONSTANT)
            image[2] = cv2.copyMakeBorder(resize_image[2], top, bottom, left, right, cv2.BORDER_CONSTANT)

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__



#########################################################
###             STUDENT MODEL
#########################################################

def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds  = [ 54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            CropAndResize(p=0.3),
            Contrast(min_contrast=0.3, max_contrast=0.9),
            Shift(max_shift=5),
            Rotate(max_angle=10),
            HorizontalFlip(p=0.3),
            VerticalFlip(p=0.3),
            ScaleDownAndPad(p=0.3, c=15),
            transforms.Normalize(dataset_means, dataset_stds)
        ])
    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN
    batch_size = 64
    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN
    epochs=50
    criterion= nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer

class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        
        self.conv1 = nn.Conv2d(3, 6, 6, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(24, 24, 3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(24, 48, 3, stride=1, padding=1)
        # self.conv7 = nn.Conv2d(48, 96, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(24 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.cls = nn.Linear(128, 16)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.1)


        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 24 * 8 * 8)
        x = self.drop1(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.cls(x)
        # TODO-BLOCK-END
        return x

#########################################################
###             ADVERSARIAL IMAGES
#########################################################

def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN
    img.requires_grad = True

    net.zero_grad()
    loss = criterion(output, label)
    loss.backward()

    grad = img.grad.data
    sign_grad = grad.sign()
    noise = epsilon*sign_grad
    perturbed_image = img + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)



    # TODO-BLOCK-END

    return perturbed_image, noise

