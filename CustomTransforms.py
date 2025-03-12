import torch
import numpy as np
import cv2
import imutils

#Reflects the image multiple times until the expected input dimensions are satisfied
class ReflectPad(torch.nn.Module):

    def __init__(self, target_image_width, target_image_height ):
        self.target_image_width = target_image_width
        self.target_image_height = target_image_height

    def __call__(self,image):

        #get the width and height from the original image
        s = image.size()
        width = s[-1]
        height = s[-2]

        #if image is bigger than target size then just return the image
        if width > self.target_image_width and height > self.target_image_height:
            return image

        #convery the tensor into an OpenCV image
        numpy_image = image.numpy()
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        
        #if image is bigger than CNN input (either width or height) then resize it
        if width > self.target_image_width:
            cv2_image = imutils.resize(cv2_image, width=self.target_image_width)
        if height > self.target_image_height:
            cv2_image = imutils.resize(cv2_image, height=self.target_image_height)
        
        #repeat the mirroring until the image is large enough to fill all both width and height target
        while cv2_image.shape[1] < self.target_image_width or cv2_image.shape[0] < self.target_image_height: 
            cv2_image = cv2.copyMakeBorder(cv2_image, height, height , width, width, cv2.BORDER_REFLECT_101)
            
        
        #crop the image to the target width and height from the centre
        mid_x, mid_y = int(cv2_image.shape[1] / 2), int(cv2_image.shape[0] / 2)
        crop_width, crop_height = int(self.target_image_width/2), int(self.target_image_height/2)
        cv2_image = cv2_image[mid_y - crop_height:mid_y + crop_height, mid_x-crop_width:mid_x + crop_width]

        #transform the image back into a pytorch tensor
        new_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        new_image = np.transpose(new_image, (2, 0, 1))
        new_tensor = torch.from_numpy(new_image)
        
        return new_tensor   