# SAIN

> __A SALIENCY-AWARE METHOD FOR ARBITRARY STYLE TRANSFER__  
> _Yitong Li, Jie Ma*_  
> _2023 IEEE International Conference on Image Processing (ICIP), 08-11 October 2023_  
> [Paper](https://ieeexplore.ieee.org/document/10222355)

# Preparation

Create directories and put corresponding data into them according to the pairs below:

         './image/trainC/' : content images for training
         
         './image/trainS/' : style images for training
         
         './mask/trainC/' : content masks for training
         
         './mask/trainS/' : style masks for training
         
         './image/testC/' : content images for testing

         './image/testS/' : style images for testing
         
         './mask/testC/' : content masks for testing
         
         './mask/testS/' : style masks for testing

Note: The saliency masks need to be created using the algorithm achieved at "https://github.com/Kinpzz/UDASOD-UPL"
or other salient object detection methods. And the saliency masks should have the same names as their corresponding 
images. Additionally, you need to download the parameters of pretrained VGG network
"vgg_normalised.pth" at "https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing".

# Train

'''python train.py'''

After running the above code, you will find the trained models every 10000 epoches in a directory named "experiments".
Meanwhile, the log which includes prediction results and calculated loss will be stored in a directory named "log_train".

# Test

'''python test.py'''

After running the above code, you will find the testing results in a directory named "output". It's worth noting that
the generated images are named as "{content_image_name}\_stylized\_{style_image_name}.jpg".
