# Art Generation using Neural Style Transfer
This project implements the Neural Style Transfer algorithm created by  [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576), and uses it to generate novel artistic images.

Neural Style Transfer (NST) merges two images, namely: a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.

![image](https://user-images.githubusercontent.com/71698670/117066056-e1f1c900-ad45-11eb-98af-9db0373d5a7e.png)


Following the original NST paper, we will use the VGG-19 network for this project. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers). The weights used in this project can be downloaded from [here.](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)


We will build the Neural Style Transfer (NST) algorithm in three steps:

- Build the content cost function J_content(C,G)
- Build the style cost function J_style(S,G)
- Put it together to get J(G) = α * J_content(C,G) + β * J_style(S,G)       
                      [α and β are hyperparamters]


# References:

The Neural Style Transfer algorithm was due to Gatys et al. (2015). The pre-trained network used in this project is a VGG network, which is due to Simonyan and Zisserman (2015). Pre-trained weights were from the work of the MatConvNet team. 

- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) 
- Karen Simonyan and Andrew Zisserman (2015). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)
