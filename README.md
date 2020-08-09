# BirdImageCNN
A CNN for the classification of images of 225 species of birds

Kaggle dataset link:
https://www.kaggle.com/gpiosenka/100-bird-species

I have tried a sequential scaled down version of VGG-16. It achieved a validation set accuracy of ~93%.

A model based off of google's Inception CNN architectures has achieved ~91% validation accuracy with the only image augmentation being random vertical flipping.
This model has a faster training time than the VGG-16 style one, due to dimensionality reduction from the 1x1 kernel convolutions. I attempt to increase generalisation with more image augmentation in batch generation. The training time on my computer is increased by a factor of around 5, suggesting that the image augmentation itself is a bottleneck.

In all cases, when experimenting with test set predictions, it appears that the models rely more on edge and shape features rather than colour.