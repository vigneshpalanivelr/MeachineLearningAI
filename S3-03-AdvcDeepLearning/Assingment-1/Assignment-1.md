```
Assignment 1 will be set up on Taxila main course page and assigned to the groups once those are finalized. In the mean time, here is the problem statement for Assignment 1.

Make every effort to use the virtual infrastructure set up for the course.

Problem Statement

This assignment is about feature extraction via dimensionality reduction using variants of autoencoders.  Use the CIFAR10 dataset provided in Keras, after conversion to gray-level images! Use randomly selected 70% of the dataset as training set and remaining 30% as the validation/test set.

Task 1: Perform standard PCA with 70% of the training dataset and identify the eigenvectors associated with top K eigenvalues with 95% total energy. With these, train a logistic regression classifier to classify the images into 10 classes. Draw the ROC curve for the test dataset. Repeat the same with randomized PCA and compare. [2 marks]

Task 2: Train a single layer autoencoder (with K encoder nodes) with linear activation function and appropriately mean and variance normalized input with constraint that encoder weight matrix and decoder weight matrix are transpose w,r,t, each other and each weight vector has unit magnitude. Compare the eigenvectors obtained in step 1 with those obtained using the autoencoders by clearly displaying  the eigenvectors in Task 1 and weight matrix obtained in Task 2 as gray scale images. Comment on these images. [2 marks]

Task 3: Design and Train an appropriate deep convolutional autoencoder with same (or approximately same) dimension K of latent space. Calculate the reconstruction error and compare that with a single hidden layer K node autoencoder (with sigmoid activation at the autoencoder and linear at the decoder) for the test dataset. What will be the reconstruction error if the hidden nodes are distributed equally (approximately) among 3 hidden layers in a new 3 hidden layer autoencoder with sigmoid activation at the autoencoder and linear at the decoder final layer? [4 marks]

Task 4. Train a deep convolutional autoencoder with MNIST dataset and using extracted features train a MLP classifier with 7 outputs (7 segment LED display) that are representative of 10 digits. For example images of "0" will be classified as

   1

1    1

   0

1     1

   1

7 will be "classified" as

   1

0    1

   0

0    1

   0

Generate the confusion matrix for the corresponding test dataset. [3 marks]

Upload both *.ipynb with all outputs embedded and corresponding *.html (or pdf) files. Marks will be deducted for inadequate training resulting in higher errors in all tasks!
```