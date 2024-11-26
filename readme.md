Hi, we are happy to display our workflow here. 
You can follow it and achieve the results as well.


1.	Enter this link: https://www.image-net.org/download-images.php and create account. 
2.	Below the: ImageNet Large-scale Visual Recognition Challenge (ILSVRC),
click the 2010 ImageNet repository.
3.	Download the Training images, Validation images and Test images.
In addition, download the patch for all images.
4.	Extract each data of the first three into the following folders in the project directory -> dataset:
5.	Create a folder dataset in the project directory and extract each data of the first three into the following folders:
Training images-> train_data folder.
Validation images -> val_data folder.
Test images -> test_data folder.
The structure must be:
project name
------dataset
----------- train_data
----------- val_data
----------- test_data

6.	Due to some error in their image collection process, a very small portion of packaged images are blank images returned from websites where the original images have become unavailable. 
ImageNet provides patch images, replace the old images with the new ones in the patch.
You can do it by coping the content of train folder in the patch images and pasting it into the train_data folder. When it asks you if to replace, approve it.
Do the same for the val and the test.
7.	In the same link of ImageNet (2010), download the Development kit and extract the data folder content into dataset.
8.	Run the file create_subfolders.py. Now you have divided_val folder in dataset which contains the val images after dividing to the appropriate classes.
9.	Run the file data_augmentation.py to create patches for the training data, to train the model on more images. Now, you will have the train data in a folder named augmented_data.
10.	Run the file final_model_with_saving.py. 
11.	After the running you will have a text file named error_rates.txt which contains the top-1 and top-5 error rates and a .pth file of the model which was created.
12. In order to run the model on the test images and test the model performance, you have to compare the results the model returns with the true labels of the test images. So you have to download from ImageNet a file named ILSVRC2010_test_ground_truth.txt:
Follow this way:
    go to imagenet site
    go to download tab
    click the 2010 link
    click the link: Ground truth for test data
    save the file (don't change the name: ILSVRC2010_test_ground_truth.txt) to the main project folder
14. Run the file test_with_true_labels.py
15. After the running, you will get 2 text files with the results:
    classification_results.txt will contain the results of the images with the comparison between the true labels and the label the model returned.
    test_error_rates.txt with top-1 and top-5 error rates after the testing.

Good Luck!!!
