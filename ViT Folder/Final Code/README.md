Before testing the code make sure that you download the training model, the VIT_plus_GUI, Archive if you want to train the model again, and selected dataset for testing the model

Test Script:
Step by step instruction to implement and use the code

1.	Load the code “VIT_plus_GUI” into your preferred development platform. 
2.	Save the Archive folder somewhere on your computer or google collab
3.	(IF TRAINING MODEL) Change the image directory to where you saved the Archive folder. This is marked with the note “Change Archive Location”
4.  Set the model save path to where you want to save or load wildefire_model.pth
5.  On the bottom you can notice that there is a lot of code commeneted out. If you wanted to retrain the code you could remove the comment and then run the code again to train the dataset.
6. Afterwards you can start running the code
7.	In colab it may ask you for access to your google drive, if so then make sure you allow access.
8. After the code is done running tt will prompt the user to import an image. Import an image from the “SELECTED DATA SET” folder. The images from this folder have not been given to the model for training. 
10.	After importing the image you should get a message that will say “This looks like:” and then it will tell you what it is your imported and for real world implementation it will also tell the user the coordinates of the fire and where the closest firestation is and their phone number to call them directly. 

Additional notes:
⁃	If you are training the model in google collab and you are not changing the number of epochs, expect it to take 5-6 hours to run. Do not let the tab disconnect from the internet nor let your computer turn off or go to sleep. If you do it is possible that you will need to start the training again.
