Hello! This is the GitHub repository for Group 9's EEL 4810 - Introduction to Deep Learning semester project, titled: "Wildfire Detection and Response: CNNs vs ViTs on Satellite Imagery with Location-Based Alerts," written by Ryan Andersen, Bailey Hitt, and Josiah Lopez.

Here you'll find the complete code base to run the Alert Notification System GUI with either the completed CNN or ViT model, both trained on the USTC SmokeRS Dataset. You can download the dataset here: https://www.kaggle.com/datasets/chandravanshishubham/ustc-smoke-dataset

In the Alert Notification System Folder, you'll find the GUI_CNN2.py file that will run the CNN model, properly adapted to your settings of course.

In the Final Code folder within the ViT folder, you'll find the VIT_plus_GUI.ipynb file, which is the ViT and GUI conveniently placed in one file. There's also a README.md file with further instructions if not clear.

In the Final Code folder within the CNN Folder, you'll find both the [Final]EfficientNet_B0_CNN_Train_and_Val_Script_EEL4810.py and [Final]EfficientNet_B0_CNN_Test_Script_EEL4810.py files, which are the training & validation script and test script for the CNN. Furthermore, in addition to the USTC SmokeRS dataset, 50 new, unseen smoke images were added to the dataset in the Smoke folder within the dataset for robustness against unseen images, so if you want to try out the dataset as I used it, download the zip file 50 new smoke images.zip and add it to the original dataset's smoke folder.

Find all the necessary libraries to use this codebase in the requirements.txt file on the home page.

I hope you find this project interesting, thank you!

- Ryan Andersen
