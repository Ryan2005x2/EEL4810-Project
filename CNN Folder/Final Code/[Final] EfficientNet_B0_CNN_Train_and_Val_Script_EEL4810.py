# This the final, improved program that trains an EfficientNet B0 CNN model on the USTC Smoke Dataset to predict wildfire from satellite images.
# Written By Ryan Andersen, 4/6/2025

# Core PyTorch Libraries
import torch # Main PyTorch library for tensor operations and model building
import torch.nn as nn # Provides neural network layers (e.g., Linear, Conv2d) and loss functions
import torch.optim as optim # Contains optimization algorithms like Adam, SGD
import torchvision.models as models # Pretrained models like EfficientNet-B0

# Dataset & Preprocessing
import torchvision.datasets as datasets # Tools to load standard datasets or custom image folders
import torchvision.transforms as transforms # Functions to transform images (resize, normalize)
from torch.utils.data import DataLoader, random_split  # DataLoader for batching, random_split for splitting datasets

# Optional Utilities
import matplotlib.pyplot as plt # Plotting library for visualizing losses and accuracies
from tqdm import tqdm # Progress bar utility to show training/validation progress
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

class EfficientNetB0(nn.Module):
    
    def __init__(self, num_classes=6):
        
        # Constructor for the EfficientNetB0 class, inherits from nn.Module
        super(EfficientNetB0, self).__init__() # Call parent class (nn.Module) constructor
        # Load pretrained EfficientNet-B0 with ImageNet weights
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Get the number of input features to the final classifier layer
        in_features = self.model.classifier[1].in_features
        # Replace the final fully connected layer with one matching our number of classes
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        
        # Define how input tensor x flows through the model
        return self.model(x) # Pass input through the EfficientNet-B0 model

def dataset_preprocessing():
    
    # Function to preprocess the dataset with transformations
    # Compose a series of transformations to apply to each image
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize images to 224x224, required by EfficientNet-B0
        transforms.ToTensor(), # Convert PIL images to PyTorch tensors (HWC to CHW)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet mean and std
        transforms.RandomHorizontalFlip()
    ])

    # Load the dataset from the specified directory
    print("Loading dataset...") # Notify user that dataset loading has started
    # Use ImageFolder to load images from subdirectories, applying the defined transforms
    full_dataset = datasets.ImageFolder(root=r'C:\Users\orang\Downloads\USTC_SmokeRS', transform=transform)
    print("Class to index mapping:", full_dataset.class_to_idx)  # Add this line
    print("Dataset loading complete!") # Confirm dataset loading is done
    
    return full_dataset # Return the preprocessed dataset

def dataset_splitting(full_dataset):
    
    # Function to split the full dataset into training and validation sets
    print("Splitting dataset into training and validation sets...") # Announce splitting start
    # Calculate sizes: 80% for training, 20% for validation
    train_size = int(0.8 * len(full_dataset)) # Integer number of training samples
    val_size = len(full_dataset) - train_size # Remaining samples for validation
    # Split the dataset randomly into training and validation subsets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Print the sizes of the resulting datasets for verification
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print("Dataset splitting complete!") # Confirm splitting is finished
    
    return train_dataset, val_dataset # Return the two subsets

def dataset_dataloader(train_dataset, val_dataset):
    
    # Function to create DataLoader objects for batching and shuffling
    print("Creating data loaders...") # Announce DataLoader creation start
    # Create DataLoader for training set with shuffling for better generalization
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    # Create DataLoader for validation set without shuffling (order doesn’t matter)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    print("Data loaders creation complete!") # Confirm DataLoader creation is done
    
    return train_loader, val_loader # Return the two DataLoaders

def verify_preprocessing(train_loader, val_loader, full_dataset):
    
    # Function to verify that preprocessing worked correctly
    print("Verifying dataset preprocessing...") # Announce verification start
    # Get an iterator from the training DataLoader
    data_iter = iter(train_loader)
    # Fetch the first batch of images and labels
    images, labels = next(data_iter)
    
    # Print shapes to verify batch dimensions
    print(f"Shape of a batch of images: {images.shape}")  # Should be [32, 3, 224, 224] for batch_size=32
    print(f"Shape of a batch of labels: {labels.shape}")  # Should be [32] for batch_size=32
    print("Pixel values of the first image:") # Show raw tensor data
    print(images[0])  # Print the tensor for the first image in the batch
    print("Labels for the batch:") # Show corresponding labels
    print(labels)  # Print class indices for the batch
    
    # Get the first image and label directly from the full dataset
    image, label = full_dataset[0]
    # Convert tensor to numpy for inspection
    norm_image = image.numpy()
    print("Normalized image shape:", norm_image.shape) # Should be [3, 224, 224]
    print("Normalized image min/max:", norm_image.min(), norm_image.max()) # Check normalization range
    print("Preprocessing verification complete!") # Confirm verification is done
    
    return None # No return value needed

def efficientnet_b0_model(num_classes=6):
    
    # Function to create and initialize the EfficientNet-B0 model
    print("Initializing EfficientNet-B0 model...") # Announce model initialization
    # Instantiate the custom EfficientNetB0 class with specified number of classes
    model = EfficientNetB0(num_classes=num_classes)
    print("Model initialization complete!") # Confirm model is ready
    
    return model # Return the initialized model

def train_model(model, train_loader, criterion, optimizer, epoch, num_epochs, device=None):
    
    # Function to train the model for one epoch
    # Set device to GPU if available, otherwise CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Move model to the specified device
    
    # Set model to training mode (enables dropout, batch norm updates)
    model.train()
    running_loss = 0.0 # Accumulate loss over the epoch
    correct_train = 0 # Count correct predictions
    total_train = 0 # Count total samples
    
    # Notify user that training is starting for this epoch
    print(f"Starting training for epoch {epoch+1}/{num_epochs}...")
    # Use tqdm to show progress bar for training batches
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
        for inputs, labels in t: # Iterate over batches
            inputs, labels = inputs.to(device), labels.to(device) # Move data to device
            optimizer.zero_grad() # Clear gradients from previous step
            outputs = model(inputs) # Forward pass: compute model predictions
            loss = criterion(outputs, labels) # Compute loss between predictions and labels
            loss.backward() # Backward pass: compute gradients
            optimizer.step() # Update model weights using gradients
            running_loss += loss.item() # Add batch loss to running total
            
            # Calculate accuracy for this batch
            _, preds = torch.max(outputs, 1) # Get predicted class indices
            correct_train += torch.sum(preds == labels).item() # Count correct predictions
            total_train += labels.size(0) # Add batch size to total samples
            
            # Update progress bar with current average loss
            t.set_postfix(loss=running_loss / (t.n + 1))
    
    # Calculate average loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    # Print epoch results
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Training for epoch {epoch+1} complete!") # Confirm training is done
    
    return avg_train_loss, train_accuracy # Return metrics for this epoch

def validate_model(model, val_loader, criterion, epoch, num_epochs, device=None):
    
    # Function to validate the model for one epoch
    # Set device to GPU if available, otherwise CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Move model to the specified device
    
    # Set model to evaluation mode (disables dropout, freezes batch norm)
    model.eval()
    val_loss = 0.0 # Accumulate validation loss
    correct_val = 0 # Count correct predictions
    total_val = 0 # Count total samples
    
    # Notify user that validation is starting for this epoch
    print(f"Starting validation for epoch {epoch+1}/{num_epochs}...")
    # Disable gradient computation for validation (saves memory and computation)
    with torch.no_grad():
        for inputs, labels in val_loader: # Iterate over validation batches
            inputs, labels = inputs.to(device), labels.to(device) # Move data to device
            outputs = model(inputs) # Forward pass: compute predictions
            loss = criterion(outputs, labels) # Compute loss
            val_loss += loss.item() # Add batch loss to running total
            
            # Calculate accuracy for this batch
            _, preds = torch.max(outputs, 1) # Get predicted class indices
            correct_val += torch.sum(preds == labels).item() # Count correct predictions
            total_val += labels.size(0) # Add batch size to total samples
    
    # Calculate average loss and accuracy for validation
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    # Print validation results
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Validation for epoch {epoch+1} complete!") # Confirm validation is done
    
    return avg_val_loss, val_accuracy # Return validation metrics

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    
    # Function to plot training and validation metrics
    print("Plotting training and validation metrics...") # Announce plotting start
    
    # Create a figure with two subplots side by side
    plt.figure(figsize=(12, 6)) # Set figure size to 12x6 inches
    
    # First subplot: Loss
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue') # Plot training loss
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red') # Plot validation loss
    plt.xlabel('Epochs') # Label x-axis
    plt.ylabel('Loss') # Label y-axis
    plt.title('Training and Validation Loss') # Set title
    plt.legend() # Add legend to distinguish lines
    plt.xticks(range(1, num_epochs + 1)) # Set x-axis ticks to integers only (e.g., [1, 2, 3])
    
    # Second subplot: Accuracy
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color='blue') # Plot training accuracy
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='red') # Plot validation accuracy
    plt.xlabel('Epochs') # Label x-axis
    plt.ylabel('Accuracy') # Label y-axis
    plt.title('Training and Validation Accuracy') # Set title
    plt.legend() # Add legend to distinguish lines
    plt.xticks(range(1, num_epochs + 1)) # Set x-axis ticks to integers only (e.g., [1, 2, 3])
    
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show() # Display the plots
    print("Metrics plotting complete!") # Confirm plotting is done
    
    return None # No return value needed

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    
    # Main function to orchestrate the entire script
    print("Starting script execution...") # Announce script start
    
    # Load and preprocess the dataset
    full_dataset = dataset_preprocessing()
    
    # Split dataset into training and validation sets
    train_dataset, val_dataset = dataset_splitting(full_dataset)
    
    # Create DataLoaders for batching
    train_loader, val_loader = dataset_dataloader(train_dataset, val_dataset)
    
    # Verify that preprocessing was applied correctly
    verify_preprocessing(train_loader, val_loader, full_dataset)
    
    # Initialize the EfficientNet-B0 model
    model = efficientnet_b0_model(num_classes=6)
    
    # Set up parameters and objects for training and validation
    print("Setting up training and validation...") # Announce setup start
    num_epochs = 4 # Number of epochs to train
    learning_rate = 5e-4 # Learning rate for the optimizer
    criterion = nn.CrossEntropyLoss() # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer with model parameters
    
    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)  # Reduce lr every 3 epochs by a factor of 0.1
    
    train_losses = [] # List to store training losses
    val_losses = [] # List to store validation losses
    train_accuracies = [] # List to store training accuracies
    val_accuracies = [] # List to store validation accuracies
    print("Setup complete!") # Confirm setup is done
    
    # Start the training and validation loop
    print("Starting training and validation loop...") # Announce loop start
    for epoch in range(num_epochs): # Loop over each epoch
        # Train the model for one epoch and get metrics
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch, num_epochs)
        train_losses.append(train_loss) # Store training loss
        train_accuracies.append(train_acc) # Store training accuracy
        
        # Validate the model for one epoch and get metrics
        val_loss, val_acc = validate_model(model, val_loader, criterion, epoch, num_epochs)
        val_losses.append(val_loss) # Store validation loss
        val_accuracies.append(val_acc) # Store validation accuracy
        
        # Step the scheduler at the end of each epoch
        scheduler.step()
        
    print("Training and validation loop complete!") # Confirm loop is done
    
    # Save the trained model weights:
    torch.save(model.state_dict(), r"C:\Users\orang\Documents\Python Files\efficientnetb0_trained.pth")
    print("Model saved successfully.")
    
    # Plot the collected metrics (i.e., validation and training loss and accuracy, and F1 scores)
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs)
    
    # Finish the script
    print("Script execution complete!")
    
    return None

if __name__ == "__main__":
    # Entry point of the script: run main() if this file is executed directly
    main()