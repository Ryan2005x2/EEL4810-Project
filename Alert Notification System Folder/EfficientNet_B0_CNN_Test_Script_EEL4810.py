# This is a program that takes a satellite image and predicts if it shows wildfire smoke by running it
# through the previously trained EfficientNet_B0_CNN_Train_and_Val_Script_EEL4810.py.
# Written by Ryan Andersen, 3/6/25

# Core PyTorch Libraries
import torch # Main PyTorch library for tensor operations and model building
import torch.nn as nn # Provides neural network layers (e.g., Linear, Conv2d) and loss functions
import torchvision.models as models # Pretrained models like EfficientNet-B0

# Dataset & Preprocessing
import torchvision.transforms as transforms # Functions to transform images (resize, normalize)
import torch.nn.functional as F # Functional interface for common functions like softmax, ReLU, etc.
from PIL import Image # Import Image for opening images

# Define the same custom EfficientNetB0 class as in the training script
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(weights=None)  # No pretrained weights since loading saved weights from training  model
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# In the training script, datasets.Imagefolder assigns class indices alphabetically
# based on the subfolders in the dataset file. Therefore, the class labels array
# need to match this alphabetical ordering exactly:
class_labels = ['Cloud', 'Dust','Haze','Land','Seaside','Smoke']

def load_image(image_path):
    
    # Load an image from the given path.
    
    raw_image = Image.open(image_path).convert('RGB') # load and convert to RGB
    
    return raw_image

def preprocess_image(raw_image):
    
    # Function to preprocess the dataset with transformations
    
    # Compose a series of transformations to apply to each image
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize images to 224x224, required by EfficientNet-B0
        transforms.ToTensor(), # Convert PIL images to PyTorch tensors (HWC to CHW)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet mean and std
    ])

    image = transform(raw_image)
    
    image = image.unsqueeze(0)  # Add batch dimension (required for model input)
    
    return image

def load_saved_model(model_path):
    
    # Load saved trained model weights from path
    
    # Use the custom EfficientNetB0 class instead of raw efficientnet_b0
    model = EfficientNetB0(num_classes=6) # Match the number of classes from training
    
    model.load_state_dict(torch.load(model_path)) # Load the trained model weights
    
    model.eval() # Set model to evaluation mode
    
    return model

def display_prediction(model, image):
    
    # Display class prediction and probability
    
    # Check for device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)
    
    # Make the prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        
        output = model(image) # Get model predictions (logits)
        
        probabilities = F.softmax(output[0], dim=0)  # Apply softmax to get probabilities

    # Get the predicted class index
    predicted_index = torch.argmax(probabilities).item()
    
    # Map predicted index to class label
    predicted_class = class_labels[predicted_index]

    # Print the predicted class name and probability
    print(f"Predicted class index: {predicted_index}")
    print(f"Prediction probability: {probabilities[predicted_index].item():.4f}")
    
    return predicted_class, probabilities[predicted_index].item()

def main():
    
    image_path = r"C:\Users\orang\Documents\Python Files\Wildfire_Positive_Test_Image(2).png"
    
    model_path = r"C:\Users\orang\Documents\Python Files\efficientnetb0_trained.pth"

    image = load_image(image_path)
    print("Image Loaded Successfully!")
    
    processed_image = preprocess_image(image)
    print("Image Preprocessed Successfully!")
    
    model = load_saved_model(model_path)
    print("Saved Model Loaded Successfully!")
    
    predicted_class, probability = display_prediction(model, processed_image)
    print(f"\033[31m\nPredicted Class: {predicted_class} \n\nProbability: {probability*100:.2f}%\n\033[0m")
    
    print("Script Execution Complete!")
    
    return None

if __name__ == "__main__":
    # Entry point of the script: run main() if this file is executed directly
    main()