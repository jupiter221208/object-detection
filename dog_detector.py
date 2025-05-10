import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class DogDetectorGUI:
    def __init__(self):
        # Initialize the detector
        self.detector = DogDetector()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Dog Detector")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Create image display label
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(pady=10)
        
        # Create result label
        self.result_label = tk.Label(self.main_frame, text="", font=('Arial', 12))
        self.result_label.pack(pady=10)
        
        # Create predictions label
        self.predictions_label = tk.Label(self.main_frame, text="", font=('Arial', 12))
        self.predictions_label.pack(pady=10)
        
        # Create select image button
        self.select_button = tk.Button(
            self.main_frame,
            text="Select New Image",
            command=self.select_and_process_image,
            font=('Arial', 12)
        )
        self.select_button.pack(pady=10)
        
        # Open file dialog when window is created
        self.root.after(100, self.select_and_process_image)

    def select_and_process_image(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_image(file_path)
        else:
            messagebox.showinfo("No Image Selected", "Please select an image to continue.")

    def process_image(self, image_path):
        try:
            # Get predictions
            is_dog, dog_probs, image, all_predictions = self.detector.predict(image_path)
            
            # Resize image for display
            display_size = (400, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Update image display
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Update result text
            if is_dog:
                result_text = "Dog Detected!\n\n"
                for prob, label in dog_probs:
                    result_text += f"{prob:.2%} - {label}\n"
            else:
                result_text = "No dog detected in image"
            
            self.result_label.configure(text=result_text)
            
            # Update all predictions text
            predictions_text = "Top 5 Predictions:\n\n"
            for prob, label in all_predictions:
                predictions_text += f"{prob:.2%} - {label}\n"
            
            self.predictions_label.configure(text=predictions_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")

    def run(self):
        self.root.mainloop()

class DogDetector:
    def __init__(self):
        # Load pre-trained ResNet model with latest weights
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Get ImageNet class labels
        self.categories = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
        
        # Get dog-related class indices
        self.dog_indices = [i for i, category in enumerate(self.categories) 
                          if 'dog' in category.lower()]
        
        print(f"Found {len(self.dog_indices)} dog-related classes")
        print("Dog classes:", [self.categories[i] for i in self.dog_indices[:5]], "...")

    def predict(self, image_path):
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # Get all top 5 predictions
        all_predictions = [(prob.item(), self.categories[catid.item()]) 
                          for prob, catid in zip(top5_prob, top5_catid)]
        
        # Check if any of the top 5 predictions are dogs
        is_dog = any(catid.item() in self.dog_indices for catid in top5_catid)
        
        # Get the highest probability dog prediction
        dog_probs = [(prob.item(), self.categories[catid.item()]) 
                    for prob, catid in zip(top5_prob, top5_catid)
                    if catid.item() in self.dog_indices]
        
        return is_dog, dog_probs, image, all_predictions

    def visualize_prediction(self, image_path):
        is_dog, dog_probs, image = self.predict(image_path)
        
        # Create figure
        plt.figure(figsize=(10, 5))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Display predictions
        plt.subplot(1, 2, 2)
        if is_dog:
            labels = [f"{prob:.2%} - {label}" for prob, label in dog_probs]
            plt.text(0.1, 0.5, "Dog Detected!\n\n" + "\n".join(labels),
                    fontsize=12, va='center')
        else:
            plt.text(0.1, 0.5, "No dog detected in image",
                    fontsize=12, va='center')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create and run the GUI
    app = DogDetectorGUI()
    app.run()

if __name__ == "__main__":
    main() 