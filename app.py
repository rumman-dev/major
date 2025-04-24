import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image # Pillow library for image handling
import io # To handle image bytes
import os

from flask import Flask, request, render_template, redirect, url_for, flash

# --- Configuration ---
NUM_CLASSES = 3         # MUST match the number used during training
CLASS_NAMES = ['Cyclone', 'Flood', 'Wildfire']# MUST match the order from training (ImageFolder sorts alphabetically by default)
MODEL_PATH = 'best_student_model.pth' # Path to your saved model
UPLOAD_FOLDER = 'uploads' # Optional: folder to temporarily save uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# --- Re-define the Student Model Architecture ---
# IMPORTANT: This MUST exactly match the architecture used during training
class StudentCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(StudentCNN, self).__init__()
        # Assuming input images are 224x224 RGB
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2) # Output: 16 x 112 x 112
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 16 x 56 x 56

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Output: 32 x 56 x 56
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32 x 28 x 28

        # Calculate the flattened size after conv/pool layers
        self.flattened_size = 32 * 28 * 28
        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.fc(x)
        return x

# --- Load Model ---
# Use CPU for wider compatibility, change to "cuda" if your server has a GPU
device = torch.device("cpu")
model = StudentCNN(num_classes=NUM_CLASSES)
try:
    # Load the state dictionary
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model '{MODEL_PATH}' loaded successfully on {device}.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Make sure it's in the same directory as app.py.")
    exit() # Exit if model can't be loaded
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- Define Image Transformations ---
# Use the validation transforms (without random augmentation)
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key' # Needed for flash messages

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Read image file stream
                img_bytes = file.read()
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                # Preprocess the image
                input_tensor = inference_transform(image)
                input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model
                input_batch = input_batch.to(device)

                # Make prediction
                with torch.no_grad():
                    output = model(input_batch)
                    probabilities = F.softmax(output, dim=1)[0] # Get probabilities for the first (and only) image
                    predicted_idx = torch.argmax(probabilities).item()
                    predicted_class = CLASS_NAMES[predicted_idx]
                    confidence = probabilities[predicted_idx].item() * 100

                print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}%")

                # Pass results to the template
                return render_template('index.html',
                                       prediction=predicted_class,
                                       confidence=f"{confidence:.2f}%")

            except Exception as e:
                flash(f'Error processing image: {e}')
                print(f"Error: {e}")
                return redirect(request.url)
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg')
            return redirect(request.url)

    # Initial page load (GET request)
    return render_template('index.html', prediction=None, confidence=None)

if __name__ == '__main__':
    # Make sure the server is accessible from your network if needed (0.0.0.0)
    # Use debug=True only for development, remove for production
    app.run(host='0.0.0.0', port=5000, debug=True)