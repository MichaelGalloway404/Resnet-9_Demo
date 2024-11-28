import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import cv2
import matplotlib.pyplot as plt
from PIL import Image

'''
This is a demo of a model I trained and presented using the resnet-9 architecture for a research course
at Eastern Oregon University in Fall 2024
The model can distinguish between planes, cars, birds, cats, deer, dogs, frogs, horses, humans, or ships.
It captures an image from the users webcam and classifies it, the user can hold up a picture or object.
Author: Michael Galloway
'''

IMAGE_PATH = './test/test_imgs/test_img.png'
CLASSES = ['Plane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Person','Ship','Unknown']
MODEL_PATH = 'resnet9_obj_analysis.pth'

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
# convert image in subfolder of ./test to tenser and Normalize it, thats what the model was trained on 
valid_data_set = ImageFolder('./test', valid_tfms)

# simple convolutional block
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
            
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # input initialy is 400 x 3 x 32 x 32
        # conv1 output is a featuremap of 400x64x32x32 all will have 400 as it is the batchsize
        self.conv1 = conv_block(in_channels, 64)
        
        self.conv2 = conv_block(64, 128, pool=True)# pool=True to 400x128x16x16
        
        #  self.res1 = nn.Sequential(nn.Sequential(*layers),nn.Sequential(*layers))
        self.res1 = nn.Sequential(conv_block(128, 128),#400x128x16x16
                                  conv_block(128, 128))#this takes its output and adds it
                                  # back in as input(a residual layer)
        
        self.conv3 = conv_block(128, 256, pool=True)# 400x128x16x16 -> 400x256x8x8
        self.conv4 = conv_block(256, 512, pool=True)# 400x256x8x8   -> 400x512x4x4
        self.res2 = nn.Sequential(conv_block(512, 512), # sencond residual block
                                  conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),# we get 400 x 512 x 1 x 1
                                        nn.Flatten(),   # into a single vector
                                        nn.Dropout(0.2),# reduce overfitting
                                        nn.Linear(512, num_classes))# 512 -> 10 outputs(10Lables)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)# final run to get 10 outputs
        return out

# LOAD OUR TRAINED MODEL
                 # ResNet9(in_channels, num_classes)
model = ResNet9(3, 10)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def predict_image(img, model):
    # Convert to a batch of 1
    xb = img.unsqueeze(0)
    # Get predictions from model
    yb = model(xb)
    
    # Pick index with highest probability
    _, predictions = torch.max(yb, dim=1)

    if(_[0].item() <= 5):
        return CLASSES[10]
    
    return CLASSES[predictions[0].item()]

def show_png_image(file_path):
    image = cv2.imread(file_path)
    # fix color representation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.show()

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Can't find camera.")
        return

    # Define the rectangle dimensions (relative to frame size)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rect_width, rect_height = 300, 300
    rect_x = (frame_width - rect_width) // 2
    rect_y = (frame_height - rect_height) // 2

    print("\nPress 'SPACE' to capture an image or 'q' to exit.\n")
    
    print("Show me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship.")
    print("Place image or object in the square.")
    caption = "Press 'space-bar' to analyse or 'q' to exit."
    caption2 = 'Show me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship.'
    caption3 = "Place image or object in the square."
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Draw a rectangle in the center of the frame
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

        # instructions for user camera control are on the screen
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (10, 20)
        fontScale = .5
        color = (255, 0, 0) # Blue
        thickness = 1
        frame = cv2.putText(frame, caption, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        origin = (10, 38)
        frame = cv2.putText(frame, caption2, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        origin = (10, 55)
        color = (0, 0, 255) # Red
        frame = cv2.putText(frame, caption3, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # Display the live feed using OpenCV
        cv2.imshow("Show me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship", frame)

        # Capture keyboard input
        key = cv2.waitKey(1) & 0xFF

        # If 'SPACE' is pressed, capture the image
        if key == ord(' '):
            cropped_image = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
            cv2.imwrite(IMAGE_PATH, cropped_image)
            image = Image.open(IMAGE_PATH)
            
            print("Image we see")
            show_png_image(IMAGE_PATH)
            
            # Resize the image to 32x32
            resized_image = image.resize((32, 32))
            resized_image.save(IMAGE_PATH)
            
            img,_ = valid_data_set[0]

            print("Image Model sees")
            show_png_image(IMAGE_PATH)
                       
            prediction = predict_image(img, model)
            print('Prediction:', prediction)
            
            print("\nShow me a plane, car, bird, cat, deer, dog, frog, horse, human, or ship.")
            print("Place image or object in the square.\n")

        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()



