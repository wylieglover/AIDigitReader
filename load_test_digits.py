from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import mnist_loader
import network

# creation of our variables, since our load_test_digits.py file is in the same directory/folder as our images, glob extracts our 
# images from the same directory/folder (might need to be changed depending on location of your images)
image_path = Path("Test Digit Images/")
image_files = image_path.glob("*.png")
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
images = []

# format images how test_data is in the mnist_loader so it can be accepted as test_data
i = 0
for image in image_files:
	img = Image.open(image).convert('L')
	img = img.resize((28, 28))
	img = ImageOps.invert(img)
	img = np.array(img, dtype=np.float32) / 255.0
	img = np.reshape(img, (784, 1))
	images.append((img, labels[i]))
	i = i + 1
# sets our network and sends in our images as test_data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 10])
net.SGD(training_data, 10, 20, 3.0, test_data=images)
