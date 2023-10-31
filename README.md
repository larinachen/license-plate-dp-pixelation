# Vehicle Licence Plate Pixelation with Differential Privacy
A differentially private approach to pixelate vehicle license plates caught in images for privacy protection.

## Motivation
With the rise of social media, images are spread constantly without the consent of those in the images. If these images need to be shared with third parties, parts of the images may need to be blurred to protect the privacy of others. However, with the rapid improvements in machine learning, it’s becoming easier to “deblur” an image. Thus, a better method is required to ensure differential privacy. Our goal is to test one such proposed method, apply it to de-identifying vehicle license plates, and see if its validity transfers. 

## Project Thesis
We reproduce the paper [Image Pixelization with Differential Privacy](https://link.springer.com/chapter/10.1007/978-3-319-95729-6_10) by applying the image pixelation technique it proposes to pixelate license plates in images from the public [Car License Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data) from Kaggle . 

## Set up and run
1. install depedencies in requirements.txt.
2. run bounding_box.py, which produces a map of car image ID to the bounding box of the license plate in the image.
3. run pixelization.py, which produces pixelated images and stores them in `output`
