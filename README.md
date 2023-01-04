# LicensePlateDetector

This repo hosts my [streamlit app](https://ryanwerthlicenseplate.streamlit.app/) which identifies license plates in images and attempts to extract the text (License Plate Number).

# Data

The labeled images I used to train my model came from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).

# Technologies

- [YOLO](https://github.com/ultralytics/yolov5) was used to train my model
- [OpenCV](https://opencv.org/) is used for image processing and to run the model on new images
- [Pytesseract](https://pypi.org/project/pytesseract/) is used to extract the text from the license plate
- All code was in Python

# Process

Yolo (You Only Look Once) is a region proposal classification network used to train object detection models. Object detection means that the model is not just i.d.'ing a photo, but actually locating an item in an image an returning the item's bounding box. To train my model I utilized transfer learning which in essense takes a trained neural network and just modifies the last layer of the network. The idea behind transfer learning is that the bulk of the model is already really good at breaking down an image into shapes, lines, patterns etc, and we really just need to alter the very end of it to identify the specific item (license plates in this case). Transfer learning is extremely quick and allowed me to train my model on a GPU in just a few minutes. 

I used Google Colab to actually run my training code because Colab provides free GPU runtimes. 

After I had my trained model I saved the weights as an (ONXX file)[https://onnx.ai/] and added that to my weights folder in this repo.

From there I was able to build my streamlit app around those weights. Streamlit is a very user friendly tool that allows you to build and deploy functional ML apps very easily. If you know python you should have no issues with Streamlit. 

# Take Aways

I am always floored with how quickly you can train a custom neural network using transfer learning. Additionally you don't even need that many training images, I trained my model with 345 images. Incredible. With Yolo, once you have the data cleaned and in the right place (images in one folder and the bounding box coordinates in another) all you need is to write a small configuration file and with one command you are off and training your model. Within minutes I had a model that could grab a license plate from any image.

Surprisingly the most fickle part of this project is the text extraction. I didn't code any of this, instead I utilized the Pytesseract python library. This library takes and image and outputs the text. Again for the ease of use and implimantation it is a pretty incredible tool. However the results for this project seem to be a bit touch and go. I think it partly is because every license plates has different fonts, colors and backgrounds so extracting the text perfectly might take some more pre processing. But more importantly, I think the plate angle is even more important. When looking at a license plate from an angle, the library had a really tough time extracting any text. 

I thought the model would be the hard part and the text extraction would be simple. However, in reality I found it to be the opposite case. 

# Repo
- [Training Notebook](./models/training_files) I ran this in Google Colab and accessed my training data from my Google Drive 
- [Weights](./models/weights)
- [Main Streamlit App File](./blob/main/app.py)
- [Additional Streamlit pages](./pages)
- [Streamlit APP Helper Functions](./src)
