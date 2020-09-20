#Digit Recognition using Tensorflow, Keras and OpenCV

ORIGINAL DATA SET is from Char74K -   http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

This project is a simple Digit Classifier used to classify digits from 0 to 9. The Convolutional Neural Network was built using TensorFlow, Keras and OpenCV Python.

Main Libraries used for Project:-
* TensorFlow 2.0.0
* Keras 2.3.1
* OpenCV 4.2.*

Other Libraries include Numpy, scikit-learn, Matplotlib and pickle

The Project has 2 files:
1. OCR_CNN_training.py
2. OCR_CNN_test.py
-----------------------------------------------------------------------------------
#OCR_CNN_training.py
This python file is used for training the Deep learning Neural Network. 
The Images are loaded into the file using OpenCV's imread() function. Each Image is preprocessed and converted to numpy array. The imgaes are normalized for better accuracy.
ImageDataGenerator() function is used for Data Augmentation. 

Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It helps reduce overfitting when training a machine learning. It is clesely related to oversampling in data analysis.

The trained model is saved as model_trained_10.p using pickle.
The Model has 11 layers and uses Adam Optimizer. The Model trains over 10 epochs with each having 2000 steps.
The trained Model is saved using pickle package. It is Later loaded into OCR_CNN_test.py for testing.
----------------------------------------------------------------------------------
#OCR_CNN_test.py
This file uses OpenCV and Numpy. The Trained Model is loaded using pickle. 
Using cv2.VideoCapture() function the video from webcam is captured. The webcam Video feed is converted to Grayscale for better accuracy in prediction. In this the Trained Model is used to predict the Number being displayed on the webcam. 
The output window shows the Number predicted and the probability of the number displayed being correct. 

Note:-
> There is a certain threshold set for predicting the number. If the probability is below the threshold the output won't be shown. 
-----------------------------------------------------------------------------------

The Metrics folder shows the output plot, accuracy etc. The Output Folder has 2 screenshots of output images.

#How to run the project
1. Create a python virtual environment and Install the dependencies stated above.
2. Run the OCR_CNN_training.py file for training the model.
3. Once training is completed, Run OCR_CNN_test.py for checking the output.
----------------------------------------------------------------------------------
