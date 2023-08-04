# drawn_number_classification
This is a simple application that allows you to draw a handwritten digit on a canvas and predicts the number you have drawn. It uses a neural network model to classify the digits.

__Prerequisites__
To run this application, you need to have the following dependencies installed:
Python 3.x
TensorFlow
NumPy
matplotlib
PIL (Python Imaging Library)
tkinter (for GUI)
pickle (for model serialization)

__Installation__
Clone or download this repository to your local machine.
Make sure you have all the required dependencies installed (see Prerequisites section).
Run the model.py file to train the neural network model and save it. (OPTIONAL - THERE IS ALREADY A MODEL.PICKLE FILE AVAILABLE)
Run the main.py file to launch the application.

__How to Use__
Once you run the application, a new window will open with a canvas.
Use your mouse to draw a single digit on the canvas.
Click the "Predict" button to see the predicted number.
Click the "Reset" button to clear the canvas and draw a new digit.

__Model Training__
The neural network model used in this application is trained on the MNIST dataset, which consists of handwritten digit images.

__Customization__
If you want to customize the model or experiment with different configurations, you can modify the model.py file. Here are a few things you can try:
Adjust the number of epochs to train the model for by changing the EPOCHS constant.
Modify the number of neurons and activation functions in the hidden layers by updating the HLAYER_NEURONES and HLAYER_ACTIVATION lists, respectively.
Feel free to explore and adapt the code to suit your needs.
