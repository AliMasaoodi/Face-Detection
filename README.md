# Face-Detection

#### To detect face with webcam using Python and OpenCV.

Certainly. The provided code is a Python script that uses OpenCV to perform face recognition. It can be broken down into two main parts.
### Part 1: Create FisherRecognizer and Train Model
•	First, it imports the necessary libraries (cv2 for OpenCV, numpy, os, and sys).
•	A variable named datasets is set to point to the directory that contains image datasets for different faces.
•	The code then collects all grayscale images 'cv2.imread(path, 0)' and their corresponding labels.
•	A face recognizer model (LBPHFaceRecognizer_create()) is then trained using these images and labels.
### Part 2: Use FisherRecognizer on Camera Stream
•	The script activates the webcam and captures frames in real-time.
•	It uses a pre-trained Haar Cascade classifier 'haarcascade_frontalface_default.xml' to identify faces within each frame.
•	The detected faces are then resized and fed into the trained face recognition model.
•	Based on the model’s prediction, text is overlaid on the image to indicate whether the face is recognized or not.
### Noteworthy Observations:
•	The code uses Local Binary Pattern Histogram (LBPH) for face recognition, which is more robust and adaptive compared to other algorithms like Eigenfaces or Fisherfaces.
•	The variable size is declared but not used in the code, which might be an oversight.
•	The prediction[1] < 500 condition seems to be an arbitrary threshold to determine if a face is recognized. A more adaptive or configurable threshold might be better.
•	Syntax and indentation appear to be inconsistent, particularly in the conditional blocks (if and else) which would actually cause a syntax error in Python.
The code is intended to be practical and straightforward for real-time face recognition, although it could benefit from some refinements for better usability and performance.



