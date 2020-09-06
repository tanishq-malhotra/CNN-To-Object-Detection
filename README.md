# Convolutinal Neural Net to Object Detector or Object Locator


## Used ResetNet50 CNN for Image classification
## We can also use a custom Image Classifier on it


## How It Works
    + Generate Image Pyramids of an image
    + Apply Sliding Window on each of Image Pyramid's
    + Classify the windows
    + Apply Non-Maxima Suppression to remove the unwanted classification with lesser confidance
