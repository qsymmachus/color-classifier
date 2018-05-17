Color Classifier
================

A neat little project where I try to develop a neural network that can classify RGB values into english color names.

Test data
---------

The test data we train with is found in `color-data.csv`. These are RGB hex values that have been labeled by humans with one of the following color classes:

* green
* blue
* brown
* violet
* red
* orange
* yellow
* black

Usage
-----

Install all dependencies by running:

```
pip -r requirements.txt
```

To train the classifier, run:

```
python color_classifier.py
```
