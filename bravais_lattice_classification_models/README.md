This classification model accepts 20 peak positions in units of angstroms, ordered from lowest to highest resolution, and returns results for Bravais lattice classification. Dependencies include Tensorflow and sklearn.

Inference is performed using the PointsClassifier object within the PointsClassification_inference.py file. Import the object in a python shell, ipython for instance, and supply the do_inference method with a list of 20 d-spacings.

<img width="898" alt="Screenshot 2024-01-05 at 5 32 00 AM" src="https://github.com/cctbx-xfel/MLCell/assets/94470169/6a4b5ce6-d74e-4a5c-b2a1-b7a0e7440583">
