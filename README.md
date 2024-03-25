# MLCell
Machine learning for unit cell determination from XRD

Unit cell inferences from a peak list are performed for each Bravais lattice separately making this a two step process:

1) Predict the Bravais lattice.
2) Predict the unit cell given a Bravais lattice.

Bravais lattice prediction is performed using the code located in the bravais_lattice_classification_models folder. This returns a classification probability for each Bravais lattice given a list of 20 peak locations. Instructions are given at bravais_lattice_classification_models/README.md.

Unit cell inference is performed using the code located in the src/scripts folder. This returns unit cell parameters given a list of 20 peak locations and a specified Bravais lattice. Instructions are given at MLCell/src/scripts/README.md.
