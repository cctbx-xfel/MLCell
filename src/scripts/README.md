# Unit cell Regression

Unit cell inferences are performed using the script src/scripts/make_inference.py, given a supplied peak list, in inverse d-spacing with units 1/angstrom, and a specified Bravais lattice.

The script is run using python make_inference.py [options]. The following options are:

* --model-path: This is the full path to the Bravais lattice's model. For performing inferences assuming triclinic (aP Bravais lattice), the model path is 'unit_cell_regression_models/aP/saved_model.pb'
* --peak-positions: ordered list of Bragg peak positions in INVERSE d-space
* --file: text file containing Bragg peak positions (one per line) in INVERSE d-space

The --model-path option is required and one of the --peak-positions or --file options must be specified.
