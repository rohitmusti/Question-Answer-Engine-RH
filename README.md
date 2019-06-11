# Context
This is a baseline implementation of the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) question and answering system. The goals with this repo is to learn about some of the challenges associated with building NLP systems and also to provide a starting point from which to create a more general Q & A dataset based on unlabeled data.

## Set Up

1. Run the set up shell script to ensure that you have all of the local dependencies needed. `sh set_up.sh`. Note, this script is based on `pip3` and you may need to modify if if you are using `conda` or some other package management system.

1. Run `python3 setup.py` to get all of the data into a usable format. 
	

## Credits

Thank you to @chrischute for his work in creating the `layers.py`, `setup.py` files and setting up the original model I started training from.

Thank you to Knowledge Computation Group@HKUST for their original code [here](https://github.com/HKUST-KnowComp/R-Net)
