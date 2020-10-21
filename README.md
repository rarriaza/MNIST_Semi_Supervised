# MNIST_Semi_Supervised

Classification model for MNIST dataset where just 1 sample per class is available, while the rest of the data is treated as unlabeled.

To run the full code:

python train.py 

Settings.py contains all the parameters with their default values. If it is necesary to change one of the parameters, it is possible by adding to the command --parameter new_value:

python train.py --parameter new_value

The architecture is trained by an iterative algorithm. In each epoch, it trains with the available labeled data. Then it assigns to the unlabeled data "temporal" labels. 
Then, the model is trained on this temporarily-labeled-data. 

This model reached 60% of accuracy on the test data.
