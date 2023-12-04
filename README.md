# Multi-RND
This project investigates if RND can be improved by considering networks with different depth.
The underlying idea builds up on the inution that deep networks split up the input space using hyperplanes via the ReLU-activations.
Therefore a finer grained decomposition of the space is achieved with increasing depth of the network.

We evaluate for `MNIST`, `KMNIST` and `FashionMNIST` the out of distribution detection by training on a single digit and classifying if the other digits are OOD.

Thanks a lot to:
"Exploration by random network distillation." Burda, Yuri, et al. arXiv preprint arXiv:1810.12894 (2018)