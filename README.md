# pokeGAN
PokeGAN, the WGAN-GP implementation of a Pokémon image dataset.

This project started as a method to build upon my foundational knowledge of Generaltive Adversarial Networks (GAN's) and attempt to create an implementation using Pytorch, a deep learning library I had not yet utilised.
The model used for this GAN went through many iterations, from implementations directly from the original GAN paper, to DCGAN's and eventually the WGAN-GP model which is shown in the code uploaded to this repo. However, to implement a GAN there has to be something for the GAN to attempt to generate... why not create brand new Pokémon? 

The dataset of images used is linked in the acknowledgements section below. This set contains ~800 images of individual Pokémon. I had originally tried to train this network on a batch size of 64 from this dataset alone however I soon realised that dataset augmentation would be needed. The script which I used for data augmentation was written by myself and uploaded as a standalone repo on my profile.

This was run for 500 epochs on a 1050ti graphics card. Even when trying to produce trichannel 64*64 pixel images I struggled to get coherent results. There is definite shape and colour resembling the training data however a pump in resources would be needed to output anything resembling new characters.


## Project Layout

### Training

This script defines the hyperparameters of our WGAN-GP, loads our image dataset (which needs to be in a folder called 'images' within our dataset root directory, opinted at in the datasets.ImageFolder line), transforms the images based on our desired augmentation and implements/trains the model defined in model.py. This also has torchvision implemented to view the progress of the GAN as training is conducted.


### Model 

This script defines the scructure of our Crititc (Determiantor) and Generator and initialises weights as per standard WGAN-GP setup. If ran as __main__ this also conducts a test to check if correct tensor dimentionality is correct throughout the model.

### Utils

This script implements the gradient penalty element of our WGAN, to be used in the training of our model.

### Gen Output

This script allows user to generate images from randomly initialised noise using the generator from model.py after utilising training.py. These are then saved into a user defined folder as PNG files.

## To Do
* Attempt hyperparamter tweaking for better output
* Implement variations of WGAN

## Acknowledgments

* Dataset used: https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types
