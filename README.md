# image_classification_simulation


This project implements a set of tools and models to perform multiclass classification of images.

Use Case: classifying client test images based on a catalog of available products.

## Setup

### Clone the repository:

    git clone https://github.com/mila-aia/image_classification_simulation.git

### Install the dependencies:
(it is strongly recommended to create and/or activate a virtual environment before this step)

Go to the folder of the repository and run the following command:

    pip install -e .

## Documentation

### Overview of the framework

![Fig0](docs/figures/fig0.jpg)

### Datasets

#### Office31 Dataset

The main dataset used in this repository is the Office31 dataset. The dataset can be downloaded from this [link](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA). It it recommended to move the dataset under `/examples/data` after downloading and unzipping it. The [Office31 Dataloader](/image_classification_simulation/data/office31_loader.py) will then take care of the processing of the data for training/testing with the available models.

#### Other Datasets

Other open-source datasets are available and can be used with the models implemented in the repository:

* [MNIST](/image_classification_simulation/data/mnist_loader.py)
* [Flowers102](/image_classification_simulation/data/flowers102_loader.py)
* [Omniglot](/image_classification_simulation/data/omniglot_loader.py)

### Models

#### Models for end-to-end classification

* [Standard CNN baseline](/image_classification_simulation/models/classic_cnn_baseline.py): Classic CNN architecture consisting of a succession of convolutional & pooling layers and ending with 3 fully connected linear layers. It includes batch normalization and dropout layers for regularization.
* [ResNet baseline](/image_classification_simulation/models/resnet_baseline.py): Pre-trained ResNet18 architecture that can be fine-tuned (transfer learning) for downstream classification tasks on new image datasets.
* [Vision Transformer baseline](/image_classification_simulation/models/vit_baseline.py): Pre-trained Vision Transformer (ViT) architecture from HuggingFace that can be fine-tuned (transfer learning) for downstream classification tasks on new image datasets.

#### Models for representation learning

* [Convolutional autoencoder baseline](/image_classification_simulation/models/autoencoder_baseline.py): Convolutional autoencoder (AE) architecture that can be used to learn compact hidden features/representations of the data. The bottleneck of the model can be used for subsequent tasks such as image clustering.

#### Models for few shot learning (FSL)

* [Prototypical Networks](/image_classification_simulation/models/protonet.py): FSL model that uses feature representations of the images (these representations can come from the trained classification models previously presented) to assign them to a class (method based on distance between the class prototypes and the representation embeddings of the input images).

#### Models for image clustering

* [Clustering](/image_classification_simulation/models/clustering_tools.py): unsupervised clustering (using K-Means or BIRCH algorithms) that creates class clusters of the data by using learned feaures/representations from a backbone (which can be any of the classification models previously presented).

## Running the code



### Run the code/examples.
Note that the code should already compile at this point.

Running examples can be found under the `examples` folder.

In particular, you will find examples for:
* local machine (e.g., your laptop).
* a slurm cluster.

For both these cases, there is the possibility to run with or without Orion.
(Orion is a hyper-parameter search tool - see https://github.com/Epistimio/orion -
that is already configured in this project)

#### Run locally

For example, to run on your local machine without Orion:

    cd examples/local
    sh run.sh


#### Run on a remote cluster (with Slurm)

First, bring you project on the cluster (assuming you didn't create your
project directly there). To do so, simply login on the cluster and git
clone your project:

    git clone git@github.com:alzaia/image_classification_simulation.git

Then activate your virtual env, and install the dependencies:

    cd image_classification_simulation
    pip install -e .

To run with Slurm, just:

    cd examples/slurm
    sh run.sh

Check the log to see that you got an almost perfect loss (i.e., 0).

#### Run with Orion on the Slurm cluster

This example will run orion for 2 trials (see the orion config file).
To do so, go into `examples/slurm_orion`.
Here you can find the orion config file (`orion_config.yaml`), as well as the config
file (`config.yaml`) for your project (that contains the hyper-parameters).

In general, you will want to run Orion in parallel over N slurm jobs.
To do so, simply run `sh run.sh` N times.

When Orion has completed the trials, you will find the orion db file and the
mlruns folder (i.e., the folder containing the mlflow results).

You will also find the output of your experiments in `orion_working_dir`, which
will contain a folder for every trial.
Inside these folders, you can find the models (the best one and the last one), the config file with
the hyper-parameters for this trial, and the log file.

You can check orion status with the following commands:
(to be run from `examples/slurm_orion`)

    export ORION_DB_ADDRESS='orion_db.pkl'
    export ORION_DB_TYPE='pickleddb'
    orion status
    orion info --name my_exp

### Building docs:

To automatically generate docs for your project, cd to the `docs` folder then run:

    make html

To view the docs locally, open `docs/_build/html/index.html` in your browser.

## Licence

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## YOUR PROJECT README:

* __TODO__
