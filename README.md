# Developing an Image Classifier with Deep Learning
*Image Classifier Project: Udacity - Machine Learning - Introduction Nanodegree Program*

## Project goal

* The first part of the project consists of implementing an image classifier with PyTorch using a Jupyter notebook.
* The second part consists of building a command line application that others can use

### Data

The project is using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories. It can be [downloaded from here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

## Software and Libraries

This project uses the following software and Python libraries:

* [Python](https://www.python.org/downloads/release/python-364/)
* [NumPy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [PyTorch](https://pytorch.org/)

You also need to have additional software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

If you do not have Python installed, I highly recommend installing the [Anaconda](https://www.anaconda.com/distribution/) distribution of Python, which already has the above packages and more included.


### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython notebook finding_donors.ipynb
```  
or
```bash
jupyter notebook finding_donors.ipynb
```

For the command line app

* Train a new network on a data set with ```train.py```
  * Basic usage: ```python train.py data_directory```
  * Prints out training loss, validation loss, and validation accuracy as the network trains
  * Options:
      * Set directory to save checkpoints: 
        ```python train.py data_dir --save_dir save_directory```
      * Choose architecture: 
        ```python train.py data_dir --arch "vgg13"```
      * Set hyperparameters: 
        ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
      * Use GPU for training: 
        ```python train.py data_dir --gpu```
* Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image ```/path/to/image``` and return the flower name and class probability.
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top KK most likely classes: 
      ```python predict.py input checkpoint --top_k 3```
    * Use a mapping of categories to real names: 
      ```python predict.py input checkpoint --category_names cat_to_name.json```
    * Use GPU for inference: 
      ```python predict.py input checkpoint --gpu```

This will open the iPython Notebook software and project file in your browser.

### Note

Both the Jupyter Notebook file and the HMTL version of it are pretty big and cannot load correctly inside GitHub. You have to clone the project locally
