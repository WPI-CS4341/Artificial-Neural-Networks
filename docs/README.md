An Artificial Neural Network
=============================
***(in Python!!!)***

"A what?"
-------
An **[artificial neural network (ANN)](https://en.wikipedia.org/wiki/Artificial_neural_network)** is an abstract concept in which artificial synapses and neurons are used to predict outcomes and classify data (much like a real brain!). It helps computers "think", if you will, and is today's "hot topic" among [deep learning and artificial intelligence enthusiasts](http://wordplay.blogs.nytimes.com/2016/02/01/brilliant-go/?_r=0). This simple ANN reads in data from a pre-formatted file and trains the network to classify any given data point as either a 0 or a 1. The provided file `hw5data.txt` is a collection of 200 x-y points and their respective labels (classifications), courtesy of Microsoft's very own [Christopher Bishop](https://web.archive.org/web/20100825131639/http://research.microsoft.com/en-us/um/people/cmbishop/PRML/webdatasets/datasets.htm).

"Cool! How do I run it?"
------------------------
To run the ANN, make sure you are in a Python environment in which both [NumPy](http://www.numpy.org) and [matplotlib](http://matplotlib.org) is installed. Don't have them? No problem! Just run Python's package manager [pip](https://pip.pypa.io/en/stable/installing/):

    pip install -r requirements.txt

which will install the latest NumPy and matplotlib versions.

**Note:** matplotlib installations have been known to fail in virtual environments on OS X. See [here](http://matplotlib.org/faq/virtualenv_faq.html#osx) for more details.

Once the dependencies are installed, just run:

    python ann.py hw5data.txt h <number of hidden nodes> p <holdout percentage>

where *h* is followed by the number of nodes in the hidden layer of the network and *p* is followed by the percentage of the input data to use for testing the network instead of training the network.

Both options can be omitted from the command. The only required parameter is the file to read from. If *h* is omitted, the network uses 5 nodes in the hidden layer. If *p* is omitted, the network withholds 20% (0.20) of the input data.

"Neat-o! Can I use my own input?"
---------------------------------
If you want to use your own input file, each input line must be in the following format:

    <input1> <input2> <output>

For now, you can only have two inputs and one output. If you wish to have more of either, you must adjust the `parse_file()` function found in `ann.py`.

"Have you experimented with any of this?"
-----------------------------------------
Heck yeah we did! You can check out the experiments we ran on certain network variables [here](ANALYSIS.md).

Acknowledgements
----------------
- This network was written by Yang Liu and Tyler Nickerson of [Worcester Polytechnic Institute (WPI)](http://wpi.edu).
- The back-propagation used in this network is loosely based on the wonderful work of [Andrew Trask](http://iamtrask.github.io/2015/07/12/basic-python-network/).
- As previously mentioned, the data found in `hw5data.txt` was provided by [Christopher Bishop](https://web.archive.org/web/20100825131639/http://research.microsoft.com/en-us/um/people/cmbishop/PRML/webdatasets/datasets.htm) of Microsoft.
