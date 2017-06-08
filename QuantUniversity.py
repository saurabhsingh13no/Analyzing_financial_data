"""Code to analyse the Stock Market Index data by QuantUniversity
Credits : GOOGLE INC for providing the link to dataset and stubs of code
to get started

Date Created: June 8 2017
Author: QuantUniversity : SS
# add version no.
"""

# Libraries used in the code

import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot,scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import MLPClassifier,MLPRegressor
import logging