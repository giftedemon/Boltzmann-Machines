import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

movies_info = pd.read_csv('movies_metadata.csv')
print(movies_info.head())