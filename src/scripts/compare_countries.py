import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
import os
#load the three dataset for the three countries
df_ben= pd.read_csv("data/benin-malaville-clean.csv")
df_sier = pd.read_csv("data/sierraleone-bumbuna-clean.csv")
df_tog = pd.read_csv("data/togo-dapaong-clean.csv")





