import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('infile')

args = parser.parse_args()

df = pd.read_csv(args.infile, names=['model','dataset','accuracy'])

groups = df.groupby('dataset')

count = groups['accuracy'].count().astype(float)
print(count)
mean = groups['accuracy'].mean()
print("Mean", mean)
sd = groups['accuracy'].std()
print("SD", sd)
sem = sd / np.sqrt(count)
print("SEM", sem)
print("1.96 SEM", 1.96 * sem)
