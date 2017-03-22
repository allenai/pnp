#!/usr/bin/python
# Generate heatmap of points

import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from heatmap_data import *

# image_name=
# im = plt.imread(image_name);
# implot = plt.imshow(im);

# Load the example flights dataset and conver to long-form
# flights_long = sns.load_dataset("flights")
# flights = flights_long.pivot("month", "year", "passengers")


def sample_kde_data(data):
    u = np.exp(data)
    z = np.sum(u)
    p = (u / z) * 1000

    xs = []
    ys = []
    for yind in xrange(len(p)):
        for xind in xrange(len(p[yind])):
            c = int(p[yind][xind])
            xs += [xind] * c
            ys += [NUM_POINTS - yind] * c

    return (np.array(xs), np.array(ys))


NUM_POINTS=25
def plot_kde(data, cmap):
    (xs, ys) = sample_kde_data(data)
    print len(xs)
    sns.kdeplot(xs, ys, cmap=cmap, shade=True, shade_lowest=False, clip=[[0,NUM_POINTS], [0, NUM_POINTS]], alpha=0.5)


# img = plt.imread("data/dqa_parts_v1/fighter-jet/fighter-jet_0000.png")
img = plt.imread("data/dqa_parts_v1/antelope/antelope_0000.png")

fig, ax = plt.subplots()
ax.imshow(img, extent=[0, NUM_POINTS, 0, NUM_POINTS])

plot_kde(neck_data3, "Blues")
# plot_kde(leg_data2, "Reds")
# plot_kde(tail_data2, "Greens")

plt.axis('off')
plt.show()

# Draw a heatmap with the numeric values in each cell
# sns.heatmap(data, cbar=False, cmap="coolwarm")
# plt.show()

