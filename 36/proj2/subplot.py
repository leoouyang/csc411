"""Examples illustrating the use of plt.subplots().

This function creates a figure and a grid of subplots with a single call, while
providing reasonable control over how the individual plots are created.  For
very refined tuning of subplot creation, you can still use add_subplot()
directly on a new figure.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from pylab import *

# Simple data to display in various forms

M = loadmat("mnist_all.mat")
plt.close('all')

# Four axes, returned as a 2-d array
f, axarr = plt.subplots(3, 2)
axarr[0, 0].imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
axarr[0, 0].set_title('Axis [0,0]')
axarr[0, 1].imshow(M["train5"][151].reshape((28,28)), cmap=cm.gray)
axarr[0, 1].set_title('Axis [0,1]')
axarr[1, 0].imshow(M["train5"][152].reshape((28,28)), cmap=cm.gray)
axarr[1, 0].set_title('Axis [1,0]')
axarr[1, 1].imshow(M["train5"][153].reshape((28,28)), cmap=cm.gray)
axarr[1, 1].set_title('Axis [1,1]')
axarr[2, 0].imshow(M["train5"][152].reshape((28,28)), cmap=cm.gray)
axarr[2, 0].set_title('Axis [1,0]')
axarr[2, 1].imshow(M["train5"][153].reshape((28,28)), cmap=cm.gray)
plt.setp(axarr[2, 1].get_xticklabels())
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, :]], visible=False)


plt.show()
