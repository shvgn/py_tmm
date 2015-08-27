#!/usr/bin/env python

import matplotlib.pylab as plt
import numpy as np

# If you're not familiar with np.r_, don't worry too much about this. It's just 
# a series with points from 0 to 1 spaced at 0.1, and 9 to 10 with the same spacing.
x = np.r_[0:1:0.1, 9:10:0.1]
y = np.sin(x)

fig,(ax,ax2) = plt.subplots(1, 2, sharey=True)
fig.patch.set_facecolor('white')
# plot the same data on both axes
ax.plot(x, y, 'bo')
ax2.plot(x, y, 'bo')

# zoom-in / limit the view to different portions of the data
ax.set_xlim(0,1) # most of the data
ax2.set_xlim(9,10) # outliers only

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labeltop='off') # don't put tick labels at the top
ax2.yaxis.tick_right()

# Make the spacing between the two axes a bit smaller
plt.subplots_adjust(wspace=0.1)

plt.show()


# Can someone please explain why the code below does not work when setting the facecolor of the figure?

# import matplotlib.pyplot as plt

# # create figure instance
fig1 = plt.figure(1)
fig1.set_figheight(11)
fig1.set_figwidth(8.5)

rect = fig1.patch
rect.set_facecolor('red') # works with plt.show().  
                          # Does not work with plt.savefig("trial_fig.png")

ax = fig1.add_subplot(1,1,1)

x = 1, 2, 3
y = 1, 4, 9
ax.plot(x, y)

# plt.show()  # Will show red face color set above using rect.set_facecolor('red')

plt.savefig("trial_fig.png") # The saved trial_fig.png DOES NOT have the red facecolor.

# plt.savefig("trial_fig.png", facecolor='red') # Here the facecolor is red.
# When I specify the height and width of the figure using fig1.set_figheight(11) fig1.set_figwidth(8.5) 
# these are picked up by the command plt.savefig("trial_fig.png"). 
# However, the facecolor setting is not picked up. Why?

# Thanks for your help.



# It's because savefig overrides the facecolor for the background of the figure.

# (This is deliberate, actually... The assumption is that you'd probably want to control 
# the background color of the saved figure with the facecolor kwarg to savefig. 
# It's a confusing and inconsistent default, though!)

# The easiest workaround is just to do 
fig.savefig('whatever.png', facecolor=fig.get_facecolor(), edgecolor='none') 
# (I'm specifying the edgecolor here because the default edgecolor for the actual 
# figure is white, which will give you a white border around the saved figure)

# Hope that helps!



