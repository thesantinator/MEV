import numpy as np
import matplotlib.pyplot as plt

# empirical pdf plotter
def emp_pdf(sample, nbins = 10, plot = False):
   '''
   compute and alternatively plot
   plot the empirical frequency density of a sample
   input: sample, nbins
   '''

   '''
      np.histogram(sample, nbins) returns the (count, edge)
      count: absolute frequency
      edge: Bin interval (as an array) e.g. [0, 20, 40, 60, 100]

      nbins can be an int --> equal width histogram or an array --> Defined intervals
   '''
   (counts, edges) = np.histogram(sample, nbins)
   dx = (edges[2]-edges[1])
   dens = counts/dx/np.sum(counts)
   binc = edges[:-1]+dx/2
   if plot == True:
       plt.plot(binc, dens,'ok') # Convert count into relative density (between 0 and 1 and independent of the size of the bin
       plt.show()
   return binc, dens