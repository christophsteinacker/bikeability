# Bikeability Optimisation
Master thesis project.

### Short Explanation

This pyhton package is designed to improve the bikeability of cities, therefore it uses bike sharing data to build a demand drive bike path network.

### Data Structure

The bike sharing trip data should be structured as follows: table (e.g. csv) with at least the following five columns:
1. latitude of start point
2. longitude of start point
3. latitude of end point
4. longitude of end point
5. number of cyclists which took this trip

todo: Explain output data structure

### Expamples

In the example folder are scripts for Hamburg (hh) and Oslo. Be aware, the data is already cleaned, both cities do not provide the data in this format. The run_{}.py skripts execute the complete programm form data preparation (prep_{}.py) to the core algorithm (plot_{}.py) to result plotting. Depending on the setting, it can take several hours to finish.
The plot_compare.py plots a comparison of the given cities. In order to work, the city-specifc plot scripts have to be executed before.
