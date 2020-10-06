# Data-drive Optimisation of Bike Networks
Master thesis project.

### Short Explanation

This pyhton package is designed to help improving the bikeability of cities. Therefore it combines a route choice model for the cyclists, based on street size and presence or absence of bike paths along streets, OpenStreetMap data and the cyclists demand provided by bike-sharing services. The street network provided by the OpenStreetMap community is described by graphs. To estimate the origin-destination demand I utilise data provided by bike-sharing services such as Call a Bike.

### Data Structure

The bike sharing trip data should be structured as follows: table (e.g. csv) with at least the following five columns:
1. latitude of start point
2. longitude of start point
3. latitude of end point
4. longitude of end point
5. number of cyclists which took this trip

todo: Explain output data structure

### Expamples

In the example folder is an example for Hamburg (hh) and Oslo. For Hamburg I use data provided by Call a Bike by Deutsche Bahn AG (License: CC BY 4.0) and for Oslo from Oslo Bysykkel (License: NLOD 2.0).
