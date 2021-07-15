# Data-drive Optimisation of Bike Networks
### Short Explanation

This python package is designed to help to improve the bikeability of cities. Therefore, it combines a route choice model for the cyclists, based on street size and presence or absence of bike paths along streets, OpenStreetMap data and the cyclists demand provided by bike-sharing services. The street network provided by the OpenStreetMap community is described by graphs. To estimate the origin-destination demand I utilise data provided by bike-sharing services such as Call a Bike.

For a more detailed explanation, and a small example take a look at the example folder.

### Data Structure
#### Demand
The bike sharing trip data should be structured as follows: table (e.g. csv) with (at least) the following five columns:
1. latitude of start point
2. longitude of start point
3. latitude of end point
4. longitude of end point
5. number of cyclists which took this trip
#### Area
You have three options for defining the used area for the algorithm.
1. By location name. Sometimes you need to check at [nominatim.openstreetmap.org](https://nominatim.openstreetmap.org) which result you want, as there might multiple results for your request.
2. By bounding box. For this version the bounding box of the given demand data is used, plus a small buffer area.
3. By polygon. Here you pass a polygon for the area you want to use, and the demand is reduced accordingly to the area. A good website to create those jsons is [geojson.io](https://geojson.io/).
#### Data Prepared for the algorithm
With the `prep_city` function from `main.data` you can prepare the aforementioned data for the algorithm. The prepared demand will be saved in `hdf5` format, the used graph in `graphml`.
#### Output of the algorithm
The results of the algorithm will also be stored in `hdf5` format, the logs which are created during the calculations will be saved in simple `txt` files.
#### Plotting he results
To make further analysis (not implemented yet) easier, the data prepared in the plot functions is saved in a separate `hdf5` file.
### Expamples

As mentioned above, in the example folder are two examples, one for Hamburg (hh) and one for Oslo. For Hamburg, I use data provided by [Call a Bike by Deutsche Bahn AG](https://data.deutschebahn.com/dataset/data-call-a-bike) (License: CC BY 4.0) and for Oslo from [Oslo Bysykkel](https://oslobysykkel.no/en/open-data/historical) (License: NLOD 2.0).
