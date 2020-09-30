# Examples for Hamburg and Oslo

## How to use
The run_{}.py skripts execute the complete programm from data preparation (prep_{}.py) to the core algorithm (algorithm_{}.py) to result plotting (plot_{}.py).  

Depending on the settings, it can take several hours to finish. For example with an AMD Ryzen 7 3700X it takes for Hamburg between 9.5 minutes 2.5 hours to finish, for Oslo between 2.5 and 47 minutes.

The plot_compare.py plots a comparison of the two cities. In order to work, the city-specifc plot scripts have to be executed beforehand.

## Data Acknowledgement
For Hamburg I use data provided by Call a Bike by Deutsche Bahn AG (https://data.deutschebahn.com/dataset/data-call-a-bike) (License: CC BY 4.0) and for Oslo from Oslo Bysykkel (https://oslobysykkel.no/en/open-data/historical) (License: NLOD 2.0). Be aware, the data is already cleaned, both cities do not provide the data in this exact format.
