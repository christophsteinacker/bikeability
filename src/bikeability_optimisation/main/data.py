"""
This module includes all necessary functions for the data preparation and
handling.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from ..helper.data_helper import *
from .plot import plot_used_nodes


def prep_city(city_name, save_name,  input_csv, output_folder, polygon_json,
              plot_folder, nominatim_name=None, nominatim_result=1,
              trunk=False, consolidate=False, tol=35, by_bbox=True,
              plot_bbox_size=None, by_city=True, plot_city_size=None,
              by_polygon=True, plot_size=None, cached_graph=False,
              cached_graph_folder=None, cached_graph_name=None, params=None):
    """
    Prepares the data of a city for the algorithm and saves it to the
    desired location.
    :param city_name: Name of the city
    :type city_name: str
    :param save_name: Savename of the city
    :type save_name: str
    :param input_csv: Path to the trip csv
    :type input_csv: str
    :param output_folder: Folder for the data output
    :type output_folder: str
    :param polygon_json: Path to the json of the polygon
    :type polygon_json: str
    :param plot_folder: Folder for the plots
    :type plot_folder: str
    :param nominatim_name: Nominatim name of the city
    :type nominatim_name: str
    :param nominatim_result: results position of the city for the given name
    :type nominatim_result: int
    :param trunk: If trunks should be included or not
    :type trunk: bool
    :param consolidate: If intersections should be consolidated
    :type consolidate: bool
    :param tol: Tolerance of consolidation in meters
    :type tol: float
    :param by_bbox: If graph should be downloaded by the bbox surrounding the
    trips
    :type by_bbox: bool
    :param plot_bbox_size: plot size of the bbox plots [width, height]
    :type plot_bbox_size: list
    :param by_city: If graph should be downloaded by the nominatim name
    :type by_city: bool
    :param plot_city_size: plot size of the nominatim name plots [width,
    height]
    :type plot_city_size: list
    :param by_polygon: If graph should be downloaded by the given polygon
    :type  by_polygon: bool
    :param plot_size: plot size of the polygon plots [width, height]
    :type plot_size: list
    :param cached_graph: If a previously downloaded graph should be used.
    :type cached_graph: bool
    :param cached_graph_folder: Folder of the downloaded graph.
    :type cached_graph_folder: str
    :param cached_graph_name: Name of the downloaded graph.
    :type cached_graph_name: str
    :param params: Dictionary with parameters for plotting etc
    :type params: dict or None
    :return: None
    """
    if nominatim_name is None:
        nominatim_name = city_name
    if plot_size is None:
        plot_size = [20, 20]
    if plot_city_size is None:
        plot_city_size = [20, 20]
    if plot_bbox_size is None:
        plot_bbox_size = [20, 20]
    if cached_graph_folder is None:
        cached_graph_folder = output_folder
    if cached_graph is None:
        cached_graph_name = save_name

    # Check if necessary folders exists, otherwise create.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    if by_bbox:
        # Get bounding box of trips
        print('Getting bbox of trips.')
        bbox = get_bbox_of_trips(input_csv)

        if not cached_graph:
            # Download map given by bbox
            print('Downloading map given by bbox.')
            G_b = download_map_by_bbox(bbox, trunk=trunk,
                                       consolidate=consolidate, tol=tol)
        else:
            print('Loading cached bbox map')
            G_b = ox.load_graphml(filepath=f'{cached_graph_folder}'
                                           f'{cached_graph_name}_bbox.graphml')
            if consolidate:
                G_b = consolidate_nodes(G_b, tol=tol)

        # Loading trips inside bbox
        print('Mapping stations and calculation trips on map given by bbox')
        trips_b, stations_b = load_trips(G_b, input_csv)

        # Colour all used nodes
        print('Plotting used nodes on graph given by bbox.')
        plot_used_nodes(city=city_name, save=f'{save_name}_bbox', G=G_b,
                        trip_nbrs=trips_b, stations=stations_b,
                        plot_folder=plot_folder, params=params)
        fig, ax = ox.plot_graph(G_b, figsize=(20, 20), dpi=300, close=False,
                                show=False)
        fig.suptitle(f'Graph used for {city_name.capitalize()}', fontsize=30)
        plt.savefig(f'{plot_folder}{save_name}_bbox.png', format='png')
        ox.save_graphml(G_b, filepath=f'{output_folder}'
                                      f'{save_name}_bbox.graphml')
        demand_b = h5py.File(f'{output_folder}{save_name}_bbox_demand.hdf5',
                             'w')
        demand_b.attrs['city'] = city_name
        for k, v in trips_b.items():
            grp = demand_b.require_group(f'{k[0]}')
            grp[f'{k[1]}'] = v
        demand_b.close()


    if by_city:
        if not cached_graph:
            # Download whole map of the city
            print('Downloading complete map of city')
            G_c = download_map_by_name(nominatim_name, nominatim_result,
                                       trunk=trunk, consolidate=consolidate,
                                       tol=tol)
        else:
            print('Loading cached map of city')
            G_c = ox.load_graphml(filepath=f'{cached_graph_folder}'
                                           f'{cached_graph_name}_city.graphml')
            if consolidate:
                G_c = consolidate_nodes(G_c, tol=tol)

        # Loading trips inside whole map
        print('Mapping stations and calculation trips on complete map.')
        trips_c, stations_c = load_trips(G_c, input_csv)

        # Colour all used nodes
        print('Plotting used nodes on complete city.')
        plot_used_nodes(city=city_name, save=f'{save_name}_city', G=G_c,
                        trip_nbrs=trips_c, stations=stations_c,
                        plot_folder=plot_folder, params=params)
        fig, ax = ox.plot_graph(G_c, figsize=(20, 20), dpi=300, close=False,
                                show=False)
        fig.suptitle(f'Graph used for {city_name.capitalize()}', fontsize=30)
        plt.savefig(f'{plot_folder}{save_name}_city.png', format='png')
        ox.save_graphml(G_c, filepath=f'{cached_graph_folder}'
                                      f'{save_name}_city.graphml')
        demand_c = h5py.File(f'{output_folder}{save_name}_city_demand.hdf5',
                             'w')
        demand_c.attrs['city'] = city_name
        for k, v in trips_c.items():
            grp = demand_c.require_group(f'{k[0]}')
            grp[f'{k[1]}'] = v
        demand_c.close()

    if by_polygon:
        # Download cropped map (polygon)
        polygon = get_polygon_from_json(polygon_json)

        if not cached_graph:
            print('Downloading polygon.')
            G = download_map_by_polygon(polygon, trunk=trunk,
                                        consolidate=consolidate, tol=tol)
        else:
            print('Loading cached map.')
            G = ox.load_graphml(filepath=f'{cached_graph_folder}'
                                         f'{cached_graph_name}.graphml')
            if consolidate:
                G = consolidate_nodes(G, tol=tol)

        # Loading trips inside the polygon
        print('Mapping stations and calculation trips in polygon.')
        trips, stations = load_trips(G, input_csv, polygon=polygon)

        # Colour all used nodes
        print('Plotting used nodes in polygon.')
        plot_used_nodes(city=city_name, save=save_name, G=G, trip_nbrs=trips,
                        stations=stations, plot_folder=plot_folder)
        fig, ax = ox.plot_graph(G, figsize=(20, 20), dpi=300, close=False,
                                show=False)
        fig.suptitle(f'Graph used for {city_name.capitalize()}', fontsize=30)
        plt.savefig(f'{plot_folder}{save_name}.png', format='png')
        ox.save_graphml(G, filepath=f'{output_folder}{save_name}.graphml')
        demand = h5py.File(f'{output_folder}{save_name}_demand.hdf5',
                           'w')
        demand.attrs['city'] = city_name
        for k, v in trips.items():
            grp = demand.require_group(f'{k[0]}')
            grp[f'{k[1]}'] = v
        demand.close()
