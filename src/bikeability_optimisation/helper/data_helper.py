"""
This module includes all necessary functions for the data preparation.
"""
import json
import geog
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, to_rgba
from matplotlib.ticker import AutoMinorLocator
from math import ceil, cos, asin, sqrt, pi
from shapely.geometry import Point, Polygon
from bikeability_optimisation.helper.algorithm_helper import get_street_type
from bikeability_optimisation.helper.plot_helper import calc_current_state, \
    magnitude
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_csv(path, delim=','):
    """
    Reads the csv given by path. Delimiter of csv can be chosen by delim.
    All column headers ar converted to lower case.
    :param path: path to load csv from
    :type path: str
    :param delim: delimiter of csv
    :type delim: str
    :return: data frame
    :rtype: pandas DataFrame
    """
    df = pd.read_csv(path, delimiter=delim)
    df.columns = map(str.lower, df.columns)
    return df


def write_csv(df, path):
    """
    Writes given data frame to csv.
    :param df: data frame
    :type df: pandas DataFrame
    :param path: path to save
    :type path: str
    :return: None
    """
    df.to_csv(path, index=False)


def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * \
        (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a))


def get_circle_from_point(lat, long, radius, n_points=20):
    """
    Returns a circle around a lat/long point with given radius.
    :param lat: latitude of point
    :type lat: float
    :param long: longitude of point
    :type long: float
    :param radius: radius of the circle
    :type radius: float
    :param n_points: number of sides of the polygon
    :type n_points: int
    :return: circle (polygon)
    :rtype: shapely Polygon
    """
    p = Point([long, lat])
    angles = np.linspace(0, 360, n_points)
    polygon = geog.propagate(p, angles, radius)
    return Polygon(polygon)


def get_lat_long_trips(path_to_trips, polygon=None, delim=','):
    """
    Returns five lists. The first stores the number of cyclists on this trip,
    the second the start latitude, the third the start longitude,
    the fourth the end latitude, the fifth the end longitude.
    An index corresponds to the same trip in each list.
    :param path_to_trips: path to the compacted trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
     pass it here.
    :type polygon: Shapely Polygon
    :param delim: Delimiter of the trips csv.
    :type delim: str
    :return: number of trips, start lat, start long, end lat, end long
    :rtype: list
    """
    trips = read_csv(path_to_trips, delim=delim)

    if polygon is None:
        start_lat = list(trips['start latitude'])
        start_long = list(trips['start longitude'])
        end_lat = list(trips['end latitude'])
        end_long = list(trips['end longitude'])
        nbr_of_trips = list(trips['number of trips'])
        return nbr_of_trips, start_lat, start_long, end_lat, end_long
    else:
        trips['start in polygon'] = \
            trips[['start latitude', 'start longitude']].apply(
                lambda row: polygon.intersects(Point(row['start longitude'],
                                                     row['start latitude'])),
                axis=1)
        trips['end in polygon'] = \
            trips[['end latitude', 'end longitude']].apply(
                lambda row: polygon.intersects(Point(row['end longitude'],
                                                     row['end latitude'])),
                axis=1)
        trips['in polygon'] = trips[['start in polygon', 'end in polygon']].\
            apply(lambda row: row['start in polygon'] and row['end in polygon'],
                  axis=1)
        start_lat = list(trips.loc[trips['in polygon']]['start latitude'])
        start_long = list(trips.loc[trips['in polygon']]['start longitude'])
        end_lat = list(trips.loc[trips['in polygon']]['end latitude'])
        end_long = list(trips.loc[trips['in polygon']]['end longitude'])
        nbr_of_trips = list(trips.loc[trips['in polygon']]['number of trips'])
        return nbr_of_trips, start_lat, start_long, end_lat, end_long


def get_bbox_of_trips(path_to_trips, polygon=None, delim=','):
    """
    Returns the bbox of the trips given by path_to_trips. If only trips inside
    a polygon should be considered, you can pass it to the polygon param.
    :param path_to_trips: path to the compacted trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
     pass it here.
    :type polygon: Shapely Polygon
    :param delim: Delimiter of the trips csv.
    :type delim: str
    :return: list of bbox [north, south, east, west]
    :rtype: list
    """
    trips_used, start_lat, start_long, end_lat, end_long = \
        get_lat_long_trips(path_to_trips, polygon, delim=delim)
    north = max(start_lat + end_lat) + 0.005
    south = min(start_lat + end_lat) - 0.005
    east = max(start_long + end_long) + 0.01
    west = min(start_long + end_long) - 0.01
    return [north, south, east, west]


def load_trips(G, path_to_trips, polygon=None, nn_method='kdtree', delim=','):
    """
    Loads the trips and maps lat/long of start and end station to nodes in
    graph G. For this the ox.get_nearest_nodes function of osmnx is used.
    :param G: graph used for lat/long to node mapping
    :param path_to_trips: path to the trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
    pass it here.
    :type polygon: Shapely Polygon
    :param nn_method: Method for the nearest node calculation.
    :param delim: Delimiter of the trips csv.
    :type delim: str {None, 'kdtree', 'balltree'}
    :return: dict with trip info and set of stations used.
    trip_nbrs structure: key=(origin node, end node), value=# of cyclists
    """

    nbr_of_trips, start_lat, start_long, end_lat, end_long = \
        get_lat_long_trips(path_to_trips, polygon, delim=delim)

    start_nodes = list(ox.get_nearest_nodes(G, start_long, start_lat,
                                            method=nn_method))
    end_nodes = list(ox.get_nearest_nodes(G, end_long, end_lat,
                                          method=nn_method))

    trip_nbrs = {}
    for trip in range(len(nbr_of_trips)):
        trip_nbrs[(int(start_nodes[trip]), int(end_nodes[trip]))] = \
            int(nbr_of_trips[trip])

    stations = set()
    for k, v in trip_nbrs.items():
        stations.add(k[0])
        stations.add(k[1])

    trip_nbrs_rexcl = {k: v for k, v in trip_nbrs.items() if not k[0] == k[1]}
    print('Number of Stations: {}, Number of trips: {} (rt incl: {}), '
          'Unique trips: {} (rt incl {})'
          .format(len(stations), sum(trip_nbrs_rexcl.values()),
                  sum(trip_nbrs.values()), len(trip_nbrs_rexcl.keys()),
                  len(trip_nbrs.keys())))
    return trip_nbrs, stations


def plot_used_nodes(G, trip_nbrs, stations, place, save, width=20, height=20,
                    dpi=300, plot_save_folder=''):
    """
    Plots usage of nodes in graph G. trip_nbrs and stations should be
    structured as returned from load_trips().
    :param G: graph to plot in.
    :type G: networkx graph
    :param trip_nbrs: trips to plot the usage of.
    :type trip_nbrs: dict
    :param stations: set of stations.
    :type stations: set
    :param place: name of the city/place you are plotting.
    :type place: str
    :param save: save name for the plot.
    :type save: str
    :param width: width of the plot.
    :type width: int or float
    :param height: height opf the plot.
    :type height: int or float
    :param dpi: dpi of the plot.
    :type dpi: int
    :param plot_save_folder:
    :type plot_save_folder: str
    :return: None
    """
    nodes = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if (s_node, e_node) in trip_nbrs:
                nodes[s_node] += trip_nbrs[(s_node, e_node)]
                nodes[e_node] += trip_nbrs[(s_node, e_node)]

    max_n = max(nodes.values())
    n_rel = {key: value for key, value in nodes.items()}
    ns = [100 if n in stations else 0 for n in G.nodes()]
    plt.hist([value for key, value in n_rel.items() if value !=0],
             bins=ceil(max_n / 250))
    plt.show()
    for n in G.nodes():
        if n not in stations:
            n_rel[n] = max_n + 1
    cmap_name = 'cool'
    cmap = plt.cm.get_cmap(cmap_name)
    cmap = ['#999999'] + \
           [rgb2hex(cmap(n)) for n in reversed(np.linspace(1, 0, max_n,
                                                           endpoint=False))] \
           + ['#ffffff']
    color_n = [cmap[v] for k, v in n_rel.items()]

    fig, ax = ox.plot_graph(G, bgcolor='#ffffff', figsize=(width, height),
                            dpi=dpi, edge_linewidth=1.5, node_color=color_n,
                            node_size=ns, node_zorder=3, show=False,
                            close=False)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap_name),
                               norm=plt.Normalize(vmin=0, vmax=max_n))
    sm._A = []
    cbaxes = fig.add_axes([0.1, 0.075, 0.8, 0.03])
    cbar = fig.colorbar(sm, orientation='horizontal', cax=cbaxes,
                        ticks=[0, round(max_n / 2), max_n])
    cbar.ax.tick_params(axis='x', labelsize=18)
    cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])
    cbar.ax.set_xlabel('Usage of Stations', fontsize=24, labelpad=20)

    fig.suptitle('Nodes used as Stations in {}'.format(place.capitalize()),
                 fontsize=30, x=0.5, y=0.9, verticalalignment='bottom')
    plt.savefig('{}/{}_stations.png'.format(plot_save_folder, save),
                format='png')

    plt.close('all')
    plt.show()


def get_polygon_from_json(path_to_json):
    """
    Reads json at path. json can be created at http://geojson.io/.
    :param path_to_json: file path to json.
    :type path_to_json: str
    :return: Polygon given by json
    :rtype: Shapely polygon
    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    coordinates = data['features'][0]['geometry']['coordinates'][0]
    coordinates = [(item[0], item[1]) for item in coordinates]
    polygon = Polygon(coordinates)
    return polygon


def get_polygon_from_bbox(bbox):
    """
    Returns the Polygon resembled by the given bbox.
    :param bbox: bbox [north, south, east, west]
    :type bbox: list
    :return: Polygon of bbox
    :rtype: Shapely Polygon
    """
    north, south, east, west = bbox
    corners = [(east, north), (west, north), (west, south), (east, south)]
    polygon = Polygon(corners)
    return polygon


def get_bbox_from_polygon(polygon):
    x, y = polygon.exterior.coords.xy
    points = [(i, y[j]) for j, i in enumerate(x)]

    west, south = float('inf'), float('inf')
    east, north = float('-inf'), float('-inf')
    for x, y in points:
        west = min(west, x)
        south = min(south, y)
        east = max(east, x)
        north = max(north, y)

    return [north, south, east, west]


def drop_invalid_values(csv, column, values, save=False, save_path='',
                        delim=','):
    """
    Drops all rows if they have the given invalid value in the given column.
    :param csv: Path to csv.
    :type csv: str
    :param column: Column naem which should be checked vor invalid values.
    :type column: str
    :param values: List of the invalid values in the column.
    :type values: list
    :param save: Set true if df should be saved as csv after dropping.
    :type save: bool
    :param save_path: Path where it should e saved.
    :type save_path: str
    :param delim: Delimiter of the original csv.
    :type delim: str
    :return: DataFrame without the invalid values.
    :rtype: pandas DataFrame
    """
    df = read_csv(csv, delim)
    for value in values:
        drop_ind = df[df[column] == value].index
        df.drop(drop_ind, inplace=True)
    if save:
        write_csv(df, save_path)
    return df


def consolidate_nodes(G, tol):
    H = ox.project_graph(G, to_crs='epsg:2955')
    H = ox.consolidate_intersections(H, tolerance=tol, rebuild_graph=True,
                                     dead_ends=True, reconnect_edges=True)
    print('Consolidating intersections. Nodes before: {}. Nodes after: {}'
          .format(len(G.nodes), len(H.nodes)))
    H = nx.convert_node_labels_to_integers(H)
    nx.set_node_attributes(H, {n: n for n in H.nodes}, 'osmid')
    G = ox.project_graph(H, to_crs='epsg:4326')
    return G


def prepare_downloaded_map(G, trunk=False, consolidate=False, tol=35):
    """
    Prepares the downloaded map. Removes all motorway edges and if
    trunk=False also all trunk edges. Turns it to undirected, removes all
    isolated nodes using networkxs isolates() function and reduces the graph to
    the greatest connected component using osmnxs get_largest_component().
    :param G: Graph to clean.
    :type G: networkx graph
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :return: Cleaned graph
    :rtype: networkx graph.
    """
    # Remove self loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print('Removed {} self loops.'.format(len(self_loops)))

    # Remove motorways and trunks
    if trunk:
        s_t = ['motorway', 'motorway_link']
    else:
        s_t = ['motorway', 'motorway_link', 'trunk', 'trunk_link']
    edges_to_remove = [e for e in G.edges()
                       if get_street_type(G, e, multi=True) in s_t]
    G.remove_edges_from(edges_to_remove)
    print('Removed {} car only edges.'.format(len(edges_to_remove)))

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print('Removed {} isolated nodes.'.format(len(isolated_nodes)))
    G = ox.utils_graph.get_largest_component(G)
    print('Reduce to largest connected component')

    if consolidate:
        G = consolidate_nodes(G, tol)

    # Bike graph assumed undirected.
    G = G.to_undirected()
    print('Turned graph to undirected.')

    return G


def download_map_by_bbox(bbox, trunk=False, consolidate=False, tol=35):
    """
    Downloads a drive graph from osm given by the bbox and cleans it for usage.
    :param bbox: Boundary box of the map.
    :type bbox: list [north, south, east, west]
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :return: Cleaned graph.
    :rtype: networkx graph
    """
    print('Downloading map from bounding box. Northern bound: {}, '
          'southern bound: {}, eastern bound: {}, western bound: {}'
          .format(bbox[0], bbox[1], bbox[2], bbox[3]))
    G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],
                           network_type='drive')

    G = prepare_downloaded_map(G, trunk, consolidate=consolidate, tol=tol)

    return G


def download_map_by_name(city, nominatim_result=1, trunk=False,
                         consolidate=False, tol=35):
    """
    Downloads a drive graph from osm given by the name and geocode of the
    nominatim database and  cleans it for usage.
    :param city: Name of the place to donload.
    :type city: str
    :param nominatim_result: Which result of the nominatim database should
    be downloaded.
    :type nominatim_result: int
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :return: Cleaned graph.
    :rtype: networkx graph
    """
    print('Downloading map py place. Name of the place: {}, '
          'Nominatim result number {}.'.format(city, nominatim_result))
    G = ox.graph_from_place(city, which_result=nominatim_result,
                            network_type='drive')

    G = prepare_downloaded_map(G, trunk, consolidate=consolidate, tol=tol)

    return G


def download_map_by_polygon(polygon, trunk=False, consolidate=False, tol=35):
    """
    Downloads a drive graph from osm given by the polygon and cleans it for
    usage.
    :param polygon: Polygon of the graph.
    :type polygon: shapely Polygon
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :return: Cleaned graph.
    :rtype: networkx graph
    """
    print('Downloading map py polygon. Given polygon: {}'.format(polygon))
    G = ox.graph_from_polygon(polygon, network_type='drive')

    G = prepare_downloaded_map(G, trunk, consolidate=consolidate, tol=tol)

    return G


def save_map(G, save_path, save_name):
    """
    Saves graph to given path.
    :param G: Graph to save.
    :type G: networkx graph
    :param save_path: Path to save folder.
    :type save_path: str
    :param save_name: Name of the graphml file.
    :type save_name: str
    :return: none
    """
    ox.save_graphml(G, filepath=save_path+'{}.graphml'.format(save_name))


def data_to_matrix(stations, trips, scale='log'):
    df = pd.DataFrame(stations, columns=['station'])
    for station in stations:
        df[station] = [np.nan for x in range(len(stations))]
    df.set_index('station', inplace=True)
    if scale == 'log':
        for k, v in trips.items():
            if not k[0] == k[1]:
                df[k[0]][k[1]] = np.log(v)
    else:
        for k, v in trips.items():
            if not k[0] == k[1]:
                df[k[0]][k[1]] = v
    return df


def plot_matrix(city, df, plot_folder, save, cmap=None, figsize=None,
                dpi=300, plot_format='png', scale='log'):
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis')
    if figsize is None:
        figsize = [10, 10]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.pcolor(df, cmap=cmap)
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False,
                   labelleft=False)

    fig.suptitle('Trips in {}'.format(city), fontsize='x-large')
    plt.savefig(plot_folder+'{0:s}-matrix-{1:s}.{2:s}'
                .format(save, scale, plot_format), format=plot_format,
                 bbox_inches='tight')


def matrix_to_graph(df, rename_columns=None):
    if rename_columns is None:
        rename_columns = {'station': 'source', 'level_1': 'target', 0: 'trips'}
    df.values[[np.arange(len(df))] * 2] = np.nan
    df = df.stack().reset_index()
    df = df.rename(columns=rename_columns)
    return nx.from_pandas_edgelist(df=df, edge_attr='trips')


def plot_graph(city, G, plot_folder, node_cmap=None, edge_cmap=None, save=None,
               plot_format='png', scale='log'):
    if node_cmap is None:
        node_cmap = plt.cm.get_cmap('viridis')
    if edge_cmap is None:
        edge_cmap = plt.cm.get_cmap('PuRd')
    degree = [x[1] for x in nx.degree(G)]
    min_degree = min(degree)
    max_degree = max(degree)


    trips = [G[u][v]['trips'] for u, v in G.edges()]
    max_trips = max(trips)
    min_trips = min(trips)

    fig, ax = plt.subplots(figsize=[10, 10], dpi=150)
    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=False,
                     node_color=degree, cmap=node_cmap,
                     vmin=min_degree, vmax=max_degree, node_size=100,
                     edge_color=trips, edge_cmap=edge_cmap,
                     edge_vmin=min_trips, edge_vmax=max_trips)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('viridis'),
                               norm=plt.Normalize(vmin=min_degree,
                                                  vmax=max_degree))
    sm._A = []
    cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.8])

    cbar = fig.colorbar(sm, orientation='vertical', cax=cbaxes,
                        ticks=[min_degree, round(max_degree / 2), max_degree])
    cbar.ax.set_yticklabels([min_degree, round(max_degree / 2), max_degree])

    cbar.ax.tick_params(axis='x', labelsize=16)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_ylabel('Total Degree of a Station', fontsize=18)

    # fig.suptitle('Trips in {}'.format(city), fontsize='x-large')
    plt.savefig(plot_folder+'{0:s}-graph-{1:s}.{2:s}'
                .format(save, scale, plot_format), format=plot_format,
                bbox_inches='tight')


def sort_clustering(G):
    clustering = nx.clustering(G, weight='trips')
    clustering = {k: v for k, v in
                  sorted(clustering.items(), key=lambda item: item[1])}
    return list(reversed(clustering.keys()))


def get_communities(requests, requests_result, stations, G):
    gdf = ox.gdf_from_places(requests, which_results=requests_result)

    communities = [x.split(',')[0] for x in requests]
    df_com_stat = pd.DataFrame(communities, columns=['community'])
    df_com_stat['stations'] = [np.nan for x in range(len(communities))]
    df_com_stat['stations'] = df_com_stat['stations'].astype('object')
    df_com_stat.set_index('community', inplace=True)

    com_poly = {c: gdf['geometry'][idx] for idx, c in enumerate(communities)}

    com_stat = {k: [] for k in com_poly.keys()}
    stat_com = {k: None for k in stations}
    for station in stations:
        for com, poly in com_poly.items():
            long = G.nodes[station]['x']
            lat = G.nodes[station]['y']
            if poly.intersects(Point(long, lat)):
                com_stat[com].append(station)
                stat_com[station] = com

    for com, stat in com_stat.items():
        df_com_stat.at[com, 'stations'] = stat
    df_stat_com = pd.DataFrame.from_dict(stat_com, orient='index',
                                         columns=['community'])

    return df_com_stat, df_stat_com


def calc_average_trip_len(nxG, trip_nbrs, penalties=True):
    if penalties:
        bike_paths = []
    else:
        bike_paths = list(nxG.edges())

    nxG = nx.Graph(nxG.to_undirected())
    data = calc_current_state(nxG, trip_nbrs, bike_paths=bike_paths)
    trips_dict = data[7]

    length = []
    for trip, trip_info in trips_dict.items():
        length += [trip_info['length felt']] * trip_info['nbr of trips']
    return np.average(length)


def plot_average_trip_length(average_trip_len, colors, plot_folder,
                             plot_format='png', figsize=None, dpi=150):
    if figsize is None:
        figsize = [10, 10]

    cities = list(average_trip_len.keys())
    values = list(average_trip_len.values())

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    y_pos = np.arange(len(cities))
    for idx, city in enumerate(cities):
        color = to_rgba(colors[city])
        ax.barh(y_pos[idx], values[idx], color=color, align='center')
        x = values[idx] / 2
        y = y_pos[idx]
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.text(x, y, '{:3.2f}'.format(values[idx]), ha='center', va='center',
                color=text_color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cities)
    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('average trip length')
    ax.set_title('Average trip length in the city')

    plt.savefig(plot_folder + 'average-trip-len.{}'.format(plot_format),
                format=plot_format, bbox_inches='tight')
