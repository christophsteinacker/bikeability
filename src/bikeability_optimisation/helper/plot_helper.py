"""
This module includes all necessary functions for the plotting functionality.
"""
import pyproj
import shapely.ops as ops
from bikeability_optimisation.helper.algorithm_helper import *
from functools import partial
from copy import deepcopy


def calc_current_state(nxG, trip_nbrs, bike_paths=None):
    """
    Calculates the data for the current bike path situation. If no bike
    paths are provided all primary and secondary roads will be assigned with a
    bike path.
    :param nxG: Street graph to calculate in.
    :param trip_nbrs: Number of trips as used for the main algorithm.
    :param bike_paths: List of edges which hav a bike path.
    :return: Data structured as from the main algorithm.
    :rtype: np.array
    """
    if bike_paths is None:
        stypes = ['primary', 'secondary']
        bike_paths = [e for e in nxG.edges() if
                      get_street_type_cleaned(nxG, e, multi=False) in stypes]
    # All street types in network
    street_types = get_all_street_types_cleaned(nxG, multi=False)
    # Add bike paths
    len_on_type = {t: 0 for t in street_types}
    len_on_type['primary'] = 0
    len_on_type['bike path'] = 0

    # Set penalties for different street types
    penalties = {'primary': 7, 'secondary': 2.4, 'tertiary': 1.4,
                 'residential': 1.1}

    # Set cost for different street types
    street_cost = {'primary': 1, 'secondary': 1, 'tertiary': 1,
                   'residential': 1}

    trips_dict = {t_id: {'nbr of trips': nbr_of_trips, 'nodes': [],
                         'edges': [], 'length real': 0, 'length felt': 0,
                         'real length on types': len_on_type,
                         'felt length on types': len_on_type,
                         'on street': False}
                  for t_id, nbr_of_trips in trip_nbrs.items()}
    edge_dict = {edge: {'felt length': get_street_length(nxG, edge),
                        'real length': get_street_length(nxG, edge),
                        'street type': get_street_type_cleaned(nxG, edge),
                        'penalty': penalties[get_street_type_cleaned(nxG, edge)],
                        'speed limit': get_speed_limit(nxG, edge),
                        'bike path': True, 'load': 0, 'trips': []}
                 for edge in nxG.edges()}

    for edge, edge_info in edge_dict.items():
        if edge not in bike_paths:
            edge_info['bike path'] = False
            edge_info['felt length'] *= edge_info['penalty']
            nxG[edge[0]][edge[1]]['length'] *= edge_info['penalty']

    calc_trips(nxG, edge_dict, trips_dict, netwx=True)

    # Initialise lists
    total_cost = get_total_cost(bike_paths, edge_dict, street_cost)
    bike_path_perc = bike_path_percentage(edge_dict)
    total_real_distance_traveled = total_len_on_types(trips_dict, 'real')
    total_felt_distance_traveled = total_len_on_types(trips_dict, 'felt')
    nbr_on_street = nbr_of_trips_on_street(trips_dict)

    # Save data of this run to data array
    data = np.array([bike_paths, total_cost, bike_path_perc,
                     total_real_distance_traveled,
                     total_felt_distance_traveled, nbr_on_street])
    return data


def len_of_bikepath_by_type(ee, G, rev):
    """
    Calculates the length of bike paths along the different street types.
    :param ee: List of edited edges.
    :type ee: list
    :param G: Street graph.
    :type G: networkx graph
    :param rev: Reversed algorithm used True/False
    :type rev: bool
    :return: Dictionary keyed by street type
    :rtype: dict
    """
    street_types = ['primary', 'secondary', 'tertiary', 'residential']
    total_len = {k: 0 for k in street_types}
    for e in G.edges():
        st = get_street_type_cleaned(G, e, multi=False)
        total_len[st] += G[e[0]][e[1]]['length']
    len_fraction = {k: [0] for k in street_types}
    if not rev:
        ee = list(reversed(ee))
    for e in ee:
        st = get_street_type_cleaned(G, e, multi=False)
        len_before = len_fraction[st][-1]
        len_fraction[st].append(len_before + G[e[0]][e[1]]['length'] /
                                total_len[st])
        for s in [s for s in street_types if s != st]:
            len_fraction[s].append(len_fraction[s][-1])
    return len_fraction


def coord_transf(x, y, xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05):
    """
    Transfers the coordinates from data to relative coordiantes.
    :param x: x data coordinate
    :type x: float
    :param y: y data coordinate
    :type y: float
    :param xmin: min value of x axis
    :type xmin: float
    :param xmax: max value of x axis
    :type xmax: float
    :param ymin: min value of y axis
    :type ymin: float
    :param ymax: max value of y axis
    :type ymax: float
    :return: transformed x, y coordinates
    :rtype: float, float
    """
    return (x - xmin) / (xmax - xmin), (y - ymin) / (ymax - ymin)


def total_distance_traveled_list(total_dist, total_dist_now, rev):
    """
    Renormalises all total distance traveled lists.
    :param total_dist: dict of tdt lists
    :type total_dist: dict
    :param total_dist_now:  dict of tdt lists for the current state
    :type total_dist_now: dict
    :param rev: Reversed algorithm used True/False
    :type rev: bool
    :return:
    """
    if rev:
        s = 0
        e = -1
    else:
        s = -1
        e = 0
    dist = {}
    dist_now = {}

    # On all
    on_all = [i['total length on all'] for i in total_dist]
    dist_now['all'] = total_dist_now['total length on all'] / on_all[s]
    dist['all'] = [x / on_all[s] for x in on_all]
    # On streets w/o bike paths
    on_street = [i['total length on street'] for i in total_dist]
    dist_now['street'] = total_dist_now['total length on street'] / \
                         on_street[s]
    dist['street'] = [x / on_street[s] for x in on_street]
    # On primary
    on_primary = [i['total length on primary'] for i in total_dist]
    if on_primary[s] == 0:
        on_primary_norm = 1
    else:
        on_primary_norm = on_primary[s]
    dist_now['primary'] = total_dist_now['total length on primary'] / \
                          on_primary_norm
    dist['primary'] = [x / on_primary_norm for x in on_primary]
    # On secondary
    on_secondary = [i['total length on secondary'] for i in total_dist]
    dist_now['secondary'] = total_dist_now['total length on secondary'] / \
                            on_secondary[s]
    dist['secondary'] = [x / on_secondary[s] for x in on_secondary]
    # On tertiary
    on_tertiary = [i['total length on tertiary'] for i in total_dist]
    dist_now['tertiary'] = total_dist_now['total length on tertiary'] / \
                           on_tertiary[s]
    dist['tertiary'] = [x / on_tertiary[s] for x in on_tertiary]
    # On residential
    on_residential = [i['total length on residential'] for i in total_dist]
    dist_now['residential'] = total_dist_now['total length on residential'] / \
                              on_residential[s]
    dist['residential'] = [x / on_residential[s] for x in on_residential]
    # On bike paths
    on_bike = [i['total length on bike paths'] for i in total_dist]
    dist_now['bike paths'] = total_dist_now['total length on bike paths'] / \
                        on_bike[e]
    dist['bike paths'] = [x / on_bike[e] for x in on_bike]
    if not rev:
        for st, len_on_st in dist.items():
            dist[st] = list(reversed(len_on_st))
    return dist, dist_now


def sum_total_cost(cost, cost_now, rev):
    """
    Sums up all total cost up to each step.
    :param cost: List of costs per step
    :type cost: list
    :param cost_now: Cost of the current state.
    :type cost_now: float
    :param rev: Reversed algorithm used True/False
    :type rev: bool
    :return: Summed and renormalised cost and renormalised cost for the
    current state
    :rtype: list, float
    """
    if not rev:
        cost = list(reversed(cost))  # costs per step
    total_cost = [sum(cost[:i]) for i in range(1, len(cost) + 1)]
    cost_now = cost_now / total_cost[-1]
    total_cost = [i / total_cost[-1] for i in total_cost]
    return total_cost, cost_now


def get_end(tdt, tdt_now, rev):
    """

    :param tdt:
    :param tdt_now:
    :param rev:
    :return:
    """
    tdt, tdt_now = total_distance_traveled_list(tdt, tdt_now, rev)
    ba = [1 - (i - min(tdt['all'])) / (max(tdt['all']) - min(tdt['all']))
          for i in tdt['all']]
    return next(x for x, val in enumerate(ba) if val >= 1)


def get_street_type_ratio(G):
    G = G.to_undirected()
    G = nx.Graph(G)
    st_len = {'primary': 0, 'secondary': 0, 'tertiary': 0, 'residential': 0}
    total_len = 0
    for edge in G.edges:
        e_st = get_street_type_cleaned(G, edge, multi=False)
        e_len = get_street_length(G, edge, multi=False)
        st_len[e_st] += e_len
        total_len += e_len
    st_len_norm = {k: v / total_len for k, v in st_len.items()}
    return st_len_norm


def calc_polygon_area(polygon, unit='sqkm'):
    geom_area = ops.transform(
            partial(
                    pyproj.transform,
                    pyproj.Proj(init='EPSG:4326'),
                    pyproj.Proj(
                            proj='aea',
                            lat_1=polygon.bounds[1],
                            lat_2=polygon.bounds[3])),
            polygon)
    if unit == 'sqkm':
        return geom_area.area / 1000000
    if unit == 'sqm':
        return geom_area.area


def calc_scale(base_city, cities, saves, comp_folder, mode):
    blp = {}
    ba = {}

    for city in cities:
        save = saves[city]
        data = np.load(comp_folder+'ba_comp_{}.npy'.format(save),
                       allow_pickle=True)
        blp[city] = data[0][mode]
        ba[city] = data[1][mode]

    blp_base = blp[base_city]
    ba_base = ba[base_city]

    cities_comp = deepcopy(cities)
    cities_comp.remove(base_city)

    min_idx = {}
    for city in cities_comp:
        m_idx = []
        for idx, x in enumerate(ba[city]):
            m_idx.append(min(range(len(ba_base)),
                             key=lambda i: abs(ba_base[i] - x)))
        min_idx[city] = m_idx

    scale = {}
    for city in cities_comp:
        scale_city = []
        min_idx_city = min_idx[city]
        blp_city = blp[city]
        for idx, x in enumerate(min_idx_city):
            if blp_city[idx] != 0:
                scale_city.append(blp_base[x] / blp_city[idx])
            else:
                scale_city.append(np.nan)
        scale[city] = scale_city

    scale_mean = {}
    for city in cities:
        if city == base_city:
            scale_mean[city] = 1.0
        else:
            scale_mean[city] = np.mean(scale[city][1:])
    return scale_mean
