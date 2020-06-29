"""
This module includes all necessary functions for the main algorithm.
"""
from bikeability_optimisation.helper.logger_helper import log_to_file
import networkit as nk
import networkx as nx
import numpy as np
import time


def get_street_type(G, edge, nk2nx=None, multi=False):
    """
    Returns the street type of the edge in G. If 'highway' in G is al list,
    return first entry.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: Street type.
    :rtype: str
    """
    if isinstance(nk2nx, dict):
        if edge in nk2nx:
            edge = nk2nx[edge]
        else:
            edge = nk2nx[(edge[1], edge[0])]
    if multi:
        street_type = G[edge[0]][edge[1]][0]['highway']
    else:
        street_type = G[edge[0]][edge[1]]['highway']
    if isinstance(street_type, str):
        return street_type
    else:
        return street_type[0]


def get_street_type_cleaned(G, edge, nk2nx=None, multi=False):
    """
    Returns the street type of the edge. Street types are reduced to
    primary, secondary, tertiary and residential.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: Street type·
    :rtype: str
    """
    st = get_street_type(G, edge, nk2nx, multi=multi)
    if st in ['primary', 'primary_link', 'trunk', 'trunk_link']:
        return 'primary'
    elif st in ['secondary', 'secondary_link']:
        return 'secondary'
    elif st in ['tertiary', 'tertiary_link', 'road']:
        return 'tertiary'
    else:
        return 'residential'


def get_all_street_types(G, multi=False):
    """
    Returns all street types appearing in G.
    :param G: Graph.
    :type G: networkx graph.
    :param multi: Set True if G is a MultiGraph
    :type multi: bool
    :return: List of all street types.
    :rtype: list of str
    """
    street_types = set()
    for edge in G.edges():
        street_types.add(get_street_type(G, edge, multi=multi))
    return list(street_types)


def get_all_street_types_cleaned(G, multi=False):
    """
    Returns all street types appearing in G. Street types are reduced to
    primary, secondary, tertiary and residential.
    :param G: Graph.
    :type G: networkx graph.
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: List of all street types.
    :rtype: list of str
    """
    street_types_cleaned = set()
    for edge in G.edges():
        street_types_cleaned.add(get_street_type(G, edge, multi=multi))
    return list(street_types_cleaned)


def get_speed_limit(G, edge, nk2nx=None, multi=False):
    """
    Returns speed limit of the edge in G.
    :param G: Graph.
    :type G: networkx graph
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: Speed limit.
    :rtype: float
    """
    if isinstance(nk2nx, dict):
        if edge in nk2nx:
            edge = nk2nx[edge]
        else:
            edge = nk2nx[(edge[1], edge[0])]
    speed_limit = 50
    if multi:
        if 'maxspeed' in G[edge[0]][edge[1]]:
            speed_limit = G[edge[0]][edge[1]][0]['maxspeed']
    else:
        if 'maxspeed' in G[edge[0]][edge[1]]:
            speed_limit = G[edge[0]][edge[1]]['maxspeed']
    if isinstance(speed_limit, float):
        return speed_limit
    elif isinstance(speed_limit, int):
        return speed_limit
    elif isinstance(speed_limit, list):
        return speed_limit[0]
    else:
        return 50


def get_street_length(G, edge, nk2nx=None, multi=False):
    """
    Returns the length of the edge in G.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: Length of edge.
    :rtype: float
    """
    if isinstance(nk2nx, dict):
        if edge in nk2nx:
            edge = nk2nx[edge]
        else:
            edge = nk2nx[(edge[1], edge[0])]
    if multi:
        length = G[edge[0]][edge[1]][0]['length']
    else:
        length = G[edge[0]][edge[1]]['length']
    return length


def get_cost(edge, edge_dict, cost_dict):
    """
    Returns the cost of an edge depending on its street type.
    :param edge: Edge.
    :type edge: tuple of integers
    :param edge_dict: Dictionary with all edge information.
    :type edge_dict: dict of dicts
    :param cost_dict: Dictionary with cost of edge depending on street type.
    :type cost_dict: dict
    :return: Cost of the edge
    :rtype: float
    """
    street_type = edge_dict[edge]['street type']
    street_length = edge_dict[edge]['real length']
    return street_length * cost_dict[street_type]


def get_total_cost(bikepaths, edge_dict, cost_dict):
    """
    Returns the cost of building bike paths.
    :param bikepaths: Edges with bike paths.
    :type bikepaths: list of tuple of int
    :param edge_dict: Dictionary with all edge information.
    :type edge_dict: dict of dicts
    :param cost_dict: Dictionary with cost of edge depending on street type.
    :type cost_dict: dict
    :return: Cost of the edge
    :rtype: float
    """
    total_cost = 0
    for edge in bikepaths:
        total_cost += get_cost(edge, edge_dict, cost_dict)
    return total_cost


def get_trip_edges(edges_dict, trip_nodes):
    """
    Returns the edge sequence of a trip given by its node sequence.
    :param edges_dict: Dictionary with all information about the edges.
    :type edges_dict: Dict of dicts.
    :param trip_nodes: Node sequence of a trip.
    :type trip_nodes: list of integers
    :return: Edge sequence.
    :rtype: list of tuples of integers
    """
    edge_sequence = []
    for i in range(len(trip_nodes) - 1):
        f_n = trip_nodes[i]         # First node
        s_n = trip_nodes[i + 1]     # Second node of the edge
        # Dict doesn't accept (2, 1) for undirected edge (1, 2):
        if (f_n, s_n) in edges_dict:
            edge_sequence.append((f_n, s_n))
        else:
            edge_sequence.append((s_n, f_n))
    return edge_sequence


def get_minimal_loaded_edge(edge_dict, trips_dict, minmode=0, rev=False):
    """
    Returns the minimal loaded edge in edge_list.
    If unedited=True it returns the minimal loaded unedited edge.
    If there are multiple edges with the same minimal load, one is randomly
    drawn.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts.
    :param trips_dict: Dictionary with al information about the trips.
    :type trips_dict: dict of dicts
    :param minmode: Mode in which way the min loaded edge should be searched.
                    0 = min(load), 1 = min(load*penalty)
    :type minmode: int
    :param rev: If True returns maximal loaded edge
    :type rev: bool
    :return: minimal loaded edge
    :rtype: Tuple of integers
    """
    if rev:
        edges = {edge: edge_info for edge, edge_info in edge_dict.items()
                 if not edge_info['bike path']}
    else:
        edges = {edge: edge_info for edge, edge_info in edge_dict.items()
                 if edge_info['bike path']}

    # Different picking rules
    if minmode == 0:
        # unweighted load
        edges_load = {edge: edge_info['load'] for edge, edge_info
                      in edges.items()}
    elif minmode == 1:
        # load weighted by penalty
        if rev:
            edges_load = {edge: edge_info['load'] * (1 / edge_info['penalty'])
                          for edge, edge_info in edges.items()}
        else:
            edges_load = {edge: edge_info['load'] * edge_info['penalty']
                          for edge, edge_info in edges.items()}
    elif minmode == 2:
        # load weighted by the average trip length
        edges_trip_length = {}
        for edge, edge_info in edges.items():
            length = []
            for trip in edge_info['trips']:
                length += [trips_dict[trip]['length felt']] * \
                          trips_dict[trip]['nbr of trips']
            edges_trip_length[edge] = np.nan_to_num(np.average(length))
        edges_load = {edge: edge_info['load'] * edges_trip_length[edge]
                      for edge, edge_info in edges.items()}
    else:
        print('Minmode has to be chosen. Aborting.')
        edges_load = {}

    if edges_load == {}:
        return 'We are done!'
    else:
        if rev:
            max_load = max(edges_load.values())
            max_edges = [e for e, load in edges_load.items()
                         if load == max_load]
            max_edge = max_edges[np.random.choice(len(max_edges))]
            return max_edge
        else:
            min_load = min(edges_load.values())
            min_edges = [e for e, load in edges_load.items()
                         if load == min_load]
            min_edge = min_edges[np.random.choice(len(min_edges))]
            return min_edge


def bike_path_percentage(edge_dict):
    """
    Returns the bike path percentage by length.
    :param edge_dict: Dictionary with all information about the edges
    :type edge_dict: dict of dicts
    :return: percentage of bike paths by length.
    :rtype float
    """
    bike_length = 0
    total_length = 0
    for edge, edge_info in edge_dict.items():
        total_length += edge_info['real length']
        if edge_info['bike path']:
            bike_length += edge_info['real length']
    return bike_length / total_length


def check_if_trip_on_street(trip_info, edge_dict):
    """
    Checks if given trip is somewhere on a street.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: True if on street false if not.
    :rtype: bool
    """
    for edge in trip_info['edges']:
        if not edge_dict[edge]['bike path']:
            return True
    return False


def nbr_of_trips_on_street(trips_dict):
    """
    Returns the number of trips that are somewhere on a street.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: Number of trips at least once on a street.
    :rtype: integer
    """
    nbr_on_street = 0
    for trip, trip_info in trips_dict.items():
        if trip_info['on street']:
            nbr_on_street += trip_info['nbr of trips']
    return nbr_on_street


def set_trips_on_street(trips_dict, edge_dict):
    """
    Sets "on street" value in trips_dict to the right value.
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips_dict.
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info['on street'] = False
        for edge in trip_info['edges']:
            if not edge_dict[edge]['bike path']:
                trip_info['on street'] = True
    return trips_dict


def get_len_of_trips_over_edge(edge, edge_list, trips_dict):
    """
    Returns the total traveled distance over the given edge.
    ttd = edge length * nbr of trips over edge
    :param edge: Edge.
    :type edge: tuple of integers
    :param edge_list: Dictionary with all information about the edges.
    :type edge_list: dict of dicts
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :return: Total traveled distance.
    :rtype float
    """
    length = 0
    for trip in edge_list[edge]['trips']:
        length += trips_dict[trip]['nbr of trips'] * \
                  trips_dict[trip]['length real']
    return length


def real_trip_length(trip_info, edge_dict):
    """
    Returns the real length og a trip.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict of dicts.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Real length of the trip.
    :rtype: float
    """
    length = sum([edge_dict[edge]['real length']
                  for edge in trip_info['edges']])
    return length


def felt_trip_length(trip_info, edge_dict):
    """
    Returns the felt length og a trip.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict of dicts.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Felt length of the trip.
    :rtype: float
    """
    length = sum([edge_dict[edge]['felt length']
                  for edge in trip_info['edges']])
    return length


def len_on_types(trip_info, edge_dict, len_type='real'):
    """
    Returns a dict with the length of the trip on the different street types.
    len_type defines if felt or real length.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param len_type: 'real' or 'felt' length is used.
    :type len_type: str
    :return: Dictionary with length on different street types.
    :rtype: dict
    """
    len_on_type = {t: 0 for t, l in
                   trip_info[len_type+' length on types'].items()}
    for edge in trip_info['edges']:
        street_type = edge_dict[edge]['street type']
        street_length = edge_dict[edge][len_type+' length']
        if edge_dict[edge]['bike path']:
            len_on_type['bike path'] += street_length
        else:
            len_on_type[street_type] += street_length
    return len_on_type


def total_len_on_types(trips_dict, len_type):
    """
    Returns the total distance driven sorted by street type.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param len_type: 'real' or 'felt' length is used.
    :type len_type: str
    :return: Dictionary with total length on different street types.
    :rtype: dict
    """
    lop = 0
    los = 0
    lot = 0
    lor = 0
    lob = 0
    for trip, trip_info in trips_dict.items():
        nbr_of_trips = trip_info['nbr of trips']
        lop += nbr_of_trips * \
               trip_info[len_type+' length on types']['primary']
        los += nbr_of_trips * \
               trip_info[len_type+' length on types']['secondary']
        lot += nbr_of_trips * \
               trip_info[len_type+' length on types']['tertiary']
        lor += nbr_of_trips * \
               trip_info[len_type+' length on types']['residential']
        lob += nbr_of_trips * \
               trip_info[len_type+' length on types']['bike path']
    tlos = lop + los + lot + lor
    tloa = tlos + lob
    return {'total length on all': tloa, 'total length on street': tlos,
            'total length on primary': lop, 'total length on secondary': los,
            'total length on tertiary': lot,
            'total length on residential': lor,
            'total length on bike paths': lob}


def set_len(trips_dict, edge_dict):
    """
    Sets the length of a trip to the correct value in the trips dictionary.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param edge_dict:  Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips dictionary
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info['length real'] = real_trip_length(trip_info, edge_dict)
        trip_info['length felt'] = felt_trip_length(trip_info, edge_dict)
    return trips_dict


def set_len_on_types(trips_dict, edge_dict):
    """
    Sets the length by type of a trip to the correct value in the trips
    dictionary.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param edge_dict:  Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips dictionary
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info['real length on types'] = len_on_types(trip_info, edge_dict,
                                                         'real')
        trip_info['felt length on types'] = len_on_types(trip_info, edge_dict,
                                                         'felt')
    return trips_dict


def get_connected_bike_components(G):
    """
    Returns the greatest connected bike component of graph G with given edited
    edges.
    :param G: Edited graph
    :type G: networkit graph
    :return: Greatest connected bike component.
    """
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    nbr_of_components = cc.numberOfComponents()
    cc_size = cc.getComponentSizes()
    return nbr_of_components, cc_size


def add_load(edge_dict, trips_dict):
    """
    Adds load and trip_id of the given trips to the edges.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: edge_dict with updated information
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        for e in trip_info['edges']:
            edge_dict[e]['trips'] += [trip]
            edge_dict[e]['load'] += trip_info['nbr of trips']
    return edge_dict


def delete_load(edge_dict, trips_dict):
    """
    Deletes load and trip_id of the given trips from the edges.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: edge_dict with updated information
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        for e in trip_info['edges']:
            edge_dict[e]['trips'].remove(trip)
            edge_dict[e]['load'] -= trip_info['nbr of trips']
    return edge_dict


def get_all_shortest_paths(G, source):
    """
    Returns all shortest paths with sources definde in sources.
    :param G: Graph.
    :type G: networkit graph
    :param source: Source node to calculate the shortest paths.
    :type source: int
    :return: Dict with all shortest paths, keyed by target.
    :rtype: dict
    """
    d = nk.distance.Dijkstra(G, source, storePaths=True)
    d.run()
    shortest_paths = {tgt: d.getPath(tgt) for tgt in list(G.iterNodes())}
    return shortest_paths


def remove_isolated_nodes(nkG):
    """
    Removes all isolated nodes in the give graph.
    :param nkG: Graph.
    :type nkG: networkit graph.
    :return: None
    """
    isolated_nodes = [n for n in nkG.iterNodes() if nkG.isIsolated(n)]
    for n in isolated_nodes:
        nkG.removeNode(n)


def get_nx_edge(nk_edge, nk2nx_edges):
    """
    Returns the networkx edge for the given networkit edge.
    :param nk_edge: Edge in the nk graph.
    :type nk_edge: tuple
    :param nk2nx_edges: Dict mapping nk edges to nx edges.
    :type nk2nx_edges: dict
    :return: Edge in the nx graph.
    :rtype: tuple
    """
    if nk_edge in nk2nx_edges:
        return nk2nx_edges[nk_edge]
    else:
        return nk2nx_edges[(nk_edge[1], nk_edge[0])]


def set_sp_info(source, shortest_paths, edge_dict, trips_dict):
    for trip, trip_info in trips_dict.items():
        if trip[0] == source:
            if not trip_info['nodes'] == shortest_paths[trip[1]]:
                delete_load(edge_dict, {trip: trip_info})
                trip_info['nodes'] = shortest_paths[trip[1]]
                trip_info['edges'] = get_trip_edges(edge_dict,
                                                    trip_info['nodes'])
                for e in trip_info['edges']:
                    edge_dict[e]['trips'] += [trip]
                    edge_dict[e]['load'] += trip_info['nbr of trips']
            trip_info['length felt'] = felt_trip_length(trip_info, edge_dict)
            trip_info['length real'] = real_trip_length(trip_info, edge_dict)
            trip_info['real length on types'] = len_on_types(trip_info,
                                                             edge_dict, 'real')
            trip_info['felt length on types'] = len_on_types(trip_info,
                                                             edge_dict, 'felt')
            trip_info['on street'] = check_if_trip_on_street(trip_info,
                                                             edge_dict)


def calc_trips(G, edge_dict, trips_dict, netwx=False):
    """
    Calculates the shortest paths for all trips and sets all corresponding
    information in the trip_dict
    :param G: graph to calculate the s.p. in.
    :type G: networkit or networkx graph
    :param edge_dict: Dictionary with all information about the edges of G.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :param netwx: If G is a networkx graph set netwx to True.
    :type netwx: bool
    :return: Updated trips_dict and edge_dict.
    :rtype: dict of dicts
    """
    # Calculate single source paths for all origin nodes
    origin_nodes = list({k[0] for k, v in trips_dict.items()})
    if not netwx:
        for source in origin_nodes:
            shortest_paths = get_all_shortest_paths(G, source)
            # Set all information to trip_info and edge_info
            set_sp_info(source, shortest_paths, edge_dict, trips_dict)
    else:
        for source in origin_nodes:
            shortest_paths = nx.single_source_dijkstra_path(G, source,
                                                            weight='length')
            # Set all information to trip_info and edge_info
            set_sp_info(source, shortest_paths, edge_dict, trips_dict)
    return trips_dict, edge_dict


def edit_edge(nkG, edge_dict, edge):
    """
    Edits "felt length" of given edge  in the edge_dict and "length" in G.
    Length change is done corresponding to the street type of the edge.
    :param nkG: Graph.
    :type nkG: networkit graph
    :param edge_dict: Dictionary with all information about the edges of G.
    :type edge_dict: dict of dicts
    :param edge: Edge to edit.
    :type edge: tuple of integers
    :return: Updated G and edge_dict.
    :rtype: networkx graph and  dict of dicts
    """
    edge_dict[edge]['bike path'] = not edge_dict[edge]['bike path']
    edge_dict[edge]['felt length'] *= edge_dict[edge]['penalty']
    nkG.setWeight(edge[0], edge[1], edge_dict[edge]['felt length'])
    return nkG, edge_dict


def save_data(path, data, logfile, message):
    """
    Saves data to disk and logs it.
    :param path: Path to save the data
    :type path: str
    :param data: Data to save
    :type data: anything that can be handled by numpy
    :param logfile: Path to logfile
    :type logfile: str
    :param message: Message to log.
    :type message: str
    :return: None
    """
    log_to_file(logfile, message, stamptime=time.localtime(), stamp=True,
                difference=False)
    np.save(path, data)
