import h5py
from pathlib import Path
from pyproj import Proj, transform
from ..helper.data_helper import *
from .plot import plot_matrix, plot_station_degree, plot_used_area


def prep_city(city_name, save_name,  input_csv, output_folder, polygon_json,
              plot_folder, nominatim_name=None, nominatim_result=1,
              trunk=False, consolidate=False, tol=35, by_bbox=True,
              plot_bbox_size=None, by_city=True, plot_city_size=None,
              by_polygon=True, plot_size=None, cached_graph=False,
              cached_graph_folder=None, cached_graph_name=None):
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
            G_b = ox.load_graphml(filepath=cached_graph_folder +
                                           '{}_bbox.graphml'
                                  .format(cached_graph_name), node_type=int)
            if consolidate:
                G_b = consolidate_nodes(G_b, tol=tol)

        # Loading trips inside bbox
        print('Mapping stations and calculation trips on map given by bbox')
        trips_b, stations_b = load_trips(G_b, input_csv)

        # Colour all used nodes
        print('Plotting used nodes on graph given by bbox.')
        plot_used_nodes(G_b, trips_b, stations_b, city_name,
                        '{}_bbox'.format(save_name),
                        plot_save_folder=plot_folder,
                        width=plot_bbox_size[0], height=plot_bbox_size[1])
        fig, ax = ox.plot_graph(G_b, figsize=(20, 20), dpi=300, close=False,
                                show=False)
        fig.suptitle('Graph used for {}'.format(city_name.capitalize()),
                     fontsize=30)
        plt.savefig('{}/{}_bbox.png'.format(plot_folder, save_name),
                    format='png')
        ox.save_graphml(G_b, filepath=output_folder + '{}_bbox.graphml'
                        .format(save_name))
        np.save('{}/{}_bbox_demand.npy'.format(output_folder, save_name),
                [trips_b])
    if by_city:
        if not cached_graph:
            # Download whole map of the city
            print('Downloading complete map of city')
            G_c = download_map_by_name(nominatim_name, nominatim_result,
                                       trunk=trunk, consolidate=consolidate,
                                       tol=tol)
        else:
            print('Loading cached map of city')
            G_c = ox.load_graphml(filepath=cached_graph_folder +
                                           '{}_city.graphml'
                                  .format(cached_graph_name), node_type=int)
            if consolidate:
                G_c = consolidate_nodes(G_c, tol=tol)

        # Loading trips inside whole map
        print('Mapping stations and calculation trips on complete map.')
        trips_c, stations_c = load_trips(G_c, input_csv)

        # Colour all used nodes
        print('Plotting used nodes on complete city.')
        plot_used_nodes(G_c, trips_c, stations_c, city_name,
                        '{}_city'.format(save_name),
                        plot_save_folder=plot_folder,
                        width=plot_city_size[0], height=plot_city_size[1])
        fig, ax = ox.plot_graph(G_c, figsize=(20, 20), dpi=300, close=False,
                                show=False)
        fig.suptitle('Graph used for {}'.format(city_name.capitalize()),
                     fontsize=30)
        plt.savefig('{}/{}_city.png'.format(plot_folder, save_name),
                    format='png')
        ox.save_graphml(G_c, filepath=cached_graph_folder +
                                      '{}_city.graphml'.format(save_name))
        np.save('{}/{}_city_demand.npy'.format(output_folder, save_name),
                [trips_c])

    if by_polygon:
        # Download cropped map (polygon)
        polygon = get_polygon_from_json(polygon_json)

        if not cached_graph:
            print('Downloading polygon.')
            G = download_map_by_polygon(polygon, trunk=trunk,
                                        consolidate=consolidate, tol=tol)
        else:
            print('Loading cached map.')
            G = ox.load_graphml(filepath=cached_graph_folder + '{}.graphml'
                                .format(cached_graph_name), node_type=int)
            if consolidate:
                G = consolidate_nodes(G, tol=tol)

        # Loading trips inside the polygon
        print('Mapping stations and calculation trips in polygon.')
        trips, stations = load_trips(G, input_csv, polygon=polygon)

        # Colour all used nodes
        print('Plotting used nodes in polygon.')
        plot_used_nodes(G, trips, stations, city_name, save_name,
                        plot_save_folder=plot_folder,
                        width=plot_size[0], height=plot_size[1])
        fig, ax = ox.plot_graph(G, figsize=(20, 20), dpi=300, close=False,
                                show=False)
        fig.suptitle('Graph used for {}'.format(city_name.capitalize()),
                     fontsize=30)
        plt.savefig('{}/{}.png'.format(plot_folder, save_name), format='png')
        ox.save_graphml(G, filepath=output_folder+'{}.graphml'
                        .format(save_name))
        np.save('{}/{}_demand.npy'.format(output_folder, save_name), [trips])


def analyse_city(save, city, input_folder, output_folder, plot_folder,
                 cluster=False, bg_map=False, bg_polygon=None,
                 overlay=False, overlay_ploy=None, communities=False,
                 comm_requests=None, comm_requests_result=None,
                 plot_format='png', dpi=150):
    """
    Analyses the demand data of the city and saves the results as hdf5. If you
    are interested in the communities inside the demand data, provide the
    smallest possible administrative level for the city (e.g. districts or
    boroughs).
    :param save: Savename of the city.
    :type save: str
    :param city: Name of the city
    :type city: str
    :param input_folder: Folder of the input data for the algorithm
    :type input_folder: str
    :param output_folder: Folder for the output of the analysed data
    :type output_folder: str
    :param plot_folder: Folder for the plots.
    :type plot_folder: str
    :param cluster: If clustering coefficient of the stations should be
    calculated and used as sorting.
    :type cluster: bool
    :param bg_map: Background map for graph plots.
    :type bg_map: bool
    :param bg_polygon: Path to polygon json of the background
    :type bg_polygon: str
    :param overlay: Overlay
    :type overlay: bool
    :param overlay_ploy: Path to polygon json of the overlay
    :type overlay_ploy: str
    :param communities: Calculate communities
    :type communities: bool
    :param comm_requests: Nominatim requests for the areas of the city
    :type comm_requests: list
    :param comm_requests_result: Nominatim which_results
    :type comm_requests_result: list
    :param plot_format: File format for the plots
    :type plot_format: str
    :param dpi: DPI of the plots
    :type dpi: int
    :return: None
    """
    plt.rcdefaults()
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    hf_comp = h5py.File(output_folder+'{}_analysis.hdf5'.format(save), 'w')

    trips = np.load(input_folder + '{}_demand.npy'.format(save),
                    allow_pickle=True)[0]
    G_city = ox.load_graphml(filepath=input_folder+'{}.graphml'.format(save),
                             node_type=int)

    stations = []
    for k in trips.keys():
        stations.append(k[0])
        stations.append(k[1])
    stations = list(set(stations))

    if bg_map:
        polygon = get_polygon_from_json(bg_polygon)
        if overlay and overlay_ploy is not None:
            overlay_p = get_polygon_from_json(overlay_ploy)
        elif overlay and overlay_ploy is None:
            overlay_p = get_polygon_from_json(bg_polygon)
        else:
            overlay_p = None
        plot_used_area(G_city, polygon, stations, folder=plot_folder,
                       filename=save+'_used_area', plot_format=plot_format)
    else:
        polygon = None
        if overlay and overlay_ploy is not None:
            overlay_p = get_polygon_from_json(overlay_ploy)
        else:
            overlay_p = None


    stations_pos = {}
    for s in stations:
        lon = G_city.nodes[s]['x']
        lat = G_city.nodes[s]['y']
        inProj = Proj('epsg:4326')
        outProj = Proj('epsg:3857')
        stations_pos[s] = transform(inProj, outProj, lat, lon)

    df1 = data_to_matrix(stations, trips)
    print('Plotting OD Matrix.')
    plot_matrix(city, df1, plot_folder, save, cmap=None, figsize=None,
                dpi=dpi, plot_format=plot_format)

    print('Plotting data.')
    G, degree, indegree, outdegree, imbalance = matrix_to_graph(df1)

    plot_station_degree(G, degree=degree, indegree=indegree,
                        outdegree=outdegree,  node_cmap=None,
                        node_pos=stations_pos, bg_area=polygon,
                        overlay_poly=overlay_p, save=save,
                        plot_folder=plot_folder, plot_format=plot_format,
                        dpi=dpi, figsize=None)

    if cluster:
        print('Calculating clustering.')
        stations_new = sort_clustering(G)
        df2 = data_to_matrix(stations_new, trips)
        print('Plotting clustered OD Matrix.')
        plot_matrix(city, df2, plot_folder, save=save+'_cluster',
                    figsize=None, dpi=150, plot_format=plot_format)

        H = matrix_to_graph(df2, data=False)

        df3 = nx.to_pandas_edgelist(H)
        rename_columns = {'source': 'NODE_ID1', 'target': 'NODE_ID2',
                          'trips': 'WEIGHT'}
        df3.rename(columns=rename_columns, inplace=True)
        path = output_folder + '{}_trips.csv'.format(save)
        df3.to_csv(path, index=False)

    if communities:
        oxG = ox.load_graphml(filepath=input_folder+'{}.graphml'.format(save),
                              node_type=int)
        df_com_stat, df_stat_com = get_communities(comm_requests,
                                                   comm_requests_result,
                                                   stations, oxG)
        df_com_stat.to_csv(output_folder + '{}_com_stat.csv'.format(save),
                           index=True)
        df_stat_com.to_csv(output_folder + '{}_stat_com.csv'.format(save),
                           index=True)

    avg_trip_len = calc_average_trip_len(G_city, trips, penalties=True)

    hf_comp['station degree'] = degree
    hf_comp['station indegree'] = indegree
    hf_comp['station outdegree'] = outdegree
    hf_comp['trips'] = [v for k, v in trips.items()]
    hf_comp['imbalance'] = imbalance
    hf_comp['avg trip length'] = avg_trip_len
    hf_comp.close()
