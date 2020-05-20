from bikeability_optimisation.helper.data_helper import *
from pathlib import Path


def prep_city(city_name, save_name, nominatim_name, nominatim_result,
              input_csv, output_folder, polygon_json, plot_folder,
              trunk=False, by_bbox=True, plot_bbox_size=None,
              by_city=True, plot_city_size=None,
              by_polygon=True, plot_size=None):
    if plot_size is None:
        plot_size = [20, 20]
    if plot_city_size is None:
        plot_city_size = [20, 20]
    if plot_bbox_size is None:
        plot_bbox_size = [20, 20]

    # Check if necessary folders exists, otherwise create.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    if by_bbox:
        # Get bounding box of trips
        print('Getting bbox of trips.')
        bbox = get_bbox_of_trips(input_csv)

        # Download map given by bbox
        print('Downloading map given by bbox.')
        G_b = download_map_by_bbox(bbox, trunk=trunk)

        # Loading trips inside bbox
        print('Mapping stations and calculation trips on map given by bbox')
        trips_b, stations_b = load_trips(G_b, input_csv)

        # Colour all used nodes
        print('Plotting used nodes on graph given by bbox.')
        plot_used_nodes(G_b, trips_b, stations_b, city_name,
                        '{}_bbox'.format(save_name),
                        plot_save_folder=plot_folder,
                        width=plot_bbox_size[0], height=plot_bbox_size[1])
        ox.save_graphml(G_b, filename='{}_bbox.graphml'.format(save_name),
                        folder=output_folder)
        np.save('{}/{}_bbox_demand.npy'.format(output_folder, save_name),
                [trips_b])
    if by_city:
        # Download whole map of the city
        print('Downloading complete map of city')
        G_c = download_map_by_name(nominatim_name, nominatim_result,
                                   trunk=trunk)

        # Loading trips inside whole map
        print('Mapping stations and calculation trips on complete map.')
        trips_c, stations_c = load_trips(G_c, input_csv)

        # Colour all used nodes
        print('Plotting used nodes on complete city.')
        plot_used_nodes(G_c, trips_c, stations_c, city_name,
                        '{}_city'.format(save_name),
                        plot_save_folder=plot_folder,
                        width=plot_city_size[0], height=plot_city_size[1])
        ox.save_graphml(G_c, filename='{}_city.graphml'.format(save_name),
                        folder=output_folder)
        np.save('{}/{}_city_demand.npy'.format(output_folder, save_name),
                [trips_c])

    if by_polygon:
        # Download cropped map (polygon)
        polygon = get_polygon_from_json(polygon_json)

        print('Downloading polygon.')
        G = download_map_by_polygon(polygon, trunk=trunk)

        # Loading trips inside the polygon
        print('Mapping stations and calculation trips in polygon.')
        trips, stations = load_trips(G, input_csv, polygon=polygon)

        # Colour all used nodes
        print('Plotting used nodes in polygon.')
        plot_used_nodes(G, trips, stations, city_name, save_name,
                        plot_save_folder=plot_folder,
                        width=plot_size[0], height=plot_size[1])
        ox.save_graphml(G, filename='{}.graphml'.format(save_name),
                        folder=output_folder)
        np.save('{}/{}_demand.npy'.format(output_folder, save_name), [trips])


def analyse_city(save, city, input_folder, output_folder, plot_folder,
                 communities=False, requests=None, requests_result=None,
                 scale='log'):

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    trips = np.load(input_folder + '{}_demand.npy'.format(save),
                    allow_pickle=True)[0]

    stations = []
    for k in trips.keys():
        stations.append(k[0])
        stations.append(k[1])
    stations = list(set(stations))

    df1 = data_to_matrix(stations, trips, scale=scale)
    plot_matrix(city, df1, plot_folder, save, cmap=None, figsize=None,
                dpi=300, plot_format='png', scale=scale)

    G = matrix_to_graph(df1)
    plot_graph(city, G, node_cmap=None, edge_cmap=None, save=save,
               plot_folder=plot_folder, plot_format='png', scale=scale)

    stations_new = sort_clustering(G)
    df2 = data_to_matrix(stations_new, trips, scale=scale)
    plot_matrix(city, df2, plot_folder, save=save+'-cluster', cmap=None,
                figsize=None, dpi=300, plot_format='png', scale=scale)

    H = matrix_to_graph(df2)
    plot_graph(city, G, node_cmap=None, edge_cmap=None, save=save+'-cluster',
               plot_folder=plot_folder, plot_format='png', scale=scale)

    df3 = nx.to_pandas_edgelist(H)
    rename_columns = {'source': 'NODE_ID1', 'target': 'NODE_ID2',
                      'trips': 'WEIGHT'}
    df3.rename(columns=rename_columns, inplace=True)
    path = output_folder + '{}_trips.csv'.format(save)
    df3.to_csv(path, index=False)

    if communities:
        oxG = ox.load_graphml(filename='{}.graphml'.format(save),
                              folder=input_folder, node_type=int)
        df_com_stat, df_stat_com = get_communities(requests, requests_result,
                                                   stations, oxG)
        df_com_stat.to_csv(output_folder + '{}_com_stat.csv'.format(save),
                           index=True)
        df_stat_com.to_csv(output_folder + '{}_stat_com.csv'.format(save),
                           index=True)
