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