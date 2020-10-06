from bikeability_optimisation.main.data import prep_city

nominatim_name = 'Hamburg, Deutschland'
nominatim_result = 2

city = 'Hamburg'
save = 'hh'

# [width, height]
plot_bbox_size = [20, 20]
plot_city_size = [20, 20]
plot_size = [20, 15]

input_csv = 'data/cleaned_data/{}_cleaned.csv'.format(save)
output = 'data/input_data/{}/'.format(save)
polygon = 'data/cropped_areas/{}.json'.format(save)
plots = 'plots/preparation/'

prep_city(city, save, input_csv, output, polygon, plots,
          nominatim_name=nominatim_name, nominatim_result=nominatim_result,
          trunk=False, consolidate=True, tol=35,
          by_bbox=False, plot_bbox_size=plot_bbox_size, by_city=False,
          plot_city_size=plot_city_size, by_polygon=True, plot_size=plot_size,
          cached_graph=False, cached_graph_folder=output,
          cached_graph_name=save)
