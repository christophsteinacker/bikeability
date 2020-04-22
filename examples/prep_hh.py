from bikeability_optimisation.main.data import prep_city

nominatim_name = 'Hamburg, Deutschland'
nominatim_result = 2

save = 'hh'
city = 'Hamburg'

# [width, height]
plot_bbox_size = [20, 20]
plot_city_size = [20, 20]
plot_size = [20, 20]

input_csv = 'data/cleaned_data/{}_cleaned.csv'.format(save)
input_algorithm = 'data/input_data/{}/'.format(save)
polygon = 'data/cropped_areas/{}.json'.format(save)
plots = 'plots/preparation/'

prep_city(city, save, nominatim_name, nominatim_result, input_csv,
          input_algorithm, polygon, plots, trunk=False,
          by_bbox=False, plot_bbox_size=plot_bbox_size,
          by_city=False, plot_city_size=plot_city_size,
          by_polygon=True, plot_size=plot_size)
