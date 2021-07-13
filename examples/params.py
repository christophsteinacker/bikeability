params = {}
params["penalties"] = {'primary': 7, 'secondary': 2.4, 'tertiary': 1.4,
                       'residential': 1.1}
params["street_cost"] = {'primary': 1, 'secondary': 1, 'tertiary': 1,
                         'residential': 1}

params["modes"] = [(False, 1)]  # Modes used for algorithm and plotting

params["cut"] = True  # If results should be normalised to first removal
params["bike_paths"] = None  # Additional ike Paths for current state
params["correct_area"] = True  # Correction of the area size
params["plot_evo"] = False  # If BP evolution should be plotted
params["evo_for"] = []  # For given modes

params["dpi"] = 150  # dpi of plots
params["titles"] = True  # If figure title should be plotted
params["legends"] = True  # If legends should be plotted
params["plot_format"] = 'png'  # Format of the saved plot

params["fs_title"] = 12  # Fontsize for title
params["fs_axl"] = 10  # Fontsize for axis labels
params["fs_ticks"] = 8  # Fontsize for axis tick numbers
params["fs_legend"] = 6  # Fontsize for legends

params["figs_ba_cost"] = (4, 3.5)  # Figsize ba-cost plot
params["c_ba"] = '#0080c0'  # Colour for ba in normal plot
params["lw_ba"] = 1
params["m_ba"] = 'D'  # Marker for ba
params["ms_ba"] = 5  # Marker size for ba
params["c_cost"] = '#4d4d4d'  # Colour for cost in normal plot
params["lw_cost"] = 1
params["m_cost"] = 's'
params["ms_cost"] = 5  # Marker size for cost

params["figs_los_nos"] = (4, 3.5)  # Figsize nos-los plot
params["c_los"] = '#e6194B'  # Colour for nos in normal plot
params["lw_los"] = 1
params["m_los"] = '8'
params["ms_los"] = 5  # Marker size for los
params["c_nos"] = '#911eb4'
params["lw_nos"] = 1
params["m_nos"] = 'v'
params["ms_nos"] = 5

params["c_st"] = {'primary': '#4d4d4d', 'secondary': '#666666',
                  'tertiary': '#808080', 'residential': '#999999',
                  'bike paths': '#0080c0'}
params["figs_comp_st"] = (0.75, 1.85)  # Figsize for comp_st_driven plot

params["figs_snetwork"] = (7, 5)
params["ns_snetwork"] = 4
params["nc_snetwork"] = '#333333'
params["ew_snetwork"] = 1
params["ec_snetwork"] = '#b3b3b3'

params["figs_station_usage"] = (1.5, 1.4)
params["figs_station_usage_hist"] = (3.65, 1.3)
params["stat_usage_norm"] = 1
params["cmap_nodes"] = 'cool'  # Cmap for nodes usage
params["nodesize"] = 4
params["ec_station_usage"] = '#b3b3b3'

params["figs_bp_evo"] = (2.2, 2.2)
params["lw_legend_bp_evo"] = 4

params["figs_bp_comp"] = (2.2, 2.2)
params["nc_pb_evo"] = '#d726ffff'
params["color_algo"] = '#007fbfff'  # 000075 0080c0
params["color_cs"] = '#40e640'  # 40e640
params["color_both"] = '#f58231'  # 40e6c0
params["color_unused"] = '#7f7f7fff'  # 808080

params["c_ed"] = '#0080c0'  # Colour ba for emp demand in rd-ed comp
params["c_rd"] = '#f58231'  # Colour ba for rand demand in rd-ed comp
params["lw_ed"] = 0.9
params["lw_rd"] = 0.9
params["c_rd_ed_area"] = '#999999'
params["a_rd_ed_area"] = 0.75
