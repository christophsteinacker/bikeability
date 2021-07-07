"""
This module includes all necessary functions for the plotting functionality.
"""
import json
import geopandas
import contextily as ctx
import osmnx as ox
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from pathlib import Path
from pyproj import Proj, transform
from scipy import stats
from collections import Counter
from .data import get_polygon_from_json, get_polygons_from_json, \
    get_bbox_from_polygon, get_polygon_from_bbox
from .algorithm import calc_current_state
from ..helper.plot_helper import *


def plot_nx_graph(G, node_pos=None, node_size=100, node_color='b',
                  node_cmap=None, node_cmin=None, node_cmax=None,
                  node_cmap_label='',
                  edge_width=1.0, edge_color='k', edge_cmap=None,
                  edge_cmin=None, edge_cmax=None, edge_cmap_label='',
                  bg_area=None, overlay_poly=None, scalebar=False,
                  figsize=None, dpi=150, folder='', filename='',
                  plot_format='png'):
    """

    :param G:
    :param node_pos:
    :param node_size:
    :param node_color:
    :param node_cmap:
    :param node_cmin:
    :param node_cmax:
    :param node_cmap_label:
    :param edge_width:
    :param edge_color:
    :param edge_cmap:
    :param edge_cmin:
    :param edge_cmax:
    :param edge_cmap_label:
    :param bg_area:
    :param overlay_poly:
    :param scalebar:
    :param figsize:
    :param dpi:
    :param folder:
    :param filename:
    :param plot_format:
    :return:
    """
    if figsize is None:
        figsize = [10, 10]
    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    if scalebar:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection=ccrs.epsg(3857))
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if bg_area is not None:
        bg_bbox = get_polygon_from_bbox(get_bbox_from_polygon(bg_area))
        bg = geopandas.GeoDataFrame(crs='epsg:4326', geometry=[bg_bbox])
        bg = bg.to_crs('epsg:3857')
        bg.plot(alpha=0, ax=ax)
        ctx.add_basemap(ax)  # , url=ctx.providers.Stamen.TonerLite
        if scalebar:
            scale_bar(ax, (0.05, 0.05), 5)
    if overlay_poly is not None:
        overlay = geopandas.GeoDataFrame(crs='epsg:4326',
                                         geometry=[overlay_poly])
        overlay = overlay.to_crs('epsg:3857')
        overlay.plot(ax=ax, lw=5, alpha=0.2)

    nx.draw_networkx(G, pos=node_pos, ax=ax, with_labels=False,
                     node_size=node_size, node_color=node_color,
                     cmap=node_cmap, vmin=node_cmin, vmax=node_cmax,
                     width=edge_width, edge_color=edge_color,
                     edge_cmap=edge_cmap, edge_vmin=edge_cmin,
                     edge_vmax=edge_cmax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    if (node_cmap is not None) and (edge_cmap is None):
        sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=Normalize(
                vmin=node_cmin, vmax=node_cmax))
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes('bottom', size="5%", pad=0.05)
        cbar = fig.colorbar(sm, orientation='horizontal', cax=cbaxes)
        cbar.ax.tick_params(axis='x', labelsize=16)
        cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        cbar.ax.set_xlabel(node_cmap_label, fontsize=18)
    if (node_cmap is None) and (edge_cmap is not None):
        sm = plt.cm.ScalarMappable(cmap=edge_cmap,
                                   norm=Normalize(vmin=edge_cmin,
                                                  vmax=edge_cmax))
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes('bottom', size="5%", pad=0.05)
        cbar = fig.colorbar(sm, orientation='horizontal', cax=cbaxes)
        cbar.ax.tick_params(axis='x', labelsize=16)
        cbar.ax.set_xlabel(edge_cmap_label, fontsize=18)
    if (node_cmap is not None) and (edge_cmap is not None):
        sm_1 = plt.cm.ScalarMappable(cmap=node_cmap,
                                     norm=Normalize(vmin=node_cmin,
                                                    vmax=node_cmax))
        divider_1 = make_axes_locatable(ax)
        cbaxes_1 = divider_1.append_axes('left', size="5%", pad=0.05)
        cbar_1 = fig.colorbar(sm_1, orientation='vertical', cax=cbaxes_1)
        cbar_1.ax.tick_params(axis='y', labelsize=16)
        cbar_1.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        cbar_1.ax.set_ylabel(node_cmap_label, fontsize=18)
        sm_2 = plt.cm.ScalarMappable(cmap=edge_cmap,
                                     norm=Normalize(vmin=edge_cmin,
                                                    vmax=edge_cmax))
        divider_2 = make_axes_locatable(ax)
        cbaxes_2 = divider_2.append_axes('right', size="5%", pad=0.05)
        cbar_2 = fig.colorbar(sm_2, orientation='vertical', cax=cbaxes_2)
        cbar_2.ax.tick_params(axis='y', labelsize=16)
        cbar_2.ax.yaxis.set_ticks_position('right')
        cbar_2.ax.yaxis.set_label_position('right')
        cbar_2.ax.set_ylabel(edge_cmap_label, fontsize=18)

    fig.savefig(f'{folder}{filename}.{plot_format}', format=plot_format,
                bbox_inches='tight')


# noinspection SpellCheckingInspection
def plot_station_degree(G, degree=None, indegree=None, outdegree=None,
                        node_cmap=None, node_pos=None, bg_area=None,
                        overlay_poly=None, save='', plot_folder='',
                        plot_format='png', dpi=150, figsize=None):
    if node_cmap is None:
        node_cmap = plt.cm.get_cmap('plasma_r')
    if node_pos is None:
        node_pos = nx.kamada_kawai_layout(G)
    if figsize is None:
        figsize = [10, 10]
    if degree is None:
        degree = [d for n, d in nx.degree(G)]

    # degree_log = [np.log10(x) for x in degree]
    min_degree = min(degree)
    # min_degree_log = np.log10(min_degree)
    max_degree = max(degree)
    # max_degree_log = np.log10(max_degree)

    plot_nx_graph(G, node_pos=node_pos, node_size=100, node_color=degree,
                  node_cmap=node_cmap, node_cmin=min_degree,
                  node_cmax=max_degree, node_cmap_label='station degree',
                  edge_width=0, bg_area=bg_area, overlay_poly=overlay_poly,
                  figsize=figsize, dpi=dpi, folder=plot_folder,
                  filename=save + '_degree_graph', plot_format=plot_format)

    if min(degree) <= 50:
        hist_xlim = (0, max(degree) + 1)
    else:
        hist_xlim = (min(degree) - 10, max(degree) + 1)
    bins = round((hist_xlim[1] - hist_xlim[0]) / 2)
    plot_histogram(degree, f'{plot_folder}{save}_degree_hist',
                   bins=bins, xlim=hist_xlim, xlabel='degree',
                   ylabel='# of stations', plot_format=plot_format, dpi=dpi)
    if indegree is not None:
        plot_histogram(indegree, f'{plot_folder}{save}_indegree_hist',
                       bins=bins, xlim=hist_xlim, xlabel='indegree',
                       ylabel='# of stations', plot_format=plot_format,
                       dpi=dpi)
    if outdegree is not None:
        plot_histogram(outdegree, f'{plot_folder}{save}_outdegree_hist',
                       bins=bins, xlim=hist_xlim, xlabel='degree',
                       ylabel='# of stations', plot_format=plot_format,
                       dpi=dpi)


def plot_od_matrix(city, df, plot_folder, save, cmap=None, figsize=None,
                   dpi=150, plot_format='png', relative=False):
    """
    Plot the origin destination matrix for a given city.
    :param city:
    :param df:
    :param plot_folder:
    :param save:
    :param cmap:
    :param figsize:
    :param dpi:
    :param plot_format:
    :param relative:
    :return:
    """
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis_r')
    if figsize is None:
        figsize = [10, 10]

    if relative:
        dfplot = df.divide(df.max().max())
    else:
        dfplot = df
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    c = ax.pcolor(dfplot, cmap=cmap, norm=LogNorm(vmin=dfplot.min().min(),
                                                  vmax=dfplot.max().max()))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size="5%", pad=0.05)
    plt.colorbar(c, cax=cax, orientation='horizontal')
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False,
                   labelleft=False)
    cax.tick_params(axis='x', labelsize=16)
    cax.set_xlabel('cyclists per trip', fontsize=18)

    ax.set_title(f'Trips in {city}', fontsize='x-large')
    fig.savefig(f'{plot_folder}{save}_od_matrix.{plot_format}',
                format=plot_format, bbox_inches='tight')


def plot_stats(saves, colors, data_folder, plot_folder, plot_format='png',
               figsize=None, dpi=150):
    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    degree = {}
    indegree = {}
    outdegree = {}
    trips = {}
    imbalance = {}
    avg_trip_len = {}
    for city, save in saves.items():
        print(city)
        city_plot = f'{plot_folder}{save}/'
        Path(city_plot).mkdir(parents=True, exist_ok=True)
        data = h5py.File(f'{data_folder}{save}/{save}_analysis.hdf5', 'r')
        degree[city] = data['station degree'][()]
        indegree[city] = data['station indegree'][()]
        outdegree[city] = data['station outdegree'][()]
        print(f'Min degree {min(degree[city])}, '
              f'max degree {max(degree[city])}')

        trips[city] = data['trips'][()]
        # total_trips = sum(trips[city])
        unique_trips = len(set(trips[city]))
        test = []
        for idx, t_1 in enumerate(sorted(trips[city])):
            test.extend([idx] * t_1)
        xticks = {i: str(i) for i in np.arange(101, step=25)}
        plot_histogram(test, f'{city_plot}{save}_trips_test',
                       bins=len(set(test)) + 1, xlim=[0, len(set(test))],
                       xlabel='', ylabel='# of Cylists', cumulative=True,
                       density=True, xticks=xticks,
                       plot_format=plot_format, dpi=dpi)
        hist_xlim = [0, max(trips[city])]
        plot_histogram(trips[city], f'{city_plot}{save}_trips',
                       bins=unique_trips, xlim=hist_xlim,
                       xlabel='Cyclists per Trip', ylabel='# of Trips',
                       plot_format=plot_format, dpi=dpi)
        print(f'Min cyclists {min(trips[city])}, '
              f'max cyclists {max(trips[city])}')

        imbalance[city] = data['imbalance'][()]
        hist_xlim = [-0.05, 1.05]
        bins = round(len(set(imbalance[city])) / 4)
        plot_histogram(imbalance[city], f'{city_plot}{save}_imbalance',
                       bins=bins, xlim=hist_xlim, xlabel='Imbalance of Trip',
                       ylabel='# of Trips', plot_format=plot_format, dpi=dpi)
        counter = Counter(imbalance[city])
        imb_sum = sum(counter.values())
        imb_0 = sum([v for k, v in counter.items() if k <= 0.1]) / imb_sum
        imb_1 = sum([v for k, v in counter.items() if k >= 0.9]) / imb_sum
        print(f'Trips with low imb: {imb_0:3.2f}, '
              f'trips with high imb: {imb_1:3.2f}')

        avg_trip_len[city] = data['avg trip length'][()]

    deg_median = {}
    deg_mean = {}
    deg_std = {}
    deg_var = {}
    trip_median = {}
    trip_mean = {}
    trip_std = {}
    trip_var = {}
    trip_ttp = {}
    trip_ltp = {}
    for k, v in degree.items():
        deg_mean[k] = np.mean(v)
        deg_median[k] = np.median(v)
        deg_std[k] = stats.tstd(v)
        deg_var[k] = stats.variation(v)
    plot_barh(deg_mean, colors, plot_folder + 'deg_mean',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(deg_median, colors, plot_folder + 'deg_median',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(deg_std, colors, plot_folder + 'deg_std',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(deg_var, colors, plot_folder + 'deg_var',
              plot_format=plot_format, dpi=dpi, figsize=figsize)

    for k, v in trips.items():
        trip_median[k] = np.mean(v)
        trip_mean[k] = np.median(v)
        trip_std[k] = stats.tstd(v)
        trip_var[k] = stats.variation(v)
        trips_s = sorted(v)
        print(f'10 most used trips: {trips_s[-10:]}')
        trip_ttp[k] = sum(trips_s[round(0.90 * len(v)):]) / sum(v)
        trip_ltp[k] = sum(trips_s[:round(0.50 * len(v))]) / sum(v)
    plot_barh(trip_median, colors, plot_folder + 'trip_mean',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(trip_mean, colors, plot_folder + 'trip_median',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(trip_std, colors, plot_folder + 'trip_std',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(trip_var, colors, plot_folder + 'trip_var',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(trip_var, colors, plot_folder + 'trip_var',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(trip_var, colors, plot_folder + 'trip_var',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(trip_ttp, colors, plot_folder + 'trip_ttp',
              plot_format=plot_format, dpi=dpi, figsize=figsize)
    plot_barh(trip_ltp, colors, plot_folder + 'trip_ltp',
              plot_format=plot_format, dpi=dpi, figsize=figsize)

    plot_barh(avg_trip_len, colors, plot_folder + 'avg_trip_len',
              plot_format=plot_format, dpi=dpi, figsize=figsize)


def plot_edge_types(G, ns, save, plot_folder, plot_format='png',
                    figsize=(20, 20), dpi=150):
    colors = {'primary': 'darkblue', 'secondary': 'darkgreen',
              'tertiary': 'darkcyan', 'residential': 'darkorange'}
    ec = get_edge_color_st(G, colors)
    save_path = f'{plot_folder}{save}_networt_st.{plot_format}'
    plot_edges(G, ec, node_size=ns, save_path=save_path,
               plot_format=plot_format, figsize=figsize, dpi=dpi)


def plot_used_area(g, area, stations, stations_color='C0', station_size=100,
                   figsize=None, dpi=150, folder='', filename='',
                   plot_format='png'):
    h = nx.Graph(nx.induced_subgraph(g, stations))
    h.remove_edges_from(list(h.edges))

    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3857')
    node_pos = {n: transform(inProj, outProj, d['y'], d['x']) for n, d in
                h.nodes(data=True)}
    plot_nx_graph(h, node_pos=node_pos, node_size=station_size,
                  node_color=stations_color, bg_area=area, scalebar=True,
                  overlay_poly=area, figsize=figsize, dpi=dpi, folder=folder,
                  filename=filename, plot_format=plot_format)


def plot_load(city, save, G, edited_edges, trip_nbrs, node_size, rev, minmode,
              plot_folder, plot_format='png', figsize=(20, 20)):
    G_calc = nx.Graph(G.to_undirected())
    nx.set_edge_attributes(G, False, 'bike path')
    nx.set_edge_attributes(G, 0, 'load')
    edge_dict = calc_current_state(nxG=G_calc, bike_paths=edited_edges,
                                   trip_nbrs=trip_nbrs)[6]

    max_load_bp = max([edge_info['load'] for edge_info in edge_dict.values()
                       if edge_info['bike path']])
    max_load_no = max([edge_info['load'] for edge_info in edge_dict.values()
                       if not edge_info['bike path']])
    max_load = max([max_load_bp, max_load_no])

    for edge in edited_edges:
        G[edge[0]][edge[1]][0]['bike path'] = True
        G[edge[1]][edge[0]][0]['bike path'] = True
    for edge, edge_info in edge_dict.items():
        G[edge[0]][edge[1]][0]['load'] = edge_info['load']
        G[edge[1]][edge[0]][0]['load'] = edge_info['load']

    cmap_bp = plt.cm.get_cmap('Blues')
    cmap_no = plt.cm.get_cmap('Reds')
    ec = []
    for u, v, data in G.edges(keys=False, data=True):
        if data['bike path']:
            ec.append(cmap_bp(data['load'] / max_load))
        else:
            ec.append(cmap_no(data['load'] / max_load))

    fig, ax = plt.subplots(dpi=300, figsize=figsize)
    ox.plot_graph(G, ax=ax, bgcolor='#ffffff',
                  node_size=node_size, node_color='C0', node_zorder=3,
                  edge_linewidth=3, edge_color=ec, show=False, close=False)
    fig.suptitle(f'Edge load in {city}', fontsize='x-large')
    plt.savefig(f'{plot_folder}{save}-load-{rev:d}{minmode}.{plot_format}',
                format=plot_format)
    plt.close(fig)


def plot_used_nodes(save, G, trip_nbrs, stations, plot_folder,
                    figsize=(3.5, 3.5), dpi=150, plot_format='svg'):
    """
    Plots usage of nodes in graph G. trip_nbrs and stations should be
    structured as returned from load_trips().
    :param G: graph to plot in.
    :type G: networkx graph
    :param trip_nbrs: trips to plot the usage of.
    :type trip_nbrs: dict
    :param stations: list of stations.
    :type stations: list
    :param save: save name for the plot.
    :type save: str
    :param figsize: width and height of the plot.
    :type figsize: tuple
    :param dpi: dpi of the plot.
    :type dpi: int
    :param plot_folder:
    :type plot_folder: str
    :type plot_format:
    :return: None
    """
    print('Plotting used nodes.')

    nodes = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if e_node == s_node:
                # print('Round trip')
                continue
            if (s_node, e_node) in trip_nbrs:
                nodes[s_node] += trip_nbrs[(s_node, e_node)]
                nodes[e_node] += trip_nbrs[(s_node, e_node)]

    nodes = {n: int(t * (365 / 1246)) for n, t in nodes.items()}
    # DD: 881 HH: 1246

    # trip_count = sum(trip_nbrs.values())
    # station_count = len(stations)

    max_n = max(nodes.values())

    print(f'Maximal station usage: {max_n}')

    n_rel = {key: value for key, value in nodes.items()}
    ns = [4 if n in stations else 0 for n in G.nodes()]  # 20

    for n in G.nodes():
        if n not in stations:
            n_rel[n] = max_n + 1
    min_n = min(n_rel.values())

    print(f'Minimal station usage: {min_n}')

    r = magnitude(max_n)

    hist_data = [value for key, value in n_rel.items() if value != max_n + 1]
    hist_save = f'{plot_folder}{save}_stations_usage_distribution'
    hist_xlim = (0.0, round(max_n, -(r - 1)))

    cmap_name = 'cool'
    cmap = plt.cm.get_cmap(cmap_name)
    print('MITTELWERT:', np.mean(hist_data))
    plot_histogram(hist_data, hist_save,
                   xlabel='total number of trips per year',
                   ylabel='number of stations', xlim=hist_xlim, bins=25,
                   cm=cmap, plot_format=plot_format, dpi=150,
                   figsize=figsize)    # (3.65, 1.3)

    cmap = ['#808080'] + \
           [rgb2hex(cmap(n)) for n in reversed(np.linspace(1, 0, max_n,
                                                           endpoint=True))] \
           + ['#ffffff']
    color_n = [cmap[v] for k, v in n_rel.items()]

    fig2, ax2 = plt.subplots(dpi=dpi, figsize=(1.5, 1.4))  # 1.5, 1.4 or 3.5
    ox.plot_graph(G, ax=ax2, bgcolor='#ffffff', edge_linewidth=0.3,
                  node_color=color_n, node_size=ns, node_zorder=3,
                  show=False, close=False, edge_color='#b3b3b3')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap_name),
                               norm=plt.Normalize(vmin=0, vmax=max_n))

    cbaxes = fig2.add_axes([0.1, 0.05, 0.8, 0.03])
    # divider = make_axes_locatable(ax2)
    # cbaxes = divider.append_axes('bottom', size="5%", pad=0.05)

    if min_n <= 0.1 * max_n:
        r = magnitude(max_n)
        cbar = fig2.colorbar(sm, orientation='horizontal', cax=cbaxes,
                             ticks=[0, round(max_n / 2), max_n])
        cbar.ax.set_xticklabels([0, int(round(max_n / 2, -(r - 2))),
                                 round(max_n, -(r - 1))])
    else:
        max_r = magnitude(max_n)
        min_r = magnitude(min_n)
        cbar = fig2.colorbar(sm, orientation='horizontal', cax=cbaxes,
                             ticks=[0, min_n, round(max_n / 2), max_n])
        cbar.ax.set_xticklabels([0, round(min_n, -(min_r - 2)),
                                 int(round(max_n / 2, -(max_r - 2))),
                                 round(max_n, -(max_r - 2))])

    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(axis='x', labelsize=8, width=0.5)
    cbar.ax.set_xlabel('total number of trips per year', fontsize=8)

    # fig2.suptitle(f'{city.capitalize()}, Stations: {station_count}, '
    #               f'Trips: {trip_count}', fontsize=24)
    fig2.savefig(f'{plot_folder}{save}_stations_used.{plot_format}',
                 format=plot_format, bbox_inches='tight')

    plt.close('all')


def plot_edges(G, edge_color, node_size, save_path, plot_format='png',
               figsize=(12, 12), dpi=150, title=''):
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ox.plot_graph(G, ax=ax, bgcolor='#ffffff', edge_color=edge_color,
                  edge_linewidth=1.5, node_color='C0', node_size=node_size,
                  node_zorder=3, show=False, close=False)
    if title != '':
        ax.set_title(title, fontsize=28)
    fig.savefig(save_path, format=plot_format, bbox_inches='tight')
    plt.close(fig)


def plot_bp_evo(save, G, edited_edges, bike_path_perc, cut, ps,
                node_size, rev, minmode, plot_folder, titles=False,
                plot_format='png'):
    print('Begin plotting bike path evolution.')
    plot_folder_evo = plot_folder + 'evolution/'
    Path(plot_folder_evo).mkdir(parents=True, exist_ok=True)
    edited_color = 'midnightblue'
    nx.set_edge_attributes(G, False, 'bike path')
    if rev:
        ee = edited_edges
        bpp = bike_path_perc
    else:
        ee = list(reversed(edited_edges))
        bpp = list(reversed(bike_path_perc))

    plots = np.linspace(0, 1, 101)
    for i, j in enumerate(plots):
        idx = next(x for x, val in enumerate(bpp) if val >= j)
        if j == 1.0:
            ec_evo = [edited_color for e in G.edges()]
        else:
            ee_evo = ee[:idx]
            ec_evo = get_edge_color(G, ee_evo, 'bike path', edited_color)
        save_path = f'{plot_folder_evo}{save}-edited-mode-{rev:d}{minmode}-' \
                    f'{i}.{plot_format}'
        if titles:
            plot_edges(G, ec_evo, node_size, save_path,
                       title=f'Fraction of Bike Paths: {j*100:3.0f}%',
                       plot_format=plot_format, figsize=(10, 10), dpi=150)
        else:
            plot_edges(G, ec_evo, node_size, save_path,
                       plot_format=plot_format, figsize=(10, 10), dpi=150)

    nx.set_edge_attributes(G, False, 'bike path')
    ee_cut = ee[:cut]
    ec_cut = get_edge_color(G, ee_cut, 'bike path', edited_color)
    save_path = f'{plot_folder_evo}{save}-edited-mode-{rev:d}{minmode}-' \
                f'cut.{plot_format:}'
    plot_edges(G, ec_cut, node_size, save_path, plot_format=plot_format,
               figsize=(10, 10), dpi=150)

    nx.set_edge_attributes(G, False, 'bike path')
    ee_ps = ee[:ps]
    ec_ps = get_edge_color(G, ee_ps, 'bike path', edited_color)
    save_path = f'{plot_folder_evo}{save}-edited-mode-{rev:d}{minmode}-' \
                f'ps.{plot_format}'
    plot_edges(G, ec_ps, node_size, save_path, plot_format=plot_format,
               figsize=(10, 10), dpi=150)
    print('Finished plotting bike path evolution.')


def plot_bp_comparison(city, save, G, ee_algo, ee_cs, bpp_algo, bpp_cs,
                       node_size, trip_nbrs, rev, minmode, plot_folder,
                       plot_format='png', figsize=None, dpi=150, mode='diff'):
    if figsize is None:
        figsize = [10, 10]
    nx.set_edge_attributes(G, False, 'algo')
    nx.set_edge_attributes(G, False, 'cs')

    if rev:
        ee_algo = ee_algo
        bpp_algo = bpp_algo
    else:
        ee_algo = list(reversed(ee_algo))
        bpp_algo = list(reversed(bpp_algo))

    idx = min(range(len(bpp_algo)), key=lambda i: abs(bpp_algo[i] - bpp_cs))

    print(f'Difference in BPP between p+s and algo: '
          f'{abs(bpp_cs - bpp_algo[idx]):6.5f}')

    ee_algo_cut = ee_algo[:idx]
    for edge in ee_algo_cut:
        G[edge[0]][edge[1]][0]['algo'] = True
        G[edge[1]][edge[0]][0]['algo'] = True
    for edge in ee_cs:
        G[edge[0]][edge[1]][0]['cs'] = True
        G[edge[1]][edge[0]][0]['cs'] = True

    ec = []
    unused = []
    ee_algo_only = []
    ee_cs_only = []
    ee_both = []

    cmap_name = 'plasma'
    cmap = plt.cm.get_cmap(cmap_name)
    # c = [cmap(n) for n in reversed(np.linspace(1, 0, 10, endpoint=True))]

    color_algo = '#007fbfff'  #000075'  # '0080c0'
    color_cs = '#40e640'  # '40e640'
    color_both = '#f58231'  # '40e6c0'
    color_unused = '#7f7f7fff' #808080'

    len_algo = 0
    len_cs = 0
    len_both = 0

    if mode == 'algo':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['algo']:
                ec.append(color_algo)
            else:
                ec.append(color_unused)
                unused.append((u, v, k))
    elif mode == 'p+s':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['cs']:
                ec.append(color_cs)
            else:
                ec.append(color_unused)
                unused.append((u, v, k))
    elif mode == 'diff':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['algo'] and data['cs']:
                ec.append(color_both)
                ee_both.append((u, v, k))
                len_both += data['length']
            elif data['algo'] and not data['cs']:
                ec.append(color_algo)
                ee_algo_only.append((u, v, k))
                len_algo += data['length']
            elif not data['algo'] and data['cs']:
                ec.append(color_cs)
                ee_cs_only.append((u, v, k))
                len_cs += data['length']
            else:
                ec.append(color_unused)
                unused.append((u, v, k))
        print(f'Overlap between p+s and algo: '
              f'{len_both / (len_cs + len_both):3.2f}')
    else:
        print('You have to choose between algo, p+s and diff.')

    fig, ax = plt.subplots(dpi=dpi, figsize=(2.2, 2.2))
    # dd(2.2, 2.2), hh (2.6, 2.6)
    ox.plot_graph(G, bgcolor='#ffffff', ax=ax,
                  node_size=node_size, node_color='#d726ffff', node_zorder=3,
                  edge_linewidth=0.6, edge_color=ec,
                  show=False, close=False)
    if mode == 'algo':
        leg = [Line2D([0], [0], color=color_algo, lw=4),
               Line2D([0], [0], color=color_unused, lw=4)]
        """ax.legend(leg, ['Algorithm', 'None'],
                  bbox_to_anchor=(0, -0.05, 1, 1), loc=3,
                  ncol=2, mode="expand", borderaxespad=0., fontsize=12)
        ax.set_title('Algorithm', fontsize=24)"""
    elif mode == 'p+s':
        leg = [Line2D([0], [0], color=color_cs, lw=4),
               Line2D([0], [0], color=color_unused, lw=4)]
        ax.legend(leg, ['Primary + Secondary', 'None'],
                  bbox_to_anchor=(0, -0.05, 1, 1), loc=3,
                  ncol=2, mode="expand", borderaxespad=0., fontsize=12)
        ax.set_title('Primary/Secondary', fontsize=24)
    elif mode == 'diff':
        leg = [Line2D([0], [0], color=color_both, lw=4),
               Line2D([0], [0], color=color_algo, lw=4),
               Line2D([0], [0], color=color_cs, lw=4),
               Line2D([0], [0], color=color_unused, lw=4)]
        ax.legend(leg, ['Both', 'Algorithm', 'Primary + Secondary', 'None'],
                  bbox_to_anchor=(0, -0.05, 1, 1), loc=3,
                  ncol=4, mode="expand", borderaxespad=0., fontsize=12)
        ax.set_title('Comparison', fontsize=24)
    plt.savefig(f'{plot_folder}{save}-bp-build-{rev:d}{minmode}_{mode}'
                f'.{plot_format}', format=plot_format, bbox_inches='tight')
    plt.close(fig)
    if mode == 'algo':
        plot_load(city, save, G, ee_algo_cut, trip_nbrs, node_size, rev,
                  minmode, plot_folder, plot_format=plot_format)


def plot_bp_diff(G, ee_1, ee_2, bpp_1, bpp_2, node_color,
                 node_size, save, rev, minmode, plot_folder,
                 plot_format='png', figsize=None, dpi=150):
    if figsize is None:
        figsize = [10, 10]

    plot_folder_evo = plot_folder + 'evolution/'
    Path(plot_folder_evo).mkdir(parents=True, exist_ok=True)

    nx.set_edge_attributes(G, False, '1')
    nx.set_edge_attributes(G, False, '2')
    if not rev:
        ee_1 = list(reversed(ee_1))
        # bpp_1 = list(reversed(bpp_1))
        ee_2 = list(reversed(ee_2))
        # bpp_2 = list(reversed(bpp_2))

    color_1 = 'midnightblue'
    color_2 = 'darkorange'
    color_both = 'crimson'
    color_unused = '#808080'

    plots = np.linspace(0, 1, 101)
    for i, j in enumerate(plots):
        idx_1 = next(x for x, val in enumerate(bpp_1) if val >= j)
        ee_1_evo = ee_1[:idx_1]
        idx_2 = next(x for x, val in enumerate(bpp_2) if val >= j)
        ee_2_evo = ee_2[:idx_2]
        for edge in ee_1_evo:
            G[edge[0]][edge[1]][0]['1'] = True
            G[edge[1]][edge[0]][0]['1'] = True
        for edge in ee_2_evo:
            G[edge[0]][edge[1]][0]['2'] = True
            G[edge[1]][edge[0]][0]['2'] = True
        ec = []
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['1'] and data['2']:
                ec.append(color_both)
            elif data['1'] and not data['2']:
                ec.append(color_1)
            elif not data['1'] and data['2']:
                ec.append(color_2)
            else:
                ec.append(color_unused)
        save_path = f'{plot_folder_evo}{save}-distr-comp-evo-{rev:d}{minmode}-' \
                    f'{i}.{plot_format}'
        plot_edges(G, ec, node_size, save_path, plot_format=plot_format,
                   figsize=figsize, dpi=dpi)
    """
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ox.plot_graph(G, bgcolor='#ffffff', ax=ax,
                  node_size=node_size, node_color='C0', node_zorder=3,
                  edge_linewidth=1.5, edge_color=ec,
                  show=False, close=False)
    save_plot = f'{plot_folder}{save}_distr_comp_'
    plt.savefig(f'{save_plot}.{plot_format}', format=plot_format, 
                bbox_inches='tight')
    """
    # plt.close(fig)


def plot_mode(city, save, data, data_now, nxG_calc, nxG_plot, stations,
              trip_nbrs, mode, end, hf_group, plot_folder, evo=False,
              plot_format='png', legends=False, titles=False):
    rev = mode[0]
    minmode = mode[1]

    bike_paths_now = data_now[0]
    cost_now = data_now[1]
    bike_path_perc_now = data_now[2]
    trdt_now = data_now[3]
    tfdt_now = data_now[4]
    nos_now = data_now[5]

    # edited_edges = data['ee_nk']
    # edited_edges_nx = data[1]
    edited_edges_nx = [(i[0], i[1]) for i in data['ee_nx'][()]]
    bike_path_perc = data['bpp'][()]
    # bike_path_perc = data[3]
    # cost = data[2]
    cost = data['cost'][()]
    # total_real_distance_traveled = data[4]
    total_real_distance_traveled = json.loads(data['trdt'][()])
    # total_felt_distance_traveled = data[5]
    total_felt_distance_traveled = json.loads(data['tfdt'][()])
    # nbr_on_street = data[6]
    nbr_on_street = data['nos'][()]
    # len_saved = data[7]
    # nbr_of_cbc = data[8]
    # gcbc_size = data[9]
    trdt, trdt_now = total_distance_traveled_list(total_real_distance_traveled,
                                                  trdt_now, rev)
    tfdt, tfdt_now = total_distance_traveled_list(total_felt_distance_traveled,
                                                  tfdt_now, rev)
    bl_st = len_of_bikepath_by_type(edited_edges_nx, nxG_calc, rev)
    bl_st_now = len_of_bikepath_by_type(bike_paths_now, nxG_calc, rev)
    bl_st_now = {st: length[-1] for st, length in bl_st_now.items()}

    if rev:
        bpp = bike_path_perc
    else:
        bpp = list(reversed(bike_path_perc))
    trdt_min = min(trdt['all'])
    trdt_max = max(trdt['all'])
    tfdt_min = min(tfdt['all'])
    tfdt_max = max(tfdt['all'])
    ba = [1 - (i - tfdt_min) / (tfdt_max - tfdt_min) for i in tfdt['all']]
    # ba = [tfdt_min / i for i in tfdt['all']]
    ba_now = 1 - (tfdt_now['all'] - tfdt_min) / (tfdt_max - tfdt_min)

    if rev:
        nos = [x / max(nbr_on_street) for x in nbr_on_street]
    else:
        nos = list(reversed([x / max(nbr_on_street) for x in nbr_on_street]))
    nos_now = nos_now / max(nbr_on_street)
    los = trdt['street']
    los_now = trdt_now['street']

    trdt_st = {st: len_on_st for st, len_on_st in trdt.items()
               if st not in ['street', 'all']}
    trdt_st_now = {st: len_on_st for st, len_on_st in trdt_now.items()
                   if st not in ['street', 'all']}

    bpp_cut = [i / bpp[end] for i in bpp[:end]]
    bpp_now = bike_path_perc_now / bpp[end]

    bpp_x = min(bpp_cut, key=lambda x: abs(x - bpp_now))
    bpp_idx = next(x for x, val in enumerate(bpp_cut) if val == bpp_x)
    ba_y = ba[bpp_idx]
    ba_improve = ba_y - ba_now

    cost_y = min(cost[:end], key=lambda x: abs(x - cost_now))
    cost_idx = next(x for x, val in enumerate(cost[:end]) if val == cost_y)
    cost_x = bpp_cut[cost_idx]

    nos_y = min(nos[:end], key=lambda x: abs(x - nos_now))
    nos_idx = next(x for x, val in enumerate(nos[:end]) if val == nos_y)
    nos_x = bpp_cut[nos_idx]
    nos_y = nos[bpp_idx]
    nos_improve = nos_now - nos[bpp_idx]

    los_y = min(los[:end], key=lambda x: abs(x - los_now))
    los_idx = next(x for x, val in enumerate(los[:end]) if val == los_y)
    los_x = bpp_cut[los_idx]
    los_y = los[bpp_idx]
    los_improve = los_now - los[bpp_idx]

    cut = next(x for x, val in enumerate(ba) if val >= 1)
    total_cost, cost_now = sum_total_cost(cost, cost_now, rev)

    cost_now = cost_now / total_cost[end]
    # gcbc_size_normed = [i / max(gcbc_size) for i in reversed(gcbc_size)]

    ns = [4 if n in stations else 0 for n in nxG_plot.nodes()]
    cmap_name = 'viridis'
    cmap = plt.cm.get_cmap(cmap_name)
    c = [cmap(n) for n in np.linspace(1, 0, 9, endpoint=True)]

    max_bpp = max(bpp[end], bpp[cut])

    print(f'Mode: {rev:d}{minmode}, ba=1 after: {end:d}, '
          f'bpp at ba=1: {bpp[end]:3.2f}, '
          f'bpp big roads: {bpp_now*bpp[end]:3.2f}, '
          f'edges: {len(edited_edges_nx)}, max bpp: {max_bpp:3.2f}')

    print(f'Reached ba=0.8 at '
          f'{bpp[next(x for x, val in enumerate(ba) if val >= 0.8)]:3.2f} bpp')
    print(f'Reached ba=0.8 at '
          f'{bpp_cut[next(x for x, val in enumerate(ba) if val >= 0.8)]:3.2f}'
          f'normed bpp')

    # Plotting
    fig1, ax1 = plt.subplots(dpi=150, figsize=(6, 5)) #(1.9, 1.675)
    ax12 = ax1.twinx()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax12.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax12.set_ylim(0.0, 1.0)

    c_ba = '#0080c0'

    ax1.plot(bpp_cut, ba[:end], c=c_ba, label='bikeability', lw=2.5)
    ax1.plot(bpp_now, ba_now, c=c_ba, ms=12, marker='D')
    xmax, ymax = coord_transf(bpp_now, max([ba_y, ba_now]),
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=bpp_now, ymax=ymax, ymin=0, c=c_ba, ls='--',
                alpha=0.5)
    ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--',
                alpha=0.5)
    ax1.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    print('ba:', ba_y)

    ax1.set_ylabel('bikeability', fontsize=20, color=c_ba) #b(m)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=c_ba)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    c_cost = '#4d4d4d'

    ax12.plot(bpp_cut, [x / total_cost[end] for x in total_cost[:end]],
              c=c_cost, label='total cost', lw=2.5)
    ax12.plot(bpp_now, cost_now, c=c_cost, ms=11, marker='s')
    xmin, ymax = coord_transf(bpp_now, cost_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=bpp_now, ymax=ymax, ymin=0, c=c_cost, ls='--',
                alpha=0.5)
    ax1.axhline(y=cost_now, xmax=1, xmin=xmin, c=c_cost, ls='--',
                alpha=0.5)
    ax1.axhline(y=cost_y, xmax=1, xmin=xmin, c=c_cost, ls='--',
                alpha=0.5)
    # ax1.axvline(x=bpp[cut] / bpp[end], c='#999999', ls='--', alpha=0.7, lw=3)

    ax12.set_ylabel('normalised cost', fontsize=20, color=c_cost)
    ax12.tick_params(axis='y', labelsize=12, labelcolor=c_cost)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax12.yaxis.set_minor_locator(AutoMinorLocator())

    ax1.set_xlabel('normalised fraction of bike paths', fontsize=20)
    if titles:
        ax1.set_title('Bikeability and Cost', fontsize=20)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(False)
    ax12.grid(False)

    handles = ax1.get_legend_handles_labels()[0]
    handles.append(ax12.get_legend_handles_labels()[0][0])
    if legends:
        ax1.legend(handles=handles, loc='lower right', fontsize=6)

    fig1.savefig(f'{plot_folder}{save}_ba_tc_mode_{rev:d}{minmode}'
                 f'.{plot_format}', format=plot_format, bbox_inches='tight')

    ax1ins = zoomed_inset_axes(ax1, 3.5, loc=1)
    x1, x2, y1, y2 = round(bpp_now - 0.05, 2), round(bpp_now + 0.05, 2), \
                     round(ba_now - 0.03, 2), min(round(ba_y + 0.03, 2), 1)
    ax1ins.plot(bpp_cut, ba[:end], lw=2.5)
    ax1ins.plot(bpp_now, ba_now, c=c_ba, ms=12, marker='D')
    xmax, ymax = coord_transf(bpp_now, max([ba_y, ba_now]),
                              xmin=x1, xmax=x2, ymin=y1, ymax=y2)
    ax1ins.axvline(x=bpp_now, ymax=ymax, ymin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)

    ax1ins.set_xlim(x1, x2)
    ax1ins.set_ylim(y1, y2)
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")
    ax1ins.tick_params(axis='y', labelsize=10, labelcolor=c_ba)
    ax1ins.tick_params(axis='x', labelsize=10)
    ax1ins.yaxis.set_minor_locator(AutoMinorLocator())
    ax1ins.xaxis.set_minor_locator(AutoMinorLocator())
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")

    fig1.savefig(f'{plot_folder}{save}_ba_tc_zoom_mode_{rev:d}{minmode}'
                 f'.{plot_format}', format=plot_format, bbox_inches='tight')

    fig2, ax2 = plt.subplots(dpi=150, figsize=(6, 5))
    ax22 = ax2.twinx()
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax22.set_ylim(0.0, 1.0)

    c_nos = '#e6194B'
    c_los = '#911eb4'

    p1, = ax2.plot(bpp_cut, los[:end], label='length', c=c_los, lw=2.5,
                   zorder=1)
    ax2.plot(bpp_now, los_now, c=c_los, ms=12, marker='8', zorder=3)
    xmax, ymax = coord_transf(max(bpp_now, los_x), los_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax2.axvline(x=bpp_now, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5,
                zorder=2)
    ax2.axhline(y=los_now, xmax=xmax, xmin=0, c=c_los, ls='--', alpha=0.5,
                zorder=2)
    # ax2.axvline(x=los_x, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5)
    ax2.axhline(y=los_y, xmax=xmax, xmin=0, c=c_los, ls='--', alpha=0.5,
                zorder=2)

    ax2.set_ylabel('length on street', fontsize=20, color=c_los)
    ax2.tick_params(axis='y', labelsize=12, labelcolor=c_los)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    p2, = ax22.plot(bpp_cut, nos[:end], label='cyclists', c=c_nos, lw=2.5,
                    zorder=1)
    ax22.plot(bpp_now, nos_now, c=c_nos, ms=12, marker='v', zorder=3)
    xmin, ymax = coord_transf(max(bpp_now, nos_x), nos_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax22.axvline(x=bpp_now, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5,
                 zorder=2)
    ax22.axhline(y=nos_now, xmax=1, xmin=xmin, c=c_nos, ls='--', alpha=0.5,
                 zorder=2)
    # ax22.axvline(x=nos_x, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5)
    ax22.axhline(y=nos_y, xmax=1, xmin=xmin, c=c_nos, ls='--', alpha=0.5,
                 zorder=2)

    ax22.set_ylabel('cyclists on street', fontsize=20, color=c_nos)
    ax22.tick_params(axis='y', labelsize=12, labelcolor=c_nos)
    ax22.yaxis.set_minor_locator(AutoMinorLocator())

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='x', labelsize=12)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=20)
    if titles:
        ax2.set_title('Cyclists and Length on Street', fontsize=20)
    if legends:
        ax2.legend([p1, p2], [l.get_label() for l in [p1, p2]], fontsize=22)
    fig2.savefig(f'{plot_folder}{save}_trips_on_street_mode_{rev:d}{minmode}'
                 f'.{plot_format}', format=plot_format, bbox_inches='tight')

    fig3, ax3 = plt.subplots(dpi=150, figsize=(12, 10))
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)

    c_st = {'primary': '#4d4d4d', 'secondary': '#666666',
            'tertiary': '#808080', 'residential': '#999999',
            'bike paths': '#0080c0'}
    m_st = {'primary': 'p', 'secondary': 'p', 'tertiary': 'p',
            'residential': 'p', 'bike paths': 'P'}

    """for st, len_on_st in trdt_st_now.items():
        xmax, ymax = coord_transf(bpp_now, len_on_st,
                                  xmax=1, xmin=0, ymax=1, ymin=0)
        ax3.axvline(x=bpp_now, ymax=ymax, ymin=0, c=c_st[st], ls='--',
                    alpha=0.5)
        ax3.axhline(y=len_on_st, xmax=xmax, xmin=0, c=c_st[st], ls='--',
                    alpha=0.5)"""

    for st, len_on_st in trdt_st.items():
        ax3.plot(bpp_cut, len_on_st[:end], c=c_st[st], label=st, lw=2.5)
    for st, len_on_st in trdt_st.items():
        ax3.plot(bpp_now, trdt_now[st], c=c_st[st], ms=16, marker=m_st[st])

    ax3.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax3.set_ylabel('length', fontsize=24)
    if titles:
        ax3.set_title(f'Length on Street Type in {city}', fontsize=28)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    if legends:
        ax3.legend(fontsize=22)
    fig3.savefig(f'{plot_folder}{save}_len_on_street_mode_{rev:d}{minmode}'
                 f'.{plot_format}', format=plot_format, bbox_inches='tight')

    fig4, ax4 = plt.subplots(dpi=150, figsize=(12, 10))
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)

    """for st, len_by_st_now in bl_st_now.items():
        xmax, ymax = coord_transf(bpp_now, len_by_st_now)
        ax4.axvline(x=bpp_now, ymax=ymax, ymin=0, c=c_st[st], ls='--',
                    alpha=0.5)
        ax4.axhline(y=len_by_st_now, xmax=xmax, xmin=0, c=c_st[st], ls='--',
                    alpha=0.5)"""
    for st, len_by_st in bl_st.items():
        if len_by_st[end] == 0:
            len_norm = 1
        else:
            len_norm = len_by_st[end]
        ax4.plot(bpp_cut, [x / len_norm for x in len_by_st[:end]],
                 c=c_st[st], label=f'{st}', lw=2.5)
    """for st, len_by_st in bl_st.items():
        ax4.plot(bpp_now, bl_st_now[st], ms=16, marker=m_st[st],  c=c_st[st])
    """

    ax4.axvline(x=bpp[cut] / bpp[end], c='#999999', ls='--', alpha=0.7, lw=3)

    ax4.set_xlabel('fraction of bike paths', fontsize=24)
    ax4.set_ylabel('length', fontsize=24)
    if titles:
        ax4.set_title(f'Length of Bike Paths along Streets in {city}',
                      fontsize=28)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    if legends:
        ax4.legend(fontsize=22)
    fig4.savefig(f'{plot_folder}{save}_len_bl_mode_{rev:d}{minmode}'
                 f'.{plot_format}', format=plot_format, bbox_inches='tight')

    comp_st_driven = {st: [len_on_st[bpp_idx], trdt_st_now[st]]
                      for st, len_on_st in trdt_st.items()}
    plot_barv_stacked(['Algorithm', 'P+S'], comp_st_driven, c_st, width=0.3,
                      title='', save=f'{plot_folder}{save}_comp_st_driven_'
                                     f'{rev:d}{minmode}',
                      plot_format=plot_format, figsize=(0.75, 1.85))

    diff_core = {'Bikeability': ba_improve, '# Trips': nos_improve,
                 'Length': los_improve}
    c_diff_core = {'Bikeability': c_ba, '# Trips': c_nos,
                   'Length': c_los}
    plot_barv(diff_core, c_diff_core,
              f'{plot_folder}{save}_improvement_core_{rev:d}{minmode}',
              plot_format=plot_format, figsize=(10, 12), y_label='',
              title='Change compared to primary and secondary only')

    diff_st = {st: trdt_st_now[st] - len_on_st[bpp_idx]
               for st, len_on_st in trdt_st.items() if st != 'bike paths'}
    diff_st['bike paths'] = trdt_st['bike paths'][bpp_idx] - \
                            trdt_st_now['bike paths']
    c_diff_st = {st: c for st, c in c_st.items()}
    plot_barv(diff_st, c_diff_st, f'{plot_folder}{save}_improvement_st_'
                                  f'{rev:d}{minmode}',
              plot_format=plot_format, figsize=(10, 12), y_label='',
              title='Change compared to primary and secondary only')
    diff = deepcopy(diff_core)
    diff.update(diff_st)
    c_diff = deepcopy(c_diff_core)
    c_diff.update(c_diff_st)
    plot_barv(diff, c_diff, f'{plot_folder}{save}_improvement_'
                            f'{rev:d}{minmode}',
              plot_format=plot_format, figsize=(10, 11), y_label='',
              title='Change compared to primary and secondary only')

    for bp_mode in ['algo', 'p+s', 'diff']:
        plot_bp_comparison(city=city, save=save, G=nxG_plot,
                           ee_algo=edited_edges_nx, ee_cs=bike_paths_now,
                           bpp_algo=bike_path_perc, bpp_cs=bike_path_perc_now,
                           node_size=ns, trip_nbrs=trip_nbrs, rev=rev,
                           minmode=minmode, plot_folder=plot_folder,
                           plot_format=plot_format, mode=bp_mode)

    if evo:
        plot_bp_evo(save=save, G=nxG_plot, edited_edges=edited_edges_nx,
                    bike_path_perc=bike_path_perc, cut=cut, ps=bpp_idx,
                    node_size=ns, rev=rev, minmode=minmode, titles=titles,
                    plot_folder=plot_folder, plot_format=plot_format)

    # plt.show()
    plt.close('all')

    print(f'ba: {ba_y:4.3f}, ba now: {ba_now:4.3f}, '
          f'los: {los_y:4.3f}, los now: {los_now:4.3f}, '
          f'nos: {nos_y:4.3f}, nos now: {nos_now:4.3f}, '
          f'bpp now: {bpp_now:4.3f}')

    hf_group['edited edges'] = edited_edges_nx
    hf_group['end'] = end
    hf_group['bpp'] = bpp_cut
    hf_group['bpp at end'] = bpp[end]
    hf_group['bpp complete'] = bike_path_perc
    hf_group['ba'] = ba[:end]
    hf_group['ba complete'] = ba
    hf_group['ba for comp'] = ba_y
    hf_group['cost'] = total_cost[:end]
    hf_group['nos'] = nos[:end]
    hf_group['nos complete'] = nos
    hf_group['nos at comp'] = nos_y
    hf_group['los'] = los[:end]
    hf_group['los complete'] = los
    hf_group['los at comp'] = los_y
    hf_group['tfdt max'] = tfdt_max
    hf_group['tfdt min'] = tfdt_min
    hf_group['trdt max'] = trdt_max
    hf_group['trdt min'] = trdt_min

    return bpp_now, ba_now, cost_now, nos_now, los_now


def compare_modes(city, save, label, comp_folder, color, plot_folder,
                  plot_format='png', legends=False, titles=False):
    hf = h5py.File(f'{comp_folder}comp_{save}.hdf5', 'r')

    bpp_now = hf['p+s']['bpp'][()]
    ba_now = hf['p+s']['ba'][()]
    nos_now = hf['p+s']['nos'][()]
    los_now = hf['p+s']['los'][()]

    fig1, ax1 = plt.subplots(dpi=150, figsize=(12, 10))
    ax1.set_xlabel('fraction of bike paths', fontsize=24)
    ax1.set_ylabel('bikeability', fontsize=24)
    if titles:
        ax1.set_title(f'Bikeability of {city}', fontsize=28)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)

    fig2, ax2 = plt.subplots(dpi=150, figsize=(12, 10))
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=16)
    ax2.set_ylabel('integrated bikeability', fontsize=16)
    if titles:
        ax2.set_title(f'Integrated Bikeability of {city}', fontsize=28)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)

    fig3, ax3 = plt.subplots(dpi=150, figsize=(12, 10))
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax3.set_ylabel('detour per cost', fontsize=24)
    if titles:
        ax3.set_title(f'CBA of {city}', fontsize=28)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)

    fig4, ax4 = plt.subplots(dpi=150, figsize=(12, 10))
    ax42 = ax4.twinx()
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)
    ax42.set_ylim(0.0, 1.0)

    ax4.set_ylabel('length on street', fontsize=24)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.yaxis.set_minor_locator(AutoMinorLocator())

    ax42.set_ylabel('cyclists on street', fontsize=24)
    ax42.tick_params(axis='y', labelsize=16)
    ax42.yaxis.set_minor_locator(AutoMinorLocator())

    if titles:
        ax4.set_title(f'Cyclists and Length on Street in {city}',
                      fontsize=28)
    ax4.set_xlabel('fraction of bike paths', fontsize=24)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())

    ax4_hand = {}
    grp_algo = hf['algorithm']
    for m, d in grp_algo.items():
        print(color)
        print(label)
        bpp = d['bpp'][()]
        """bikeab = [np.trapz(ba[m][:idx], bpp[m][:idx]) for idx in
                  range(len(bpp[m]))]"""
        """cba = [bikeab[idx] / cost[m][idx] for idx in
               range(len(bpp[m]))]"""
        ax1.plot(bpp, d['ba'][()], color=color[m], label=label[m], lw=2.5)
        """ax2.plot(bpp, bikeab, color=color[m], label=label[m])
        ax3.plot(bpp, cba, color=color[m], label=label[m])"""
        space = round(len(bpp) / 20)
        ax4.plot(bpp, d['nos'][()], color=color[m], ms=16, marker='v',
                 markevery=space, label=label[m], lw=2.5)
        ax42.plot(bpp, d['los'][()], color=color[m], ms=16, marker='8',
                  markevery=space, label=label[m], lw=2.5)
        ax4_hand[m] = mlines.Line2D([], [], color=color[m], label=label[m])

    # ax1.plot(bpp_now, ba_now, c='#999999', ms=16, marker='D')
    xmax, ymax = coord_transf(bpp_now, ba_now, xmax=1, ymax=1, xmin=0, ymin=0)
    # ax1.axvline(x=bpp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    # ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    # ax4.plot(bpp_now, nos_now, c='#999999', ms=16, marker='v')
    # ax4.plot(bpp_now, los_now, c='#999999', ms=16, marker='8')
    xmax, ymax = coord_transf(bpp_now, nos_now, xmax=1, ymax=1, xmin=0, ymin=0)
    # ax4.axvline(x=bpp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    # ax4.axhline(y=nos_now, xmax=1, xmin=xmax, c='#999999', ls=':', alpha=0.5)
    # ax4.axhline(y=los_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    l_keys_r = []
    l_keys_b = []
    for mode, l_key in ax4_hand.items():
        if int(mode[0]) == 0:
            l_keys_r.append(l_key)
        else:
            l_keys_b.append(l_key)
    if legends:
        if l_keys_b:
            l_keys = [tuple(l_keys_r)] + [tuple(l_keys_b)]
            l_labels = ['Removing', 'Building']
        else:
            l_keys = l_keys_r
            l_labels = ['Unweighted', 'Street Type Penalty',
                        'Average Trip Length']
    else:
        l_keys = []
        l_labels = []

    if legends:
        ax1.legend(l_keys, l_labels, numpoints=1, loc=4, fontsize=22,
                   # markerscale=2,
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        ax2.legend(l_keys, l_labels, numpoints=1, loc=4, fontsize=22,
                   # markerscale=2,
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        ax3.legend(l_keys, l_labels, numpoints=1, loc=4, fontsize=22,
                   # markerscale=2,
                   handler_map={tuple: HandlerTuple(ndivide=None)})

    l_keys.append(mlines.Line2D([], [], color='k', ms=16, marker='v',
                                label='cyclists'))
    l_labels.append('cyclists')
    l_keys.append(
            mlines.Line2D([], [], color='k', ms=16, marker='8',
                          label='length'))
    l_labels.append('length')
    ax4.legend(l_keys, l_labels, numpoints=1, loc=1, fontsize=22,
               # markerscale=2,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    fig1.savefig(f'{plot_folder}{save}_1.{plot_format}', format=plot_format,
                 bbox_inches='tight')
    fig2.savefig(f'{plot_folder}{save}_2.{plot_format}', format=plot_format,
                 bbox_inches='tight')
    fig3.savefig(f'{plot_folder}{save}_3.{plot_format}', format=plot_format,
                 bbox_inches='tight')
    fig4.savefig(f'{plot_folder}{save}_4.{plot_format}', format=plot_format,
                 bbox_inches='tight')

    plt.close('all')
    # plt.show()


def plot_city(city, save, polygon_folder, input_folder, output_folder,
              comp_folder, plot_folder, modes, cut=True, correct_area=False,
              comp_modes=False, bike_paths=None, plot_evo=False,
              evo_for=None, titles=False, legends=True, plot_format='png'):
    if evo_for is None:
        evo_for = [(False, 1), (False, 3)]

    hf_comp = h5py.File(f'{comp_folder}comp_{save}.hdf5', 'w')
    hf_comp.attrs['city'] = city

    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    Path(comp_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    trip_nbrs = np.load(input_folder + '{}_demand.npy'.format(save),
                        allow_pickle=True)[0]
    trip_nbrs_re = {trip_id: nbr_of_trips for trip_id, nbr_of_trips
                    in trip_nbrs.items() if not trip_id[0] == trip_id[1]}
    trips = sum(trip_nbrs.values())
    trips_re = sum(trip_nbrs_re.values())
    hf_comp.attrs['total trips'] = trips_re
    hf_comp.attrs['total trips (incl round trips)'] = trips
    utrips = len(trip_nbrs.keys())
    utrips_re = len(trip_nbrs_re.keys())
    hf_comp.attrs['unique trips'] = trips_re
    hf_comp.attrs['unique trips (incl round trips)'] = trips

    stations = [station for trip_id, nbr_of_trips in trip_nbrs.items() for
                station in trip_id]
    stations = list(set(stations))
    hf_comp.attrs['nbr of stations'] = len(stations)
    hf_comp['stations'] = stations

    print(f'City: {city}, stations: {len(stations)},'
          f'trips: {trips} (rt excl.: {trips_re}), '
          f'unique trips: {utrips} (rt excl. {utrips_re})')

    nxG = ox.load_graphml(filepath=f'{input_folder}{save}.graphml')
    hf_comp.attrs['nodes'] = len(nxG.nodes)
    hf_comp.attrs['edges'] = len(nxG.edges)
    print(f'City: {city}, nodes: {len(nxG.nodes)}, edges: {len(nxG.edges)}')
    nxG_plot = nxG.to_undirected()
    nxG_calc = nx.Graph(nxG.to_undirected())
    st_ratio = get_street_type_ratio(nxG_calc)
    hf_comp['ratio street type'] = json.dumps(st_ratio)
    sn_ratio = len(stations) / len(nxG_calc.nodes())
    hf_comp['ratio stations nodes'] = sn_ratio

    ns = [75 if n in stations else 0 for n in nxG_plot.nodes()]
    plot_edge_types(nxG_plot, ns, save, plot_folder)

    polygon = get_polygon_from_json(f'{polygon_folder}{save}.json')

    remove_area = None
    if correct_area:
        correct_area_path = f'{polygon_folder}{save}_delete.json'
        if Path(correct_area_path).exists():
            remove_area = get_polygons_from_json(correct_area_path)
        else:
            print('No polygons for area size correction found.')

    area = calc_polygon_area(polygon, remove_area)
    hf_comp.attrs['area'] = area
    print(f'Area {round(area, 1)}')
    sa_ratio = len(stations) / area
    hf_comp['ratio stations area'] = sa_ratio

    plot_used_nodes(save, nxG_plot, trip_nbrs, stations, plot_folder,
                    dpi=150, plot_format=plot_format)

    data_now = calc_current_state(nxG_calc, trip_nbrs, bike_paths)
    # modes = [f'{m[0]:d}{m[1]:}' for m in modes]
    data = {}
    for m in modes:
        hf_in = h5py.File(f'{output_folder}{save}_data_mode_{m[0]:d}{m[1]}'
                          f'.hdf5', 'r')
        data[m] = hf_in['all']

    if cut:
        end = max([get_end(json.loads(d['trdt'][()]), data_now[3], m[0])
                   for m, d in data.items()])
    else:
        end = max([len(d['bpp'][()]) for m, d in data.items()]) - 1
    print('Cut after: ', end)

    bpp_now, ba_now, cost_now, nos_now, los_now = 0, 0, 0, 0, 0

    grp_algo = hf_comp.create_group('algorithm')
    for m, d in data.items():
        m_1 = f'{m[0]:d}{m[1]}'
        sbgrp_algo = grp_algo.create_group(m_1)
        if plot_evo and (m in evo_for):
            evo = True
        else:
            evo = False
        bpp_now, ba_now, cost_now, nos_now, los_now = \
            plot_mode(city=city, save=save, data=d, data_now=data_now,
                      nxG_calc=nxG_calc, nxG_plot=nxG_plot, stations=stations,
                      trip_nbrs=trip_nbrs, mode=m, end=end, evo=evo,
                      hf_group=sbgrp_algo, plot_folder=plot_folder,
                      plot_format=plot_format, titles=titles, legends=legends)

    grp_ps = hf_comp.create_group('p+s')
    grp_ps['bpp'] = bpp_now
    grp_ps['ba'] = ba_now
    grp_ps['cost'] = cost_now
    grp_ps['nos'] = nos_now
    grp_ps['los'] = los_now

    if comp_modes:
        c_norm = ['darkorange', 'darkblue', 'darkviolet']
        # c_norm = ['darkblue']
        c_rev = ['red', 'orangered', 'indianred']
        c = {}
        for m, d in data.items():
            m_1 = '{m[0]:d}{m[1]}'
            if m[0]:
                c[m_1] = c_rev[m[1]]
            else:
                c[m_1] = c_norm[m[1]]
        label = {f'{m[0]:d}{m[1]}': 'Removing' if not m[0]
        else 'Building' for m in modes}
        """label = {'00': 'Unweigthed', '01': 'Street Type Penalty',
                 '02': 'Average Trip Length'}"""
        compare_modes(city=city, save=save, color=c, label=label,
                      comp_folder=comp_folder, plot_folder=plot_folder,
                      plot_format=plot_format, titles=titles, legends=legends)


def compare_distributions(city, base, base_save, graph_folder, data_folder,
                          plot_folder, mode, titles=False, legends=False,
                          figsize=None, plot_format='png', evo=False):
    if figsize is None:
        figsize = (12, 10)
    rev = mode[0]
    minmode = mode[1]
    mode = f'{rev:d}{minmode}'

    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    G = ox.load_graphml(filepath=f'{graph_folder}{base_save}.graphml')

    saves = {'Real Data': base_save, 'HDD': base_save + '_ata',
             'HSD': base_save + '_homog'}
    dist_modes = [k for k, v in saves.items()]

    scale_x = calc_scale(base, dist_modes, saves, data_folder, mode)

    colors = ['#469990', '#800080', '#0000ff']
    color = {m: colors[idx] for idx, m in enumerate(dist_modes)}

    ee = {}
    end = {}
    bpp = {}
    bpp_c = {}
    bpp_now = {}
    ba = {}
    ba_c = {}
    ba_now = {}
    cost = {}
    nos = {}
    nos_c = {}
    nos_now = {}
    los = {}
    los_c = {}
    los_now = {}
    bpp_end = {}
    tfdt_min = {}
    tfdt_max = {}
    tfdt_rat = {}
    trdt_min = {}
    trdt_max = {}
    trdt_rat = {}
    st_ratio = {}
    sn_ratio = {}
    sa_ratio = {}
    ba_improve = {}
    ba_y = {}

    for dist_mode in dist_modes:
        print(dist_mode)
        save = saves[dist_mode]
        data = h5py.File(f'{data_folder}comp_{save}.hdf5', 'r')
        data_algo = data['algorithm']
        data_ps = data['p+s']
        ee[dist_mode] = [(i[0], i[1]) for i in
                         data_algo[mode]['edited edges'][()]]
        end[dist_mode] = data_algo[mode]['end'][()]
        bpp[dist_mode] = data_algo[mode]['bpp'][()]
        bpp_c[dist_mode] = list(reversed(data_algo[mode]['bpp complete'][()]))
        bpp_end[dist_mode] = data_algo[mode]['bpp at end'][()]
        bpp_now[dist_mode] = data_ps['bpp'][()] * bpp_end[dist_mode]
        ba[dist_mode] = data_algo[mode]['ba'][()]
        ba_c[dist_mode] = data_algo[mode]['ba complete'][()]
        ba_y[dist_mode] = data_algo[mode]['ba for comp'][()]
        ba_now[dist_mode] = data_ps['ba'][()]
        ba_improve[dist_mode] = ba_y[dist_mode] - ba_now[dist_mode]
        cost[dist_mode] = data_algo[mode]['cost'][()]
        nos[dist_mode] = data_algo[mode]['nos'][()]
        nos_c[dist_mode] = data_algo[mode]['nos complete'][()]
        nos_now[dist_mode] = data_ps['nos'][()]
        los[dist_mode] = data_algo[mode]['los'][()]
        los_c[dist_mode] = data_algo[mode]['los complete'][()]
        los_now[dist_mode] = data_ps['los'][()]
        tfdt_min[dist_mode] = data_algo[mode]['tfdt min'][()]
        tfdt_max[dist_mode] = data_algo[mode]['tfdt max'][()]
        tfdt_rat[dist_mode] = tfdt_max[dist_mode] / tfdt_min[dist_mode]
        trdt_min[dist_mode] = data_algo[mode]['trdt min'][()]
        trdt_max[dist_mode] = data_algo[mode]['trdt max'][()]
        trdt_rat[dist_mode] = trdt_max[dist_mode] / trdt_min[dist_mode]
        st_ratio[dist_mode] = json.loads(data['ratio street type'][()])
        sn_ratio[dist_mode] = data['ratio stations nodes'][()]
        sa_ratio[dist_mode] = data['ratio stations area'][()]

    fig1, ax1 = plt.subplots(dpi=150, figsize=figsize)
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax1.set_ylabel('bikeability', fontsize=24)
    if titles:
        ax1.set_title('Comparison of Bikeabilities', fontsize=28)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0.0, 1.0)

    fig2, ax2 = plt.subplots(dpi=150, figsize=figsize)
    ax2.set_xlabel('scaled normalised fraction of bike paths', fontsize=24)
    ax2.set_ylabel('bikeability', fontsize=24)
    if titles:
        ax2.set_title('Comparison of Scaled Bikeabilities', fontsize=28)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlim(0.0, 1.0)

    fig3, ax3 = plt.subplots(dpi=150, figsize=figsize)
    ax32 = ax3.twinx()
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax32.set_ylim(0.0, 1.0)

    ax3.set_ylabel('length on street', fontsize=24)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.yaxis.set_minor_locator(AutoMinorLocator())

    ax32.set_ylabel('cylcists on street', fontsize=24)
    ax32.tick_params(axis='y', labelsize=16)
    ax32.yaxis.set_minor_locator(AutoMinorLocator())
    if titles:
        ax3.set_title('Comaprison of Cyclists and Length on Street',
                      fontsize=28)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3_hand = {}

    # cut = max(end.values())
    for dist_mode in dist_modes:
        # bpp_m = [x / bpp_c[dist_mode][cut] for x in bpp_c[dist_mode][:cut]]
        bpp_scaled = [x * scale_x[dist_mode] for x in bpp[dist_mode]]
        ax1.plot(bpp[dist_mode], ba[dist_mode], color=color[dist_mode],
                 label=f'{dist_mode}', lw=2.5)
        ax2.plot(bpp_scaled, ba[dist_mode], color=color[dist_mode],
                 label=f'{dist_mode}', lw=2.5)
        space = round(len(bpp[dist_mode]) / 25)
        ax3.plot(bpp[dist_mode], los[dist_mode], ms=16, marker='8',
                 markevery=space, color=color[dist_mode], lw=2.5)
        ax32.plot(bpp[dist_mode], nos[dist_mode], ms=16, marker='v',
                  markevery=space, color=color[dist_mode], lw=2.5)
        ax3_hand[dist_mode] = mlines.Line2D([], [], color=color[dist_mode])

    if legends:
        ax1.legend(loc='lower right', fontsize=22)
        ax2.legend(loc='lower right', fontsize=22)
        l_keys = [l_key for city, l_key in ax3_hand.items()]
        l_cities = [city for city, l_key in ax3_hand.items()]
    else:
        l_keys = []
        l_cities = []
    l_keys.append(
            mlines.Line2D([], [], color='k', ms=16, marker='v',
                          label='cyclists'))
    l_cities.append('cyclists')
    l_keys.append(
            mlines.Line2D([], [], color='k', ms=16, marker='8',
                          label='length'))
    l_cities.append('length')

    ax3.legend(l_keys, l_cities, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=22)

    save_plot = f'{plot_folder}{base_save}_distr_comp_'
    fig1.savefig(f'{save_plot}ba_unscaled_{rev:d}{minmode}.{plot_format}',
                 format=plot_format, bbox_inches='tight')
    fig2.savefig(f'{save_plot}ba_scaled_{rev:d}{minmode}.{plot_format}' ,
                 format=plot_format, bbox_inches='tight')
    fig3.savefig(f'{save_plot}los_nos_{rev:d}{minmode}.{plot_format}',
                 format=plot_format,  bbox_inches='tight')
    plot_barh(scale_x, color, f'{save_plot}scalefactor_{rev:d}{minmode}',
              plot_format=plot_format)
    plot_barh(trdt_rat, color, f'{save_plot}ratio_max_min_traveled_'
                               f'{rev:d}{minmode}',
              plot_format=plot_format, x_label=r'$trdt_{max} / trdt_{min}$')
    plot_barh(sn_ratio, color, f'{save_plot}ratio_stations_nodes_'
                               f'{rev:d}{minmode}', plot_format=plot_format)
    plot_barh(sa_ratio, color, f'{save_plot}ratio_stations_area_'
                               f'{rev:d}{minmode}',
              plot_format=plot_format, x_label=r'stations / $km^{2}$')
    plot_barh(bpp_end, color, f'{save_plot}fmax_{rev:d}{minmode}',
              plot_format=plot_format)
    plot_barh(ba_improve, color, f'{save_plot}baimprove_{rev:d}{minmode}',
              plot_format=plot_format)
    st = ['primary', 'secondary', 'tertiary', 'residential']
    st_data = {city: list(ratio.values()) for city, ratio in st_ratio.items()}
    st_colors = ['darkblue', 'darkgreen', 'darkcyan', 'darkorange']
    plot_barh_stacked(st_data, st, st_colors, f'{save_plot}ratio_st_'
                                              f'{rev:d}{minmode}',
                      plot_format=plot_format)
    if evo:
        plot_bp_diff(G, ee[dist_modes[0]], ee[dist_modes[1]],
                     bpp_c[dist_modes[0]], bpp_c[dist_modes[1]], 'k', 0,
                     base_save, rev, minmode, plot_folder, plot_format='png')
        plot_bp_diff(G, ee[dist_modes[1]], ee[dist_modes[2]],
                     bpp_c[dist_modes[1]], bpp_c[dist_modes[2]], 'k', 0,
                     base_save + '_ata', rev, minmode, plot_folder,
                     plot_format='png')
    plt.close('all')
    # plt.show()


def compare_cities(cities, saves, mode, color, data_folder, plot_folder,
                   scale_x=None, base_city=None, titles=False, legends=False,
                   figsize=None, plot_format='png'):
    if scale_x is None:
        scale_x = {city: 1 for city in cities}
    if base_city is None:
        base_city = np.random.choice(cities)
    if figsize is None:
        figsize = [12, 9]
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    rev = mode[0]
    minmode = mode[1]
    mode = f'{rev:d}{minmode}'

    bpp = {}
    ba = {}
    cost = {}
    nos = {}
    los = {}
    bpp_end = {}
    tfdt_min = {}
    tfdt_max = {}
    tfdt_rat = {}
    trdt_min = {}
    trdt_max = {}
    trdt_rat = {}
    st_ratio = {}
    sn_ratio = {}
    sa_ratio = {}
    ba_improve = {}
    nos_improve = {}
    los_improve = {}
    ba_y = {}

    for city in cities:
        save = saves[city]
        data = h5py.File(f'{data_folder}comp_{save}.hdf5', 'r')
        data_algo = data['algorithm']
        data_ps = data['p+s']
        bpp[city] = data_algo[mode]['bpp'][()]
        bpp_end[city] = data_algo[mode]['bpp at end'][()]
        ba[city] = data_algo[mode]['ba'][()]
        ba_y[city] = data_algo[mode]['ba for comp'][()]
        ba_improve[city] = ba_y[city] - data_ps['ba'][()]
        cost[city] = data_algo[mode]['cost'][()]
        nos[city] = data_algo[mode]['nos'][()]
        nos_improve[city] = data_ps['nos'][()] - \
                            data_algo[mode]['nos at comp'][()]
        los[city] = data_algo[mode]['los'][()]
        los_improve[city] = data_ps['los'][()] - \
                            data_algo[mode]['los at comp'][()]
        tfdt_min[city] = data_algo[mode]['tfdt min'][()]
        tfdt_max[city] = data_algo[mode]['tfdt max'][()]
        tfdt_rat[city] = tfdt_max[city] / tfdt_min[city]
        trdt_min[city] = data_algo[mode]['trdt min'][()]
        trdt_max[city] = data_algo[mode]['trdt max'][()]
        trdt_rat[city] = trdt_max[city] / trdt_min[city]
        st_ratio[city] = json.loads(data['ratio street type'][()])
        sn_ratio[city] = data['ratio stations nodes'][()]
        sa_ratio[city] = data['ratio stations area'][()]

    fig1, ax1 = plt.subplots(dpi=150, figsize=figsize)
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax1.set_ylabel('bikeability', fontsize=24)
    if titles:
        ax1.set_title('Comparison of Bikeabilities', fontsize=28)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0.0, 1.0)

    fig2, ax2 = plt.subplots(dpi=150, figsize=figsize)
    ax2.set_xlabel('scaled normalised fraction of bike paths', fontsize=24)
    ax2.set_ylabel('bikeability', fontsize=24)
    if titles:
        ax2.set_title('Comparison of Scaled Bikeabilities', fontsize=28)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlim(0.0, 1.0)

    fig3, ax3 = plt.subplots(dpi=150, figsize=figsize)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax3.set_ylabel('integrated bikeability', fontsize=24)
    if titles:
        ax3.set_title('Comparison of Integrated Bikeabilities', fontsize=28)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.set_ylim(0.0, 1.0)
    ax3.set_xlim(0.0, 1.0)

    fig4, ax4 = plt.subplots(dpi=150, figsize=figsize)
    ax42 = ax4.twinx()
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)
    ax42.set_ylim(0.0, 1.0)

    ax4.set_ylabel('length on street', fontsize=24)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.yaxis.set_minor_locator(AutoMinorLocator())

    ax42.set_ylabel('cyclists on street', fontsize=24)
    ax42.tick_params(axis='y', labelsize=16)
    ax42.yaxis.set_minor_locator(AutoMinorLocator())

    if titles:
        ax4.set_title('Comaprison of Trips and Length on Street', fontsize=28)
    ax4.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())

    ax4_hand = {}

    for city in cities:
        bikeab = [np.trapz(ba[city][:idx], bpp[city][:idx]) for idx in
                  range(len(bpp[city]))]
        bpp_scaled = [x * scale_x[city] for x in bpp[city]]
        ax1.plot(bpp[city], ba[city], color=color[city],
                 label=f'{city}', lw=2.5)
        ax2.plot(bpp_scaled, ba[city], color=color[city],
                 label=f'{city}', lw=2.5)
        ax3.plot(bpp[city], bikeab, color=color[city],
                 label=f'{city}', lw=2.5)
        space = round(len(bpp[city]) / 25)
        ax4.plot(bpp[city], los[city], ms=16, marker='8', markevery=space,
                 color=color[city], lw=2.5)
        ax42.plot(bpp[city], nos[city], ms=16, marker='v', markevery=space,
                  color=color[city], lw=2.5)
        ax4_hand[city] = mlines.Line2D([], [], color=color[city])

    if legends:
        l_keys = [l_key for city, l_key in ax4_hand.items()]
        l_cities = [city for city, l_key in ax4_hand.items()]
    else:
        l_keys = []
        l_cities = []
    l_keys.append(mlines.Line2D([], [], color='k', ms=16, marker='v',
                                label='cyclists'))
    l_cities.append('cyclists')
    l_keys.append(mlines.Line2D([], [], color='k', ms=16, marker='8',
                                label='length'))
    l_cities.append('length')

    if legends:
        ax1.legend(loc='lower right', fontsize=22)
        ax2.legend(loc='lower right', fontsize=22)
        ax3.legend(loc='lower right', fontsize=22)
    ax4.legend(l_keys, l_cities, numpoints=1, loc=1,
               handler_map={tuple: HandlerTuple(ndivide=None)},
               fontsize=18).set_zorder(33)

    save_plot = f'{plot_folder}{saves[base_city]}_'
    fig1.savefig(f'{save_plot}ba_unscaled_{rev:d}{minmode}.{plot_format}',
                 format=plot_format, bbox_inches='tight')
    fig2.savefig(f'{save_plot}ba_scaled_{rev:d}{minmode}.{plot_format}',
                 format=plot_format, bbox_inches='tight')
    # fig3.savefig(f'{plot_folder}comparison-3-{rev:d}{minmode}.{plot_format}',
    #             format=plot_format)
    fig4.savefig(f'{save_plot}los_nos_{rev:d}{minmode}.{plot_format}',
                 format=plot_format, bbox_inches='tight')
    scale_x_bar = {k: v for k, v in sorted(scale_x.items(),
                                           key=lambda item: item[1],
                                           reverse=True)}
    plot_barv(scale_x_bar, color, f'{save_plot}scalefactor_{rev:d}{minmode}',
              plot_format=plot_format, ymin=0, ymax=1, xticks=False,
              y_label='scaling factor')
    plot_barh(trdt_rat, color, f'{save_plot}ratio_max_min_traveled_'
                               f'{rev:d}{minmode}',
              plot_format=plot_format, x_label=r'$trdt_{max} / trdt_{min}$')
    plot_barh(sn_ratio, color, f'{save_plot}ratio_stations_nodes_'
                               f'{rev:d}{minmode}', plot_format=plot_format)
    plot_barh(sa_ratio, color, f'{save_plot}ratio_stations_area_'
                               f'{rev:d}{minmode}',
              plot_format=plot_format, x_label=r'stations / $km^{2}$')
    plot_barh(bpp_end, color, f'{save_plot}fmax_{rev:d}{minmode}',
              plot_format=plot_format)
    ymax = ceil(max(ba_improve.values()) * 100) / 100.0
    ba_improve_bar = {k: ba_improve[k] for k in scale_x_bar.keys()}
    plot_barv(ba_improve_bar, color, f'{save_plot}baimprove_{rev:d}{minmode}',
              plot_format=plot_format, xticks=False,
              y_label='improvement over P+S', ymin=0, ymax=ymax)
    ymax = ceil(max(nos_improve.values()) * 100) / 100.0
    nos_improve_bar = {k: nos_improve[k] for k in scale_x_bar.keys()}
    plot_barv(nos_improve_bar, color,
              f'{save_plot}nosimprove_{rev:d}{minmode}',
              plot_format=plot_format, xticks=False,
              y_label='improvement over P+S', ymin=0, ymax=ymax)
    ymax = ceil(max(los_improve.values()) * 100) / 100.0
    los_improve_bar = {k: los_improve[k] for k in scale_x_bar.keys()}
    plot_barv(los_improve_bar, color,
              f'{save_plot}losimprove_{rev:d}{minmode}',
              plot_format=plot_format, xticks=False,
              y_label='improvement over P+S', ymin=0, ymax=ymax)
    st = ['primary', 'secondary', 'tertiary', 'residential']
    st_data = {city: list(ratio.values()) for city, ratio in st_ratio.items()}
    st_colors = ['#000075', '#3cb44b', '#469990', '#f58231']
    plot_barh_stacked(st_data, st, st_colors,
                      f'{save_plot}ratio_st_{rev:d}{minmode}',
                      plot_format=plot_format, legend=legends)

    plt.close('all')
    # plt.show()
