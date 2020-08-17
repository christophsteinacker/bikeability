"""
This module includes all necessary functions for the plotting functionality.
"""
from math import ceil, floor, log10
import h5py
import pyproj
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import to_rgba, LogNorm
import shapely.ops as ops
from bikeability_optimisation.helper.algorithm_helper import \
    get_street_type_cleaned, get_street_length
from bikeability_optimisation.helper.data_helper import get_polygon_from_bbox,\
    get_bbox_from_polygon
from functools import partial
from copy import deepcopy
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable


def magnitude(x):
    return int(floor(log10(x)))


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
    on_all_now = total_dist_now['total length on all']

    dist['all'] = on_all
    dist_now['all'] = total_dist_now['total length on all']

    # On streets w/o bike paths
    on_street = [i['total length on street'] for i in total_dist]
    dist['street'] = [x / on_all[idx] for idx, x in enumerate(on_street)]
    dist_now['street'] = total_dist_now['total length on street'] / on_all_now

    # On primary
    on_primary = [i['total length on primary'] for i in total_dist]
    dist['primary'] = [x / on_all[idx] for idx, x in enumerate(on_primary)]
    dist_now['primary'] = total_dist_now['total length on primary'] / \
                          on_all_now

    # On secondary
    on_secondary = [i['total length on secondary'] for i in total_dist]
    dist['secondary'] = [x / on_all[idx] for idx, x in enumerate(on_secondary)]
    dist_now['secondary'] = total_dist_now['total length on secondary'] / \
                            on_all_now
    # On tertiary
    on_tertiary = [i['total length on tertiary'] for i in total_dist]
    dist['tertiary'] = [x / on_all[idx] for idx, x in enumerate(on_tertiary)]
    dist_now['tertiary'] = total_dist_now['total length on tertiary'] / \
                           on_all_now

    # On residential
    on_residential = [i['total length on residential'] for i in total_dist]
    dist['residential'] = [x / on_all[idx] for idx, x in
                           enumerate(on_residential)]
    dist_now['residential'] = total_dist_now['total length on residential'] / \
                              on_all_now

    # On bike paths
    on_bike = [i['total length on bike paths'] for i in total_dist]
    dist['bike paths'] = [x / on_all[idx] for idx, x in enumerate(on_bike)]
    dist_now['bike paths'] = total_dist_now['total length on bike paths'] /\
                             on_all_now

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


def calc_polygon_area(polygon, remove=None, unit='sqkm'):
    if (not isinstance(remove, list)) ^ (remove is None):
        remove = [remove]
    geom_area = ops.transform(
            partial(
                    pyproj.transform,
                    pyproj.Proj(init='EPSG:4326'),
                    pyproj.Proj(
                            proj='aea',
                            lat_1=polygon.bounds[1],
                            lat_2=polygon.bounds[3])),
            polygon)
    remove_area = 0
    if remove is not None:
        for p in remove:
            a_r = ops.transform(
                partial(
                        pyproj.transform,
                        pyproj.Proj('EPSG:4326'),
                        pyproj.Proj(
                                proj='aea',
                                lat_1=p.bounds[1],
                                lat_2=p.bounds[3])),
                p)
            remove_area += a_r.area

    if unit == 'sqkm':
        return (geom_area.area - remove_area) / 1000000
    if unit == 'sqm':
        return geom_area.area - remove_area


def calc_scale(base_city, cities, saves, comp_folder, mode):
    blp = {}
    ba = {}

    if isinstance(mode, tuple):
        mode = '{:d}{}'.format(mode[0], mode[1])

    for city in cities:
        save = saves[city]
        data = h5py.File(comp_folder+'comp_{}.hdf5'.format(save), 'r')
        blp[city] = data['algorithm'][mode]['bpp'][()]
        ba[city] = data['algorithm'][mode]['ba'][()]

    blp_base = blp[base_city]
    ba_base = ba[base_city]

    cities_comp = deepcopy(cities)
    cities_comp.remove(base_city)

    min_idx = {}
    for city in cities_comp:
        m_idx = []
        for idx, x in enumerate(ba[city]):
            # Create list where each value corresponds to the index of the
            # item from the base city ba list closest to the ba value of the
            # comparing city at the current index.
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


def get_edge_color(G, edges, attr, color):
    nx.set_edge_attributes(G, False, attr)
    for edge in edges:
        G[edge[0]][edge[1]][0][attr] = True
        G[edge[1]][edge[0]][0][attr] = True
    return [color if data[attr] else '#999999' for u, v, data in
            G.edges(keys=False, data=True)]


def get_edge_color_st(G, colors):
    return [colors[get_street_type_cleaned(G, e, multi=True)]
            for e in G.edges()]


def plot_barh(data, colors, save, figsize=None, plot_format='png',
              x_label='', title='', dpi=150):
    if figsize is None:
        figsize = [16, 9]
    keys = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    y_pos = np.arange(len(keys))
    max_value = max(values)
    for idx, key in enumerate(keys):
        color = to_rgba(colors[key])
        ax.barh(y_pos[idx], values[idx], color=color, align='center')
        x = values[idx] / 2
        y = y_pos[idx]
        if values[idx] > 0.05 * max_value:
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.text(x, y, '{:3.2f}'.format(values[idx]), ha='center',
                    va='center', color=text_color, fontsize=16)
        else:
            ax.text(2*values[idx], y, '{:3.2f}'.format(values[idx]),
                    ha='center', va='center', color='darkgrey', fontsize=16)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(x_label)
    ax.set_title(title)

    plt.savefig(save + '.{}'.format(plot_format), format=plot_format,
                bbox_inches='tight')


def plot_barh_stacked(data, stacks, colors, save, figsize=None,
                      plot_format='png', title='', dpi=150):
    if figsize is None:
        figsize = [16, 9]

    labels = list(data.keys())
    values = np.array(list(data.values()))
    values_cum = values.cumsum(axis=1)
    colors = [to_rgba(c) for c in colors]

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, max(np.sum(values, axis=1)))

    for i, (colname, color) in enumerate(zip(stacks, colors)):
        widths = values[:, i]
        starts = values_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname,
                color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c != 0.0:
                ax.text(x, y, '{:3.2f}'.format(c), ha='center', va='center',
                        color=text_color)
    ax.legend(ncol=len(stacks), bbox_to_anchor=(0, 1), loc='lower left',
              fontsize='small')
    ax.set_title(title)
    plt.savefig(save + '.{}'.format(plot_format), format=plot_format,
                bbox_inches='tight')


def plot_barv(data, colors, save, figsize=None, plot_format='png', y_label='',
              title='', ymin=-0.1, ymax=0.7, xticks=True, dpi=150):
    if figsize is None:
        figsize = [10, 10]
    keys = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.set_ylim(ymin, ymax)
    x_pos = np.arange(len(keys))
    for idx, key in enumerate(keys):
        color = to_rgba(colors[key])
        ax.bar(x_pos[idx], values[idx], color=color, align='center')
        y = values[idx] / 2
        x = x_pos[idx]
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.text(x, y, '{:3.2f}'.format(values[idx]), ha='center', va='center',
                color=text_color)
    if xticks:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(keys)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(y_label)
    ax.set_xlabel(' ', fontsize=12)
    ax.set_title(title)

    plt.savefig(save + '.{}'.format(plot_format), format=plot_format,
                bbox_inches='tight')


def plot_barv_stacked(labels, data, colors, title='', ylabel='', save='',
                      width=0.8, figsize=None, dpi=150, plot_format='png'):
    if figsize is None:
        figsize = [10, 12]

    stacks = list(data.keys())
    values = list(data.values())
    x_pos = np.arange((len(labels)))

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.set_ylim(0.0, 1.0)
    bottom = np.zeros(len(values[0]))
    for idx in range(len(stacks)):
        ax.bar(x_pos, values[idx], width, label=stacks[idx],
               bottom=bottom, color=colors[stacks[idx]])
        for v_idx, v in enumerate(values[idx]):
            if v > 0.05:
                color = to_rgba(colors[stacks[idx]])
                y = bottom[v_idx] + v / 2
                x = x_pos[v_idx]
                r, g, b, _ = color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                ax.text(x, y, '{:3.2f}'.format(v), ha='center',
                        va='center', color=text_color, fontsize=16)
        bottom = [sum(x) for x in zip(bottom, values[idx])]
        print(stacks[idx], values[idx])

    ax.set_ylabel(ylabel, fontsize=24)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_title(title, fontsize=24)

    plt.savefig(save+'.{}'.format(plot_format), format=plot_format,
                bbox_inches='tight')


def plot_histogram(data, save_path, bins=None, cumulative=False, xlabel='',
                   ylabel='', xlim=None,  plot_format='png', dpi=150):
    max_d = max(data)
    min_d = min(data)
    r = magnitude(max_d)

    fig1, ax1 = plt.subplots(figsize=(12, 10), dpi=dpi)
    ax1.set_xlim(left=0.0, right=round(max_d + 0.1 * max_d, -(r - 1)))
    if xlim is not None:
        ax1.set_xlim(left=xlim[0], right=xlim[1])
    if bins is None:
        if max_d == min_d:
            bins = 1
        elif (max_d - min_d) <= 200:
            bins = 50
        else:
            bins = ceil((max_d - min_d) / (10 ** (r - 2)))
    ax1.hist(data, bins=bins, align='mid', cumulative=cumulative)
    ax1.set_xlabel(xlabel, fontsize=24)
    ax1.set_ylabel(ylabel, fontsize=24)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    fig1.savefig('{}.{}'.format(save_path, plot_format), format=plot_format,
                 bbox_inches='tight')


def plot_matrix(city, df, plot_folder, save, cmap=None, figsize=None,
                dpi=150, plot_format='png'):
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis_r')
    if figsize is None:
        figsize = [10, 10]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    c = ax.pcolor(df, cmap=cmap, norm=LogNorm(vmin=df.min().min(),
                                                     vmax=df.max().max()))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size="5%", pad=0.05)
    plt.colorbar(c, cax=cax, orientation='horizontal')
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False,
                   labelleft=False)
    cax.set_xlabel('Total Trips', fontsize=18)

    # ax.set_title('Trips in {}'.format(city), fontsize='x-large')
    fig.savefig('{}{}_od_matrix.{}'.format(plot_folder, save, plot_format),
                format=plot_format, bbox_inches='tight')
