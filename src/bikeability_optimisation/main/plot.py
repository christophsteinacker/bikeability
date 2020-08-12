from math import ceil
import h5py
import json
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from matplotlib.colors import rgb2hex, to_rgba
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from pathlib import Path
import matplotlib.lines as mlines
import osmnx as ox

from bikeability_optimisation.helper.plot_helper import *


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
    fig.suptitle('Edge load in {}'.format(city), fontsize='x-large')
    plt.savefig(plot_folder+'{0:s}-load-{1:d}{2:}.{3:s}'
                .format(save, rev, minmode, plot_format), format=plot_format)
    plt.close(fig)


def plot_used_nodes(city, save, G, trip_nbrs, stations, plot_folder,
                    figsize=(12, 12), dpi=150, plot_format='png'):
    print('Plotting used nodes.')

    nodes = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if (s_node, e_node) in trip_nbrs:
                nodes[s_node] += trip_nbrs[(s_node, e_node)]
                nodes[e_node] += trip_nbrs[(s_node, e_node)]

    trip_count = sum(trip_nbrs.values())
    station_count = len(stations)

    max_n = max(nodes.values())
    print('Maximal station usage: {}'.format(max_n))

    n_rel = {key: value for key, value in nodes.items()}
    ns = [75 if n in stations else 0 for n in G.nodes()]

    for n in G.nodes():
        if n not in stations:
            n_rel[n] = max_n + 1
    min_n = min(n_rel.values())
    print('Minimal station usage: {}'.format(min_n))

    fig1, ax1 = plt.subplots(dpi=dpi)
    if max_n == min_n:
        bins = 1
    else:
        bins = ceil((max_n - min_n) / 100)
    ax1.hist([value for key, value in n_rel.items() if value != max_n+1],
             bins=bins)
    ax1.set_xlabel('# of Trips')
    ax1.set_ylabel('# of Stations', fontsize=12)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    fig1.suptitle('Usage distribution of Stations')

    fig1.savefig(plot_folder + '{:}_stations_usage_distribution.{}'
                 .format(save, plot_format), format=plot_format)


    cmap_name = 'cool'
    cmap = plt.cm.get_cmap(cmap_name)
    cmap = ['#999999'] + \
           [rgb2hex(cmap(n)) for n in reversed(np.linspace(1, 0, max_n,
                                                           endpoint=True))] \
           + ['#ffffff']
    color_n = [cmap[v] for k, v in n_rel.items()]

    fig2, ax2 = plt.subplots(dpi=dpi, figsize=figsize)
    ox.plot_graph(G, ax=ax2, bgcolor='#ffffff',
                  edge_linewidth=1.5, node_color=color_n,
                  node_size=ns, node_zorder=3, show=False, close=False)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap_name),
                               norm=plt.Normalize(vmin=0, vmax=max_n))
    sm._A = []
    cbaxes = fig2.add_axes([0.1, 0.05, 0.8, 0.03])

    if min_n <= 0.1 * max_n:
        r = magnitude(max_n)
        cbar = fig2.colorbar(sm, orientation='horizontal', cax=cbaxes,
                             ticks=[0, round(max_n / 2), max_n])
        cbar.ax.set_xticklabels([0, int(round(max_n / 2, -(r-2))),
                                 round(max_n, -(r-1))])
    else:
        max_r = magnitude(max_n)
        min_r = magnitude(min_n)
        cbar = fig2.colorbar(sm, orientation='horizontal', cax=cbaxes,
                             ticks=[0, min_n, round(max_n / 2), max_n])
        cbar.ax.set_xticklabels([0, round(min_n, -(min_r-1)),
                                 int(round(max_n / 2, -(max_r-2))),
                                 round(max_n, -(max_r-1))])

    cbar.ax.tick_params(axis='x', labelsize=16)
    cbar.ax.set_xlabel('Total Number of Trips', fontsize=18)

    #fig2.suptitle('{}, Stations: {}, Trips: {}'
    #              .format(city.capitalize(), station_count, trip_count),
    #              fontsize=24)
    fig2.savefig(plot_folder + '{:}_stations_used.{}'
                 .format(save, plot_format), format=plot_format,
                 bbox_inches='tight')

    plt.close('all')
    # plt.show()


def plot_edges(G, edge_color, node_size, save_path, plot_format='png',
               fig_size=(6, 6), dpi=150, fig_title=''):

    fig, ax = ox.plot_graph(G, bgcolor='#ffffff', node_size=node_size,
                            node_zorder=3, node_color='C0',
                            edge_color=edge_color, figsize=fig_size,
                            dpi=dpi, show=False, close=False)
    fig.suptitle(fig_title, fontsize=24)
    fig.savefig(save_path, format=plot_format, bbox_inches='tight')
    plt.close(fig)


def plot_bp_evo(save, G, edited_edges, bike_path_perc, cut, ps,
                node_size, rev, minmode, plot_folder, plot_format='png'):
    print('Plotting bike path evolution.')
    plot_folder_evo = plot_folder + 'evolution/'
    Path(plot_folder_evo).mkdir(parents=True, exist_ok=True)
    edited_color = '#0000FF'
    nx.set_edge_attributes(G, False, 'bike path')
    if rev:
        ee = edited_edges
        blp = bike_path_perc
    else:
        ee = list(reversed(edited_edges))
        blp = list(reversed(bike_path_perc))

    plots = np.linspace(0, 1, 101)
    for i, j in enumerate(plots):
        idx = next(x for x, val in enumerate(blp) if val >= j)
        ee_evo = ee[:idx]
        for edge in ee_evo:
            G[edge[0]][edge[1]][0]['bike path'] = True
            G[edge[1]][edge[0]][0]['bike path'] = True
        ec_evo = get_edge_color(G, ee_evo, 'bike path', edited_color)
        save_path = '{:}{:}-edited-mode-{:d}{:}-{:}.{:}'.format(
                plot_folder_evo, save, rev, minmode, i, plot_format)
        plot_edges(G, ec_evo, node_size, save_path, plot_format=plot_format,
                   fig_size=(6, 6), dpi=150)

    nx.set_edge_attributes(G, False, 'bike path')
    ee_cut = ee[:cut]
    ec_cut = get_edge_color(G, ee_cut, 'bike path', edited_color)
    save_path = '{:}{:}-edited-mode-{:d}{:}-{:}.{:}'.format(
            plot_folder_evo, save, rev, minmode, 'cut', plot_format)
    plot_edges(G, ec_cut, node_size, save_path, plot_format=plot_format,
               fig_size=(6, 6), dpi=150)

    nx.set_edge_attributes(G, False, 'bike path')
    ee_ps = ee[:ps]
    ec_ps = get_edge_color(G, ee_ps, 'bike path', edited_color)
    save_path = '{:}{:}-edited-mode-{:d}{:}-{:}.{:}'.format(
            plot_folder_evo, save, rev, minmode, 'ps', plot_format)
    plot_edges(G, ec_ps, node_size, save_path, plot_format=plot_format,
               fig_size=(6, 6), dpi=150)


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

    idx = next(x for x, val in enumerate(bpp_algo) if abs(val - bpp_cs) <=
               0.001)
    idx = min(range(len(bpp_algo)), key=lambda i: abs(bpp_algo[i]-bpp_cs))

    print('Difference in BPP between p+s and algo: {}'
          .format(abs(bpp_cs - bpp_algo[idx])))

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
    c = [cmap(n) for n in reversed(np.linspace(1, 0, 10, endpoint=True))]

    color_algo = 'midnightblue'
    color_cs = 'darkorange'
    color_both = 'crimson'
    color_unused = '#999999'

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
            elif data['algo'] and not data['cs']:
                ec.append(color_algo)
                ee_algo_only.append((u, v, k))
            elif not data['algo'] and data['cs']:
                ec.append(color_cs)
                ee_cs_only.append((u, v, k))
            else:
                ec.append(color_unused)
                unused.append((u, v, k))
    else:
        print('You have to choose between algo, p+s and diff.')

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ox.plot_graph(G, bgcolor='#ffffff', ax=ax,
                  node_size=node_size, node_color='C0', node_zorder=3,
                  edge_linewidth=1.5, edge_color=ec,
                  show=False, close=False)
    if mode == 'algo':
        leg = [Line2D([0], [0], color=color_algo, lw=4),
               Line2D([0], [0], color=color_unused, lw=4)]
        #ax.legend(leg, ['Algorithm', 'None'],
        #          bbox_to_anchor=(0, -0.05, 1, 1), loc=3,
        #          ncol=2, mode="expand", borderaxespad=0., fontsize=12)
        #ax.set_title('Algorithm', fontsize=24)
    elif mode == 'p+s':
        leg = [Line2D([0], [0], color=color_cs, lw=4),
               Line2D([0], [0], color=color_unused, lw=4)]
        #ax.legend(leg, ['Primary + Secondary', 'None'],
        #          bbox_to_anchor=(0, -0.05, 1, 1), loc=3,
        #          ncol=2, mode="expand", borderaxespad=0., fontsize=12)
        #ax.set_title('Primary/Secondary', fontsize=24)
    elif mode == 'diff':
        leg = [Line2D([0], [0], color=color_both, lw=4),
               Line2D([0], [0], color=color_algo, lw=4),
               Line2D([0], [0], color=color_cs, lw=4),
               Line2D([0], [0], color=color_unused, lw=4)]
        # ax.legend(leg, ['Both', 'Algorithm', 'Primary + Secondary', 'None'],
        #          bbox_to_anchor=(0, -0.05, 1, 1), loc=3,
        #          ncol=4, mode="expand", borderaxespad=0., fontsize=12)
        #ax.set_title('Comparison', fontsize=24)
    plt.savefig(plot_folder+'{0:s}-bp-build-{1:d}{2:d}_{3:s}.{4:s}'
                .format(save, rev, minmode, mode, plot_format),
                format=plot_format, bbox_inches='tight')
    plt.close(fig)
    if mode == 'algo':
        plot_load(city, save, G, ee_algo_cut, trip_nbrs, node_size, rev,
                  minmode, plot_folder, plot_format=plot_format)


def plot_bp_diff(G, ee_1, ee_2, bpp_1, bpp_2, bpp_comp, node_color,
                 node_size, save, rev, minmode, plot_folder,
                 plot_format='png', figsize=None, dpi=150):
    if figsize is None:
        figsize = [10, 10]
    nx.set_edge_attributes(G, False, '1')
    nx.set_edge_attributes(G, False, '2')

    if not rev:
        ee_1 = list(reversed(ee_1))
        bpp_1 = list(reversed(bpp_1))
        ee_2 = list(reversed(ee_2))
        bpp_2 = list(reversed(bpp_2))

    idx_1 = min(range(len(bpp_1)), key=lambda i: abs(bpp_1[i] - bpp_comp))
    ee_1_cut = ee_1[:idx_1]
    idx_2 = min(range(len(bpp_2)), key=lambda i: abs(bpp_2[i] - bpp_comp))
    ee_2_cut = ee_2[:idx_2]

    for edge in ee_1_cut:
        G[edge[0]][edge[1]][0]['1'] = True
        G[edge[1]][edge[0]][0]['1'] = True
    for edge in ee_2_cut:
        G[edge[0]][edge[1]][0]['2'] = True
        G[edge[1]][edge[0]][0]['2'] = True

    ec = []
    unused = []
    ee_1_only = []
    ee_2_only = []
    ee_both = []

    color_algo = 'midnightblue'
    color_cs = 'darkorange'
    color_both = 'crimson'
    color_unused = '#999999'

    for u, v, k, data in G.edges(keys=True, data=True):
        if data['1'] and data['2']:
            ec.append(color_both)
            ee_both.append((u, v, k))
        elif data['1'] and not data['2']:
            ec.append(color_algo)
            ee_1_only.append((u, v, k))
        elif not data['1'] and data['2']:
            ec.append(color_cs)
            ee_2_only.append((u, v, k))
        else:
            ec.append(color_unused)
            unused.append((u, v, k))

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ox.plot_graph(G, bgcolor='#ffffff', ax=ax,
                  node_size=node_size, node_color='C0', node_zorder=3,
                  edge_linewidth=1.5, edge_color=ec,
                  show=False, close=False)
    save_plot = '{}{}_distr_comp_'.format(plot_folder, save)
    plt.savefig('{}.{}'.format(save_plot, plot_format),
                format=plot_format, bbox_inches='tight')
    plt.close(fig)


def plot_barh(data, colors, save, figsize=None, plot_format='png',
              x_label='', title=''):
    if figsize is None:
        figsize = [16, 9]
    keys = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(dpi=150, figsize=figsize)
    y_pos = np.arange(len(keys))
    for idx, key in enumerate(keys):
        color = to_rgba(colors[key])
        ax.barh(y_pos[idx], values[idx], color=color, align='center')
        x = values[idx] / 2
        y = y_pos[idx]
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.text(x, y, '{:3.2f}'.format(values[idx]), ha='center', va='center',
                color=text_color, fontsize=16)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(x_label)
    ax.set_title(title)

    plt.savefig(save + '.{}'.format(plot_format), format=plot_format,
                bbox_inches='tight')


def plot_barh_stacked(data, stacks, colors, save, figsize=None,
                      plot_format='png', title=''):
    if figsize is None:
        figsize = [16, 9]

    labels = list(data.keys())
    values = np.array(list(data.values()))
    values_cum = values.cumsum(axis=1)
    colors = [to_rgba(c) for c in colors]

    fig, ax = plt.subplots(dpi=150, figsize=figsize)
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
              title=''):
    if figsize is None:
        figsize = [10, 10]
    keys = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(dpi=150, figsize=figsize)
    ax.set_ylim(-0.1, 0.7)
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
        blp = bike_path_perc
    else:
        blp = list(reversed(bike_path_perc))
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

    blp_cut = [i / blp[end] for i in blp[:end]]
    blp_now = bike_path_perc_now / blp[end]

    blp_x = min(blp_cut, key=lambda x: abs(x - blp_now))
    blp_idx = next(x for x, val in enumerate(blp_cut) if val == blp_x)
    ba_y = ba[blp_idx]
    ba_improve = ba_y - ba_now

    cost_y = min(cost[:end], key=lambda x: abs(x - cost_now))
    cost_idx = next(x for x, val in enumerate(cost[:end]) if val == cost_y)
    cost_x = blp_cut[cost_idx]

    nos_y = min(nos[:end], key=lambda x: abs(x - nos_now))
    nos_idx = next(x for x, val in enumerate(nos[:end]) if val == nos_y)
    nos_x = blp_cut[nos_idx]
    nos_y = nos[blp_idx]
    nos_improve = nos_now - nos[blp_idx]

    los_y = min(los[:end], key=lambda x: abs(x - los_now))
    los_idx = next(x for x, val in enumerate(los[:end]) if val == los_y)
    los_x = blp_cut[los_idx]
    los_y = los[blp_idx]
    los_improve = los_now - los[blp_idx]

    cut = next(x for x, val in enumerate(ba) if val >= 1)
    total_cost, cost_now = sum_total_cost(cost, cost_now, rev)

    cost_now = cost_now / total_cost[end]
    # gcbc_size_normed = [i / max(gcbc_size) for i in reversed(gcbc_size)]

    ns = [50 if n in stations else 0 for n in nxG_plot.nodes()]
    cmap_name = 'viridis'
    cmap = plt.cm.get_cmap(cmap_name)
    c = [cmap(n) for n in np.linspace(1, 0, 9, endpoint=True)]

    max_bpp = max(blp[end], blp[cut])

    print('Mode: {:d}{:}, ba=1 after: {:d}, bpp at ba=1: {:3.2f}, '
          'bpp big roads: {:3.2f}, edges: {:}, max bpp: {:3.2f}'
          .format(rev, minmode, end, blp[end], blp_now,
                  len(edited_edges_nx), max_bpp))

    # Plotting
    fig1, ax1 = plt.subplots(dpi=150, figsize=(12, 10))
    ax12 = ax1.twinx()
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax12.set_ylim(0.0, 1.0)

    c_ba = 'C0'

    ax1.plot(blp_cut, ba[:end], c=c_ba, label='bikeability')
    ax1.plot(blp_now, ba_now, c=c_ba, marker='D')
    ax1.tick_params(axis='x', labelsize=16)
    xmax, ymax = coord_transf(blp_now, max([ba_y, ba_now]),
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_ba, ls='--', alpha=0.5)
    ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    ax1.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)

    ax1.set_ylabel('bikeability', fontsize=24, color=c_ba)
    ax1.tick_params(axis='y', labelsize=16, labelcolor=c_ba)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    c_cost = 'C8'

    ax12.plot(blp_cut, [x / total_cost[end] for x in total_cost[:end]],
              c=c_cost, label='total cost')
    ax12.plot(blp_now, cost_now, c=c_cost, marker='s')
    xmin, ymax = coord_transf(min(blp_now, cost_x), cost_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_cost, ls='--', alpha=0.5)
    ax1.axhline(y=cost_now, xmax=1, xmin=xmin, c=c_cost, ls='--', alpha=0.5)
    ax1.axhline(y=cost_y, xmax=xmax, xmin=0, c=c_cost, ls='--', alpha=0.5)
    # ax1.axvline(x=blp[cut] / blp[end], c='#999999', ls='--', alpha=0.7, lw=3)

    ax12.set_ylabel('cost', fontsize=24, color=c_cost)
    ax12.tick_params(axis='y', labelsize=16, labelcolor=c_cost)
    ax12.yaxis.set_minor_locator(AutoMinorLocator())

    ax1.set_xlabel('normalised fraction of bike paths', fontsize=24)
    if titles:
        ax1.set_title('Bikeability and Cost in {}'.format(city), fontsize=24)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='x', labelsize=16)
    ax1.grid(False)
    ax12.grid(False)

    handles = ax1.get_legend_handles_labels()[0]
    handles.append(ax12.get_legend_handles_labels()[0][0])
    if legends:
        ax1.legend(handles=handles, loc='lower right', fontsize=18)

    fig1.savefig(plot_folder + '{:}_ba_tc_mode_{:d}{:}.{}'
                 .format(save, rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')

    ax1ins = zoomed_inset_axes(ax1, 3.5, loc=1)
    x1, x2, y1, y2 = round(blp_now - 0.05, 2), round(blp_now + 0.05, 2), \
                     round(ba_now - 0.03, 2), min(round(ba_y + 0.03, 2), 1)
    ax1ins.plot(blp_cut, ba[:end])
    ax1ins.plot(blp_now, ba_now, c=c_ba, marker='D')
    xmax, ymax = coord_transf(blp_now, max([ba_y, ba_now]),
                              xmin=x1, xmax=x2, ymin=y1, ymax=y2)
    ax1ins.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)

    ax1ins.set_xlim(x1, x2)
    ax1ins.set_ylim(y1, y2)
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")
    ax1ins.tick_params(axis='y', labelsize=16, labelcolor=c_ba)
    ax1ins.tick_params(axis='x', labelsize=16)
    ax1ins.yaxis.set_minor_locator(AutoMinorLocator())
    ax1ins.xaxis.set_minor_locator(AutoMinorLocator())
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")

    fig1.savefig(plot_folder + '{:}_ba_tc_zoom_mode_{:d}{:}.{}'
                 .format(save, rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')

    fig2, ax2 = plt.subplots(dpi=150, figsize=(12, 10))
    ax22 = ax2.twinx()
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax22.set_ylim(0.0, 1.0)

    c_nos = 'C1'
    c_los = 'm'

    p1, = ax2.plot(blp_cut, los[:end], label='length', c=c_los)
    ax2.plot(blp_now, los_now, c=c_los, marker='8')
    xmax, ymax = coord_transf(max(blp_now, los_x), los_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax2.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5)
    ax2.axhline(y=los_now, xmax=xmax, xmin=0, c=c_los, ls='--', alpha=0.5)
    # ax2.axvline(x=los_x, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5)
    ax2.axhline(y=los_y, xmax=xmax, xmin=0, c=c_los, ls='--', alpha=0.5)

    ax2.set_ylabel('length of trips', fontsize=24, color=c_los)
    ax2.tick_params(axis='y', labelsize=16, labelcolor=c_los)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    p2, = ax22.plot(blp_cut, nos[:end], label='trips', c=c_nos)
    ax22.plot(blp_now, nos_now, c=c_nos, marker='v')
    xmin, ymax = coord_transf(max(blp_now, nos_x), nos_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax22.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5)
    ax22.axhline(y=nos_now, xmax=1, xmin=xmin, c=c_nos, ls='--', alpha=0.5)
    # ax22.axvline(x=nos_x, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5)
    ax22.axhline(y=nos_y, xmax=1, xmin=xmin, c=c_nos, ls='--', alpha=0.5)

    ax22.set_ylabel('number of trips', fontsize=24, color=c_nos)
    ax22.tick_params(axis='y', labelsize=16, labelcolor=c_nos)
    ax22.yaxis.set_minor_locator(AutoMinorLocator())

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='x', labelsize=16)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=24)
    if titles:
        ax2.set_title('Number of Trips and Length on Street in {}'
                      .format(city), fontsize=24)
    if legends:
        ax2.legend([p1, p2], [l.get_label() for l in [p1, p2]], fontsize=18)
    fig2.savefig(plot_folder + '{:}_trips_on_street_mode_{:d}{:}.{}'
                 .format(save, rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')

    fig3, ax3 = plt.subplots(dpi=150, figsize=(12, 10))
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.15)

    c_st = {'primary': 'darkblue', 'secondary': 'darkgreen',
            'tertiary': 'darkcyan', 'residential': 'darkorange',
            'bike paths': 'gold'}
    m_st = {'primary': 'p', 'secondary': 'p', 'tertiary': 'p',
            'residential': 'p', 'bike paths': 'P'}

    for st, len_on_st in trdt_st_now.items():
        xmax, ymax = coord_transf(blp_now, len_on_st, ymax=1.15)
        ax3.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_st[st], ls='--',
                    alpha=0.5)
        ax3.axhline(y=len_on_st, xmax=xmax, xmin=0, c=c_st[st], ls='--',
                    alpha=0.5)

    for st, len_on_st in trdt_st.items():
        ax3.plot(blp_cut, len_on_st[:end], c=c_st[st], label=st)
    for st, len_on_st in trdt_st.items():
        ax3.plot(blp_now, trdt_now[st], c=c_st[st], marker=m_st[st])

    ax3.set_xlabel('% of bike paths by length', fontsize=18)
    ax3.set_ylabel('length', fontsize=18)
    ax3.set_title('Length on Street in {}'.format(city), fontsize=14)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.legend()
    fig3.savefig(plot_folder + '{:}_len_on_street_mode_{:d}{:}.{}'
                 .format(save, rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')

    fig4, ax4 = plt.subplots(dpi=150, figsize=(12, 10))
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)

    """for st, len_by_st_now in bl_st_now.items():
        xmax, ymax = coord_transf(blp_now, len_by_st_now)
        ax4.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_st[st], ls='--',
                    alpha=0.5)
        ax4.axhline(y=len_by_st_now, xmax=xmax, xmin=0, c=c_st[st], ls='--',
                    alpha=0.5)"""
    for st, len_by_st in bl_st.items():
        if len_by_st[end] == 0:
            len_norm = 1
        else:
            len_norm = len_by_st[end]
        ax4.plot(blp_cut, [x / len_norm for x in len_by_st[:end]],
                 c=c_st[st], label='{}'.format(st))
    """for st, len_by_st in bl_st.items():
        ax4.plot(blp_now, bl_st_now[st], marker=m_st[st],  c=c_st[st])"""

    ax4.axvline(x=blp[cut] / blp[end], c='#999999', ls='--', alpha=0.7, lw=3)

    ax4.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax4.set_ylabel('length', fontsize=24)
    if titles:
        ax4.set_title('Length of Bike Paths along Streets in {}'.format(city),
                      fontsize=14)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    if legends:
        ax4.legend()
    fig4.savefig(plot_folder + '{:}_len_bl_mode_{:d}{:}.{}'
                 .format(save, rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')

    comp_st_driven = {st: [len_on_st[blp_idx], trdt_st_now[st]]
                      for st, len_on_st in trdt_st.items()}
    plot_barv_stacked(['Algorithm', 'Primary/Secondary'], comp_st_driven, c_st,
                      title='', save=plot_folder+'{:}_comp_st_driven_{:d}{:}'
                      .format(save, rev, minmode), plot_format=plot_format)

    diff_core = {}
    diff_core['Bikeability'] = ba_improve
    diff_core['# Trips'] = nos_improve
    diff_core['Length'] = los_improve
    c_diff_core = {'Bikeability': c_ba, '# Trips': c_nos,
                   'Length': c_los}
    plot_barv(diff_core, c_diff_core,
              plot_folder+'{:}_improvement_core_{:d}{:}'
              .format(save, rev, minmode), plot_format=plot_format,
              figsize=[10, 12], y_label='',
              title='Change compared to primary and secondary only')

    diff_st = {st: trdt_st_now[st] - len_on_st[blp_idx]
               for st, len_on_st in trdt_st.items() if st != 'bike paths'}
    diff_st['bike paths'] = trdt_st['bike paths'][blp_idx] - \
                            trdt_st_now['bike paths']
    c_diff_st = {st: c for st, c in c_st.items()}
    plot_barv(diff_st, c_diff_st, plot_folder+'{:}_improvement_st_{:d}{:}'
              .format(save, rev, minmode), plot_format=plot_format,
              figsize=[10, 12], y_label='',
              title='Change compared to primary and secondary only')
    diff = deepcopy(diff_core)
    diff.update(diff_st)
    c_diff = deepcopy(c_diff_core)
    c_diff.update(c_diff_st)
    plot_barv(diff, c_diff, plot_folder + '{:}_improvement_{:d}{:}'
              .format(save, rev, minmode), plot_format=plot_format,
              figsize=[10, 11], y_label='',
              title='Change compared to primary and secondary only')


    plot_used_nodes(city=city, save=save, G=nxG_plot, trip_nbrs=trip_nbrs,
                    stations=stations, plot_folder=plot_folder,
                    plot_format=plot_format)
    for bp_mode in ['algo', 'p+s', 'diff']:
        plot_bp_comparison(city=city, save=save, G=nxG_plot,
                           ee_algo=edited_edges_nx, ee_cs=bike_paths_now,
                           bpp_algo=bike_path_perc, bpp_cs=bike_path_perc_now,
                           node_size=ns, trip_nbrs=trip_nbrs, rev=rev,
                           minmode=minmode, plot_folder=plot_folder,
                           plot_format=plot_format, mode=bp_mode)

    if evo:
        plot_bp_evo(save=save, G=nxG_plot, edited_edges=edited_edges_nx,
                    bike_path_perc=bike_path_perc, cut=cut, ps=blp_idx,
                    node_size=ns, rev=rev, minmode=minmode,
                    plot_folder=plot_folder, plot_format='png')

    # plt.show()
    plt.close('all')

    hf_group['edited edges'] = edited_edges_nx
    hf_group['bpp'] = blp_cut
    hf_group['bpp at end'] = blp[end]
    hf_group['ba'] = ba[:end]
    hf_group['ba for comp'] = ba_y
    hf_group['cost'] = total_cost[:end]
    hf_group['nos'] = nos[:end]
    hf_group['nos at comp'] = nos_y
    hf_group['los'] = los[:end]
    hf_group['los at comp'] = los_y
    hf_group['tfdt max'] = tfdt_max
    hf_group['tfdt min'] = tfdt_min
    hf_group['trdt max'] = trdt_max
    hf_group['trdt min'] = trdt_min

    return blp_now, ba_now, cost_now, nos_now, los_now


def compare_modes(city, save, label, comp_folder, color, plot_folder,
                  plot_format='png', labels=False, titles=False):
    hf = h5py.File(comp_folder + 'comp_{}.hdf5'.format(save), 'r')

    bpp_now = hf['p+s']['bpp'][()]
    ba_now = hf['p+s']['ba'][()]
    nos_now = hf['p+s']['nos'][()]
    los_now = hf['p+s']['los'][()]

    fig1, ax1 = plt.subplots(dpi=150, figsize=(12, 10))
    ax1.set_xlabel('fraction of bike paths', fontsize=24)
    ax1.set_ylabel('bikeability', fontsize=24)
    if titles:
        ax1.set_title('bikeability of {}'.format(city), fontsize=14)
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
        ax2.set_title('Integrated Bikeability of {}'.format(city), fontsize=30)
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
        ax3.set_title('CBA of {}'.format(city), fontsize=30)
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

    ax42.set_ylabel('number of trips', fontsize=24)
    ax42.tick_params(axis='y', labelsize=16)
    ax42.yaxis.set_minor_locator(AutoMinorLocator())

    if titles:
        ax4.set_title('Trips and Length on Street in {}'.format(city),
                      fontsize=30)
    ax4.set_xlabel('fraction of bike paths', fontsize=24)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())

    ax4_hand = {}
    grp_algo = hf['algorithm']
    for m, d in grp_algo.items():
        bpp = d['bpp'][()]
        """bikeab = [np.trapz(ba[m][:idx], blp[m][:idx]) for idx in
                  range(len(blp[m]))]"""
        """cba = [bikeab[idx] / cost[m][idx] for idx in
               range(len(blp[m]))]"""
        ax1.plot(bpp, d['ba'][()], color=color[m], label=label[m])
        """ax2.plot(bpp, bikeab, color=color[m], label=label[m])
        ax3.plot(bpp, cba, color=color[m], label=label[m])"""
        space = round(len(bpp) / 20)
        ax4.plot(bpp, d['nos'][()], color=color[m], marker='v',
                 markevery=space, label=label[m])
        ax42.plot(bpp, d['los'][()], color=color[m], marker='8',
                  markevery=space, label=label[m])
        ax4_hand[m] = mlines.Line2D([], [], color=color[m], label=label[m])

    # ax1.plot(blp_now, ba_now, c='#999999', marker='D')
    xmax, ymax = coord_transf(bpp_now, ba_now, xmax=1, ymax=1, xmin=0, ymin=0)
    # ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    # ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    # ax4.plot(blp_now, nos_now, c='#999999', marker='v')
    # ax4.plot(blp_now, los_now, c='#999999', marker='8')
    xmax, ymax = coord_transf(bpp_now, nos_now, xmax=1, ymax=1, xmin=0, ymin=0)
    # ax4.axvline(x=blp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    # ax4.axhline(y=nos_now, xmax=1, xmin=xmax, c='#999999', ls=':', alpha=0.5)
    # ax4.axhline(y=los_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    l_keys_r = []
    l_keys_b = []
    for mode, l_key in ax4_hand.items():
        if not mode[0]:
            l_keys_r.append(l_key)
        else:
            l_keys_b.append(l_key)

    if labels:
        if not l_keys_b:
            l_keys = [tuple(l_keys_r)] + [tuple(l_keys_b)]
            l_labels = ['Removing', 'Building']
        else:
            l_keys = l_keys_r
            l_labels = ['Unweigthed', 'Street Type Penalty',
                        'Average Trip Length']
    else:
        l_keys = []
        l_labels = []

    if labels:
        ax1.legend(l_keys, l_labels, numpoints=1, loc=4, fontsize=24,
                   markerscale=2,
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        ax2.legend(l_keys, l_labels, numpoints=1, loc=4, fontsize=24,
                   markerscale=2,
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        ax3.legend(l_keys, l_labels, numpoints=1, loc=4, fontsize=24,
                   markerscale=2,
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        ax1.legend(loc=4, fontsize=24, markerscale=2)
        ax2.legend(loc=4, fontsize=24, markerscale=2)
        ax3.legend(numpoints=1, loc=4, fontsize=24, markerscale=2)

    l_keys.append(mlines.Line2D([], [], color='k', marker='v', label='trips'))
    l_labels.append('trips')
    l_keys.append(mlines.Line2D([], [], color='k', marker='8', label='length'))
    l_labels.append('length')
    ax4.legend(l_keys, l_labels, numpoints=1, loc=1, fontsize=24,
               markerscale=2,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    fig1.savefig(plot_folder + '{}_1.{}'.format(save, plot_format),
                 format=plot_format, bbox_inches='tight')
    """fig2.savefig(plot_folder + '{}_2.{}'.format(save, plot_format),
                 format=plot_format, bbox_inches='tight')
    fig3.savefig(plot_folder + '{}_3.{}'.format(save, plot_format),
                 format=plot_format, bbox_inches='tight')"""
    fig4.savefig(plot_folder + '{}_4.{}'.format(save, plot_format),
                 format=plot_format, bbox_inches='tight')

    plt.close('all')
    # plt.show()


def plot_city(city, save, polygon, input_folder, output_folder, comp_folder,
              plot_folder, modes, comp_modes=False, bike_paths=None,
              plot_evo=False, evo_for=None, plot_format='png'):
    if evo_for is None:
        evo_for = [(False, 1)]

    hf_comp = h5py.File(comp_folder+'comp_{}.hdf5'.format(save), 'w')
    hf_comp.attrs['city'] = city

    plt.rcdefaults()

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


    print('City: {}, stations: {},trips: {} (rt excl.: {}), '
          'unique trips: {} (rt excl. {})'
          .format(city, len(stations), trips, trips_re, utrips, utrips_re))

    nxG = ox.load_graphml(filepath=input_folder+'{}.graphml'.format(save),
                          node_type=int)
    hf_comp.attrs['nodes'] = len(nxG.nodes)
    hf_comp.attrs['edges'] = len(nxG.edges)
    print('City: {}, nodes: {}, edges: {}'.format(city, len(nxG.nodes),
                                                  len(nxG.edges)))
    nxG_plot = nxG.to_undirected()
    nxG_calc = nx.Graph(nxG.to_undirected())
    st_ratio = get_street_type_ratio(nxG_calc)
    hf_comp['ratio street type'] = json.dumps(st_ratio)
    sn_ratio = len(stations) / len(nxG_calc.nodes())
    hf_comp['ratio staions nodes'] = sn_ratio

    area = calc_polygon_area(polygon)
    hf_comp.attrs['area'] = area
    print('Area {}'.format(round(area, 1)))
    sa_ratio = len(stations) / area
    hf_comp['ratio stations area'] = sa_ratio

    data_now = calc_current_state(nxG_calc, trip_nbrs, bike_paths)
    # modes = ['{:d}{:}'.format(m[0], m[1]) for m in modes]
    data = {}
    for m in modes:
        hf_in = h5py.File(output_folder+'{}_data_mode_{:d}{:}.hdf5'
                          .format(save, m[0], m[1]), 'r')
        data[m] = hf_in['all']
        """data[m] = np.load(output_folder + '{}_data_mode_{:d}{:}.npy'
                          .format(save, m[0], m[1]), allow_pickle=True)"""

    end = max([get_end(json.loads(d['trdt'][()]), data_now[3], m[0])
               for m, d in data.items()])
    # end = max([len(d[4]) for m, d in data.items()])-1
    print('Cut after: ', end)

    bpp_now, ba_now, cost_now, nos_now, los_now = 0, 0, 0, 0, 0

    grp_algo = hf_comp.create_group('algorithm')
    for m, d in data.items():
        m_1 = '{:d}{:}'.format(m[0], m[1])
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
                      plot_format=plot_format)

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
            m_1 = '{:d}{:}'.format(m[0], m[1])
            if m[0]:
                c[m_1] = c_rev[m[1]]
            else:
                c[m_1] = c_norm[m[1]]
        # label = {m: 'Removing' if m[0] == '0' else 'Building' for m in modes}
        label = {'00': 'Unweigthed', '01': 'Street Type Penalty',
                 '02': 'Average Trip Length'}
        compare_modes(city=city, save=save, color=c, label=label,
                      comp_folder=comp_folder, plot_folder=plot_folder,
                      plot_format=plot_format)


def compare_distributions(city, base, base_save, graph_folder, data_folder,
                          plot_folder, mode, titles=False, legends=False,
                          figsize=None,  plot_format='png'):
    if figsize is None:
        figsize = [16, 9]
    rev = mode[0]
    minmode = mode[1]
    mode = '{:d}{:}'.format(rev, minmode)

    G = ox.load_graphml(filepath=graph_folder+'{}.graphml'.format(base_save),
                        node_type=int)

    saves = {'Real Data': base_save, 'Homog. Demand': base_save+'_ata',
             'Homog. Stations': base_save+'_homog'}
    dist_modes = [k for k, v in saves.items()]

    scale_x = calc_scale(base, dist_modes, saves, data_folder, mode)


    colors = ['teal', 'purple', 'blue']
    color = {m: colors[idx] for idx, m in enumerate(dist_modes)}

    ee = {}
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
    ba_y = {}
    bpp_now = {}

    for dist_mode in dist_modes:
        data = h5py.File(data_folder + 'comp_{}{}.hdf5'.format(base_save,
                                                               dist_mode), 'r')
        data_algo = data['algorithm']
        data_ps = data['p+s']
        ee[dist_mode] = [(i[0], i[1]) for i in data_algo[mode]['ee'][()]]
        bpp[dist_mode] = data_algo[mode]['bpp'][()]
        bpp_end[dist_mode] = data_algo[mode]['bpp at end'][()]
        ba[dist_mode] = data_algo[mode]['ba'][()]
        ba_y[dist_mode] = data_algo[mode]['ba for comp'][()]
        ba_improve[dist_mode] = ba_y[dist_mode] - data_ps['ba'][()]
        cost[dist_mode] = data_algo[mode]['cost'][()]
        nos[dist_mode] = data_algo[mode]['nos'][()]
        los[dist_mode] = data_algo[mode]['los'][()]
        tfdt_min[dist_mode] = data_algo[mode]['tfdt min'][()]
        tfdt_max[dist_mode] = data_algo[mode]['tfdt max'][()]
        tfdt_rat[dist_mode] = tfdt_max[dist_mode] / tfdt_min[dist_mode]
        trdt_min[dist_mode] = data_algo[mode]['trdt min'][()]
        trdt_max[dist_mode] = data_algo[mode]['trdt max'][()]
        trdt_rat[dist_mode] = trdt_max[dist_mode] / trdt_min[dist_mode]
        bpp_now[dist_mode] = data_ps['bpp'][()]
        st_ratio[dist_mode] = json.loads(data['ratio street type'][()])
        sn_ratio[dist_mode] = data['ratio stations nodes'][()]
        sa_ratio[dist_mode] = data['ratio stations area'][()]

    fig1, ax1 = plt.subplots(dpi=150, figsize=figsize)
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax1.set_ylabel('bikeability', fontsize=16)
    if titles:
        ax1.set_title('Comparison of Bikeabilities', fontsize=30)
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
        ax2.set_title('Comparison of Scaled Bikeabilities', fontsize=24)
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

    ax3.set_ylabel('number of trips', fontsize=24)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.yaxis.set_minor_locator(AutoMinorLocator())

    ax32.set_ylabel('length on street', fontsize=24)
    ax32.tick_params(axis='y', labelsize=16)
    ax32.yaxis.set_minor_locator(AutoMinorLocator())
    if titles:
        ax3.set_title('Comaprison of Trips and Length on Street', fontsize=30)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=24)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3_hand = {}

    for dist_mode in dist_modes:
        blp_scaled = [x * scale_x[dist_mode] for x in bpp[dist_mode]]
        ax1.plot(bpp[dist_mode], ba[dist_mode], color=color[dist_mode],
                 label='{}'.format(dist_mode))
        ax2.plot(blp_scaled, ba[dist_mode], color=color[dist_mode],
                 label='{}'.format(dist_mode))

        space = round(len(bpp[dist_mode]) / 25)
        ax3.plot(bpp[dist_mode], nos[dist_mode], marker='v', markevery=space,
                 color=color[dist_mode])
        ax32.plot(bpp[dist_mode], los[dist_mode], marker='8', markevery=space,
                  color=color[dist_mode])
        ax3_hand[dist_mode] = mlines.Line2D([], [], color=color[dist_mode])

    if legends:
        ax1.legend(loc='lower right')
        ax2.legend(loc='lower right')
        l_keys = [l_key for city, l_key in ax3_hand.items()]
        l_cities = [city for city, l_key in ax3_hand.items()]
    else:
        l_keys = []
        l_cities = []
    l_keys.append(mlines.Line2D([], [], color='k', marker='v', label='trips'))
    l_cities.append('trips')
    l_keys.append(mlines.Line2D([], [], color='k', marker='8', label='length'))
    l_cities.append('length')

    ax3.legend(l_keys, l_cities, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    save_plot = '{}{}_distr_comp_'.format(plot_folder, base_save)
    fig1.savefig(save_plot + 'ba_unscaled_{:d}{:}.{}'
                 .format(rev, minmode, plot_format),
                 format=plot_format, bbox_inches='tight')
    fig2.savefig(save_plot + 'ba_scaled_{:d}{:}.{}'
                 .format(rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')
    fig3.savefig(save_plot + 'los_nos_{:d}{:}.{}'
                 .format(rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')
    plot_barh(scale_x, color, save_plot + 'scalefactor_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    plot_barh(trdt_rat, color, save_plot + 'ratio_max_min_traveled_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    plot_barh(sn_ratio, color, save_plot + 'ratio_stations_nodes_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format,
              x_label=r'$trdt_{max} / trdt_{min}$')
    plot_barh(sa_ratio, color, save_plot + 'ratio_stations_area_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format,
              x_label=r'stations / $km^{2}$')
    plot_barh(bpp_end, color, save_plot + 'fmax_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    plot_barh(ba_improve, color, save_plot + 'baimprove_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    st = ['primary', 'secondary', 'tertiary', 'residential']
    st_data = {city: list(ratio.values()) for city, ratio in st_ratio.items()}
    st_colors = ['darkblue', 'darkgreen', 'darkcyan', 'darkorange']
    plot_barh_stacked(st_data, st, st_colors, save_plot + 'ratio_st_{:d}{:}'
                      .format(rev, minmode), plot_format=plot_format)

    plot_bp_diff(G, ee[dist_modes[0]], ee[dist_modes[1]],
                 bpp[dist_modes[0]], bpp[dist_modes[1]],
                 bpp_now[dist_modes[0]], 'k', 0, save, rev, minmode,
                 plot_folder, plot_format='png', figsize=None, dpi=150)
    plot_bp_diff(G, ee[dist_modes[1]], ee[dist_modes[2]],
                 bpp[dist_modes[1]], bpp[dist_modes[2]],
                 bpp_now[dist_modes[1]], 'k', 0, save+'_ata', rev, minmode,
                 plot_folder, plot_format='png', figsize=None, dpi=150)
    plt.close('all')
    # plt.show()


def compare_cities(cities, saves, mode, color, data_folder, plot_folder,
                   scale_x=None, base_city=None, figsize=None,
                   plot_format='png'):
    if scale_x is None:
        scale_x = {city: 1 for city in cities}
    if base_city is None:
        base_city = np.random.choice(cities)
    if figsize is None:
        figsize = [12, 9]
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()

    rev = mode[0]
    minmode = mode[1]
    mode = '{:d}{:}'.format(rev, minmode)

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
    ba_y = {}

    for city in cities:
        save = saves[city]
        data = h5py.File(data_folder + 'comp_{}.hdf5'.format(save), 'r')
        data_algo = data['algorithm']
        data_ps = data['p+s']
        bpp[city] = data_algo[mode]['bpp'][()]
        bpp_end[city] = data_algo[mode]['bpp at end'][()]
        ba[city] = data_algo['ba'][mode][()]
        ba_y[city] = data_algo[mode]['ba for comp']
        ba_improve[city] = ba_y[city] - data_ps['ba'][()]
        cost[city] = data_algo[mode]['cost'][()]
        nos[city] = data_algo[mode]['nos'][()]
        los[city] = data_algo[mode]['los'][()]
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
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax1.set_ylabel('bikeability', fontsize=12)
    # ax1.set_title('Comparison of Bikeabilities', fontsize=14)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0.0, 1.0)

    fig2, ax2 = plt.subplots(dpi=150, figsize=figsize)
    ax2.set_xlabel('scaled normalised fraction of bike paths', fontsize=12)
    ax2.set_ylabel('bikeability', fontsize=12)
    # ax2.set_title('Comparison of Scaled Bikeabilities', fontsize=14)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlim(0.0, 1.0)

    fig3, ax3 = plt.subplots(dpi=150, figsize=figsize)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax3.set_ylabel('integrated bikeability', fontsize=12)
    # ax3.set_title('Comparison of Integrated Bikeabilities', fontsize=14)
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

    ax4.set_ylabel('number of trips', fontsize=12)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.yaxis.set_minor_locator(AutoMinorLocator())

    ax42.set_ylabel('length on street', fontsize=12)
    ax42.tick_params(axis='y', labelsize=16)
    ax42.yaxis.set_minor_locator(AutoMinorLocator())

    # ax4.set_title('Comaprison of Trips and Length on Street', fontsize=14)
    ax4.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())

    ax4_hand = {}

    for city in cities:
        bikeab = [np.trapz(ba[city][:idx], bpp[city][:idx]) for idx in
                  range(len(bpp[city]))]
        blp_scaled = [x * scale_x[city] for x in bpp[city]]
        ax1.plot(bpp[city], ba[city], color=color[city],
                 label='{}'.format(city))
        ax2.plot(blp_scaled, ba[city], color=color[city],
                 label='{}'.format(city))
        ax3.plot(bpp[city], bikeab, color=color[city],
                 label='{}'.format(city))
        space = round(len(bpp[city]) / 25)

        ax4.plot(bpp[city], nos[city], marker='v', markevery=space,
                 color=color[city])
        ax42.plot(bpp[city], los[city], marker='8', markevery=space,
                  color=color[city])
        ax4_hand[city] = mlines.Line2D([], [], color=color[city])

    # l_keys = [l_key for city, l_key in ax4_hand.items()]
    # l_cities = [city for city, l_key in ax4_hand.items()]
    l_keys = []
    l_cities = []
    l_keys.append(mlines.Line2D([], [], color='k', marker='v', label='trips'))
    l_cities.append('trips')
    l_keys.append(mlines.Line2D([], [], color='k', marker='8', label='length'))
    l_cities.append('length')

    # ax1.legend(loc='lower right')
    # ax2.legend(loc='lower right')
    # ax3.legend(loc='lower right')
    ax4.legend(l_keys, l_cities, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    save_plot = '{}{}_'.format(plot_folder, saves[base_city])
    fig1.savefig(save_plot + 'ba_unscaled_{:d}{:}.{}'
                 .format(rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')
    fig2.savefig(save_plot + 'ba_scaled_{:d}{:}.{}'
                 .format(rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')
    # fig3.savefig(plot_folder+'comparison-3-{:d}{:}.{}'.format(rev, minmode,
    #                                                            plot_format),
    #             format=plot_format)
    fig4.savefig(save_plot + 'los_nos_{:d}{:}.{}'
                 .format(rev, minmode, plot_format), format=plot_format,
                 bbox_inches='tight')
    plot_barh(scale_x, color, save_plot+'scalefactor_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    plot_barh(trdt_rat, color, save_plot+'ratio_max_min_traveled_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    plot_barh(sn_ratio, color, save_plot+'ratio_stations_nodes_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format,
              x_label=r'$trdt_{max} / trdt_{min}$')
    plot_barh(sa_ratio, color, save_plot+'ratio_stations_area_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format,
              x_label=r'stations / $km^{2}$')
    plot_barh(bpp_end, color, save_plot+'fmax_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    plot_barh(ba_improve, color, save_plot+'baimprove_{:d}{:}'
              .format(rev, minmode), plot_format=plot_format)
    st = ['primary', 'secondary', 'tertiary', 'residential']
    st_data = {city: list(ratio.values()) for city, ratio in st_ratio.items()}
    st_colors = ['darkblue', 'darkgreen', 'darkcyan', 'darkorange']
    plot_barh_stacked(st_data, st, st_colors, save_plot+'ratio_st_{:d}{:}'
                      .format(rev, minmode), plot_format=plot_format)

    plt.close('all')
    # plt.show()
