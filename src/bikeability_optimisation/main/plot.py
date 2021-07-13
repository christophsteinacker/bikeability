"""
This module includes all necessary functions for the plotting functionality.
"""
import json
import osmnx as ox
import matplotlib.ticker as ticker
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from pathlib import Path
from shapely.geometry import Polygon
from .data import get_polygon_from_json, get_polygons_from_json, \
    average_rand_demand, create_default_paths, create_default_params
from .algorithm import calc_current_state
from ..helper.plot_helper import *


def plot_street_network(city, save, G, plot_folder, params=None):
    """
    Plots the street network of graph G.

    """
    fig, ax = plt.subplots(figsize=params["figs_snetwork"],
                           dpi=params["dpi"])
    ox.plot_graph(G, ax=ax, bgcolor='#ffffff', show=False, close=False,
                  node_color=params["nc_snetwork"],
                  node_size=params["ns_snetwork"], node_zorder=3,
                  edge_color=params["ec_snetwork"],
                  edge_linewidth=params["ew_snetwork"])
    if params["titles"]:
        fig.suptitle(f'Graph used for {city.capitalize()}',
                     fontsize=params["fs_title"])

    plt.savefig(f'{plot_folder}{save}_street_network.{params["plot_format"]}')


def plot_used_nodes(city, save, G, trip_nbrs, stations, plot_folder,
                    params=None):
    """
    Plots usage of nodes in graph G. trip_nbrs and stations should be
    structured as returned from load_trips().
    """
    print('Plotting used nodes.')

    if params is None:
        params = create_default_params()

    nodes = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if e_node == s_node:
                continue
            if (s_node, e_node) in trip_nbrs:
                nodes[s_node] += trip_nbrs[(s_node, e_node)]
                nodes[e_node] += trip_nbrs[(s_node, e_node)]

    nodes = {n: int(t / params["stat_usage_norm"]) for n, t in nodes.items()}

    trip_count = sum(trip_nbrs.values())
    station_count = len(stations)

    max_n = max(nodes.values())

    print(f'Maximal station usage: {max_n}')

    n_rel = {key: value for key, value in nodes.items()}
    ns = [params["nodesize"] if n in stations else 0 for n in G.nodes()]

    for n in G.nodes():
        if n not in stations:
            n_rel[n] = max_n + 1
    min_n = min(n_rel.values())

    print(f'Minimal station usage: {min_n}')

    r = magnitude(max_n)

    hist_data = [value for key, value in n_rel.items() if value != max_n + 1]
    hist_save = f'{plot_folder}{save}_stations_usage_distribution'
    hist_xlim = (0.0, round(max_n, -(r - 1)))

    cmap = plt.cm.get_cmap(params["cmap_nodes"])

    plot_histogram(hist_data, hist_save,
                   xlabel='total number of trips per year',
                   ylabel='number of stations', xlim=hist_xlim, bins=25,
                   cm=cmap, plot_format=params["plot_format"],
                   dpi=params["dpi"],
                   figsize=params["figs_station_usage_hist"])

    cmap = ['#808080'] + [rgb2hex(cmap(n)) for n in
                          reversed(np.linspace(1, 0, max_n, endpoint=True))] \
           + ['#ffffff']
    color_n = [cmap[v] for k, v in n_rel.items()]

    fig2, ax2 = plt.subplots(dpi=params["dpi"],
                             figsize=params["figs_station_usage"])
    ox.plot_graph(G, ax=ax2, bgcolor='#ffffff', edge_linewidth=0.3,
                  node_color=color_n, node_size=ns, node_zorder=3,
                  show=False, close=False,
                  edge_color=params["ec_station_usage"])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(params["cmap_nodes"]),
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
    cbar.ax.tick_params(axis='x', labelsize=params["fs_ticks"], width=0.5)
    cbar.ax.set_xlabel('total number of trips per year',
                       fontsize=params["fs_axl"])

    if params["titles"]:
        fig2.suptitle(f'{city.capitalize()}, Stations: {station_count}, '
                      f'Trips: {trip_count}', fontsize=params["fs_title"])

    fig2.savefig(f'{plot_folder}{save}_stations_used.{params["plot_format"]}',
                 bbox_inches='tight')

    plt.close('all')


def plot_bp_comparison(city, save, G, ee_algo, ee_cs, bpp_algo, bpp_cs,
                       stations, rev, minmode, plot_folder,
                       mode='diff', params=None):
    nx.set_edge_attributes(G, False, 'algo')
    nx.set_edge_attributes(G, False, 'cs')

    ns = [params["nodesize"] if n in stations else 0 for n in G.nodes()]

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

    len_algo = 0
    len_cs = 0
    len_both = 0

    if mode == 'algo':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['algo']:
                ec.append(params["color_algo"])
            else:
                ec.append(params["color_unused"])
                unused.append((u, v, k))
    elif mode == 'p+s':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['cs']:
                ec.append(params["color_cs"])
            else:
                ec.append(params["color_unused"])
                unused.append((u, v, k))
    elif mode == 'diff':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['algo'] and data['cs']:
                ec.append(params["color_both"])
                ee_both.append((u, v, k))
                len_both += data['length']
            elif data['algo'] and not data['cs']:
                ec.append(params["color_algo"])
                ee_algo_only.append((u, v, k))
                len_algo += data['length']
            elif not data['algo'] and data['cs']:
                ec.append(params["color_cs"])
                ee_cs_only.append((u, v, k))
                len_cs += data['length']
            else:
                ec.append(params["color_unused"])
                unused.append((u, v, k))
        print(f'Overlap between p+s and algo: '
              f'{len_both / (len_cs + len_both):3.2f}')
    else:
        print('You have to choose between algo, p+s and diff.')

    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_bp_comp"])
    ox.plot_graph(G, bgcolor='#ffffff', ax=ax,
                  node_size=ns, node_color=params["nc_pb_evo"],
                  node_zorder=3, edge_linewidth=0.6, edge_color=ec,
                  show=False, close=False)
    if params["legends"]:
        lw_leg = params["lw_legend_bp_evo"]
        if mode == 'algo':
            leg = [Line2D([0], [0], color=params["color_algo"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_unused"], lw=lw_leg)]
            ax.legend(leg, ['Algorithm', 'None'],
                      bbox_to_anchor=(0, -0.05, 1, 1), loc=3, ncol=2,
                      mode="expand", borderaxespad=0.,
                      fontsize=params["fs_legend"])
        elif mode == 'p+s':
            leg = [Line2D([0], [0], color=params["color_cs"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_unused"], lw=lw_leg)]
            ax.legend(leg, ['Primary + Secondary', 'None'],
                      bbox_to_anchor=(0, -0.05, 1, 1), loc=3, ncol=2,
                      mode="expand", borderaxespad=0.,
                      fontsize=params["fs_legend"])
        elif mode == 'diff':
            leg = [Line2D([0], [0], color=params["color_both"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_algo"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_cs"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_unused"], lw=lw_leg)]
            ax.legend(leg, ['Both', 'Algorithm', 'Primary+Secondary', 'None'],
                      bbox_to_anchor=(0, -0.05, 1, 1), loc=3, ncol=4,
                      mode="expand", borderaxespad=0.,
                      fontsize=params["fs_legend"])
    if params["titles"]:
        if mode == 'algo':
            ax.set_title(f'{city}: Algorithm', fontsize=params["fs_title"])
        elif mode == 'p+s':
            ax.set_title(f'{city}: Primary/Secondary',
                         fontsize=params["fs_title"])
        elif mode == 'diff':
            ax.set_title(f'{city}: Comparison', fontsize=params["fs_title"])

    plt.savefig(f'{plot_folder}{save}-bp-build-{rev:d}{minmode}_{mode}'
                f'.{params["plot_format"]}', bbox_inches='tight')
    plt.close(fig)


def plot_edges(G, edge_color, node_size, save_path, title='', figsize=(12, 12),
               params=None):
    fig, ax = plt.subplots(dpi=params["dpi"], figsize=figsize)
    ox.plot_graph(G, ax=ax, bgcolor='#ffffff', edge_color=edge_color,
                  edge_linewidth=1.5, node_color='C0', node_size=node_size,
                  node_zorder=3, show=False, close=False)
    if title != '':
        ax.set_title(title, fontsize=params["fs_title"])
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_bp_evo(save, G, edited_edges, bike_path_perc, cut, ps,
                node_size, rev, minmode, plot_folder, params=None):
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
                    f'{i}.{params["plot_format"]}'
        if params["titles"]:
            plot_edges(G, ec_evo, node_size, save_path,
                       title=f'Fraction of Bike Paths: {j*100:3.0f}%',
                       figsize=params["figs_bp_evo"], params=params)
        else:
            plot_edges(G, ec_evo, node_size, save_path,
                       figsize=params["figs_bp_evo"], params=params)

    nx.set_edge_attributes(G, False, 'bike path')
    ee_cut = ee[:cut]
    ec_cut = get_edge_color(G, ee_cut, 'bike path', edited_color)
    save_path = f'{plot_folder_evo}{save}-edited-mode-{rev:d}{minmode}-' \
                f'cut.{params["plot_format"]:}'
    plot_edges(G, ec_cut, node_size, save_path, figsize=params["figs_bp_evo"],
               params=params)

    nx.set_edge_attributes(G, False, 'bike path')
    ee_ps = ee[:ps]
    ec_ps = get_edge_color(G, ee_ps, 'bike path', edited_color)
    save_path = f'{plot_folder_evo}{save}-edited-mode-{rev:d}{minmode}-' \
                f'ps.{params["plot_format"]}'
    plot_edges(G, ec_ps, node_size, save_path, figsize=params["figs_bp_evo"],
               params=params)
    print('Finished plotting bike path evolution.')


def plot_mode(city, save, data, data_now, nxG_calc, nxG_plot, stations,
              trip_nbrs, mode, end, hf_group, plot_folder, evo=False,
              params=None):
    if params is None:
        params = create_default_params()

    rev = mode[0]
    minmode = mode[1]

    bike_paths_now = data_now[0]
    cost_now = data_now[1]
    bike_path_perc_now = data_now[2]
    trdt_now = data_now[3]
    tfdt_now = data_now[4]
    nos_now = data_now[5]

    edited_edges_nx = [(i[0], i[1]) for i in data['ee_nx'][()]]
    bike_path_perc = data['bpp'][()]
    cost = data['cost'][()]
    total_real_distance_traveled = json.loads(data['trdt'][()])
    total_felt_distance_traveled = json.loads(data['tfdt'][()])
    nbr_on_street = data['nos'][()]
    trdt, trdt_now = total_distance_traveled_list(total_real_distance_traveled,
                                                  trdt_now, rev)
    tfdt, tfdt_now = total_distance_traveled_list(total_felt_distance_traveled,
                                                  tfdt_now, rev)

    if rev:
        bpp = bike_path_perc
    else:
        bpp = list(reversed(bike_path_perc))
    trdt_min = min(trdt['all'])
    trdt_max = max(trdt['all'])
    tfdt_min = min(tfdt['all'])
    tfdt_max = max(tfdt['all'])
    ba = [1 - (i - tfdt_min) / (tfdt_max - tfdt_min) for i in tfdt['all']]

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

    cost_y = min(cost[:end], key=lambda x: abs(x - cost_now))
    cost_idx = next(x for x, val in enumerate(cost[:end]) if val == cost_y)
    cost_x = bpp_cut[cost_idx]

    nos_y = min(nos[:end], key=lambda x: abs(x - nos_now))
    nos_idx = next(x for x, val in enumerate(nos[:end]) if val == nos_y)
    nos_x = bpp_cut[nos_idx]
    nos_y = nos[bpp_idx]

    los_y = min(los[:end], key=lambda x: abs(x - los_now))
    los_idx = next(x for x, val in enumerate(los[:end]) if val == los_y)
    los_x = bpp_cut[los_idx]
    los_y = los[bpp_idx]

    cut = next(x for x, val in enumerate(ba) if val >= 1)
    total_cost, cost_now = sum_total_cost(cost, cost_now, rev)
    cost_now = cost_now / total_cost[end]

    ns = [params["nodesize"] if n in stations else 0 for n in nxG_plot.nodes()]

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
    fig1, ax1 = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    ax12 = ax1.twinx()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax12.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax12.set_ylim(0.0, 1.0)

    ax1.plot(bpp_cut, ba[:end], c=params["c_ba"], label='bikeability',
             lw=params["lw_ba"])
    ax1.plot(bpp_now, ba_now, c=params["c_ba"], ms=params["ms_ba"],
             marker=params["m_ba"])
    xmax, ymax = coord_transf(bpp_now, max([ba_y, ba_now]),
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=bpp_now, ymax=ymax, ymin=0, c=params["c_ba"], ls='--',
                alpha=0.5)
    ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c=params["c_ba"], ls='--',
                alpha=0.5)
    ax1.axhline(y=ba_y, xmax=xmax, xmin=0, c=params["c_ba"], ls='--',
                alpha=0.5)
    print('ba:', ba_y)

    ax1.set_ylabel('bikeability b(m)', fontsize=params["fs_axl"],
                   color=params["c_ba"])
    ax1.tick_params(axis='y', labelsize=params["fs_ticks"],
                    labelcolor=params["c_ba"])
    ax1.tick_params(axis='x', labelsize=params["fs_ticks"])
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax12.plot(bpp_cut, [x / total_cost[end] for x in total_cost[:end]],
              c=params["c_cost"], label='total cost', lw=params["lw_cost"])
    ax12.plot(bpp_now, cost_now, c=params["c_cost"], ms=params["ms_cost"],
              marker=params["m_cost"])
    xmin, ymax = coord_transf(bpp_now, cost_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=bpp_now, ymax=ymax, ymin=0, c=params["c_cost"], ls='--',
                alpha=0.5)
    ax1.axhline(y=cost_now, xmax=1, xmin=xmin, c=params["c_cost"], ls='--',
                alpha=0.5)
    ax1.axhline(y=cost_y, xmax=1, xmin=xmin, c=params["c_cost"], ls='--',
                alpha=0.5)

    ax12.set_ylabel('normalised cost', fontsize=params["fs_axl"],
                    color=params["c_cost"])
    ax12.tick_params(axis='y', labelsize=params["fs_ticks"],
                     labelcolor=params["c_cost"])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax12.yaxis.set_minor_locator(AutoMinorLocator())

    ax1.set_xlabel('normalised fraction of bike paths',
                   fontsize=params["fs_axl"])
    if params["titles"]:
        ax1.set_title('Bikeability and Cost', fontsize=params["fs_title"])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(False)
    ax12.grid(False)

    handles = ax1.get_legend_handles_labels()[0]
    handles.append(ax12.get_legend_handles_labels()[0][0])
    if params["legends"]:
        ax1.legend(handles=handles, loc='lower right',
                   fontsize=params["fs_legend"])

    fig1.savefig(f'{plot_folder}{save}_ba_tc_mode_{rev:d}{minmode}'
                 f'.{params["plot_format"]}', bbox_inches='tight')

    ax1ins = zoomed_inset_axes(ax1, 3.5, loc=1)
    x1, x2, y1, y2 = round(bpp_now - 0.05, 2), round(bpp_now + 0.05, 2), \
                     round(ba_now - 0.03, 2), min(round(ba_y + 0.03, 2), 1)
    ax1ins.plot(bpp_cut, ba[:end], lw=params["lw_ba"])
    ax1ins.plot(bpp_now, ba_now, c=params["c_ba"], ms=params["ms_ba"],
                marker=params["m_ba"])
    xmax, ymax = coord_transf(bpp_now, max([ba_y, ba_now]),
                              xmin=x1, xmax=x2, ymin=y1, ymax=y2)
    ax1ins.axvline(x=bpp_now, ymax=ymax, ymin=0, c=params["c_ba"], ls='--',
                   alpha=0.5)
    ax1ins.axhline(y=ba_now, xmax=xmax, xmin=0, c=params["c_ba"], ls='--',
                   alpha=0.5)
    ax1ins.axhline(y=ba_y, xmax=xmax, xmin=0, c=params["c_ba"],  ls='--',
                   alpha=0.5)

    ax1ins.set_xlim(x1, x2)
    ax1ins.set_ylim(y1, y2)
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")
    ax1ins.tick_params(axis='y', labelsize=params["fs_ticks"],
                       labelcolor=params["c_ba"])
    ax1ins.tick_params(axis='x', labelsize=params["fs_ticks"])
    ax1ins.yaxis.set_minor_locator(AutoMinorLocator())
    ax1ins.xaxis.set_minor_locator(AutoMinorLocator())
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")

    fig1.savefig(f'{plot_folder}{save}_ba_tc_zoom_mode_{rev:d}{minmode}'
                 f'.{params["plot_format"]}', bbox_inches='tight')

    fig2, ax2 = plt.subplots(dpi=params["dpi"], figsize=params["figs_los_nos"])
    ax22 = ax2.twinx()
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax22.set_ylim(0.0, 1.0)

    p1, = ax2.plot(bpp_cut, los[:end], label='length', c=params["c_los"],
                   lw=params["lw_los"], zorder=1)
    ax2.plot(bpp_now, los_now, c=params["c_los"], ms=params["ms_los"],
             marker=params["m_los"], zorder=3)
    xmax, ymax = coord_transf(max(bpp_now, los_x), los_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax2.axvline(x=bpp_now, ymax=ymax, ymin=0, c=params["c_los"], ls='--',
                alpha=0.5, zorder=2)
    ax2.axhline(y=los_now, xmax=xmax, xmin=0, c=params["c_los"], ls='--',
                alpha=0.5, zorder=2)
    ax2.axvline(x=los_x, ymax=ymax, ymin=0, c=params["c_los"], ls='--',
                alpha=0.5)
    ax2.axhline(y=los_y, xmax=xmax, xmin=0, c=params["c_los"], ls='--',
                alpha=0.5, zorder=2)

    ax2.set_ylabel('length on street', fontsize=params["fs_axl"],
                   color=params["c_los"])
    ax2.tick_params(axis='y', labelsize=params["fs_ticks"],
                    labelcolor=params["c_los"])
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    p2, = ax22.plot(bpp_cut, nos[:end], label='cyclists', c=params["c_nos"],
                    lw=params["lw_nos"], zorder=1)
    ax22.plot(bpp_now, nos_now, c=params["c_nos"], ms=params["ms_nos"],
              marker=params["m_nos"], zorder=3)
    xmin, ymax = coord_transf(max(bpp_now, nos_x), nos_now,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax22.axvline(x=bpp_now, ymax=ymax, ymin=0, c=params["c_nos"], ls='--',
                 alpha=0.5, zorder=2)
    ax22.axhline(y=nos_now, xmax=1, xmin=xmin, c=params["c_nos"], ls='--',
                 alpha=0.5, zorder=2)
    ax22.axvline(x=nos_x, ymax=ymax, ymin=0, c=params["c_nos"], ls='--',
                 alpha=0.5)
    ax22.axhline(y=nos_y, xmax=1, xmin=xmin, c=params["c_nos"], ls='--',
                 alpha=0.5, zorder=2)

    ax22.set_ylabel('cyclists on street', fontsize=params["fs_axl"],
                    color=params["c_nos"])
    ax22.tick_params(axis='y', labelsize=params["fs_ticks"],
                     labelcolor=params["c_nos"])
    ax22.yaxis.set_minor_locator(AutoMinorLocator())

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='x', labelsize=params["fs_ticks"])
    ax2.set_xlabel('normalised fraction of bike paths',
                   fontsize=params["fs_axl"])
    if params["titles"]:
        ax2.set_title('Cyclists and Length on Street',
                      fontsize=params["fs_title"])
    if params["legends"]:
        ax2.legend([p1, p2], [l.get_label() for l in [p1, p2]],
                   fontsize=params["fs_legend"])
    fig2.savefig(f'{plot_folder}{save}_trips_on_street_mode_{rev:d}{minmode}'
                 f'.{params["plot_format"]}', bbox_inches='tight')

    comp_st_driven = {st: [len_on_st[bpp_idx], trdt_st_now[st]]
                      for st, len_on_st in trdt_st.items()}
    plot_barv_stacked(['Algorithm', 'P+S'], comp_st_driven, params["c_st"],
                      width=0.3, title='',
                      save=f'{plot_folder}{save}_comp_st_driven_'
                           f'{rev:d}{minmode}',
                      plot_format=params["plot_format"],
                      figsize=params["figs_comp_st"])

    for bp_mode in ['algo', 'p+s', 'diff']:
        plot_bp_comparison(city=city, save=save, G=nxG_plot,
                           ee_algo=edited_edges_nx, ee_cs=bike_paths_now,
                           bpp_algo=bike_path_perc, bpp_cs=bike_path_perc_now,
                           stations=stations, rev=rev, minmode=minmode,
                           plot_folder=plot_folder, mode=bp_mode,
                           params=params)

    if evo:
        plot_bp_evo(save=save, G=nxG_plot, edited_edges=edited_edges_nx,
                    bike_path_perc=bike_path_perc, cut=cut, ps=bpp_idx,
                    node_size=ns, rev=rev, minmode=minmode,
                    plot_folder=plot_folder, params=params)

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


def plot_comp_rand_demand(city, save, bpp_ed, bpp_rd, ba_ed, ba_rd, end,
                          params, paths):
    fig, ax = plt.subplots(dpi=params["dpi"], figsize=(2.7, 2.6))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    bpp_ed_cut = [x / bpp_ed[end] for x in bpp_ed[:end]]
    ba_ed_cut = ba_ed[:end]
    bpp_rd_cut = [x / bpp_rd[end] for x in bpp_rd[:end]]
    ba_rd_cut = ba_rd[:end]

    ax.plot(bpp_ed_cut, ba_ed_cut, c=params["c_ed"], lw=params["lw_ed"])
    ax.plot(bpp_rd_cut, ba_rd_cut, c=params["c_rd"], lw=params["lw_rd"])
    poly = ax.fill(np.append(bpp_ed_cut, bpp_rd_cut[::-1]),
                   np.append(ba_ed_cut, ba_rd_cut[::-1]),
                   color=params["c_rd_ed_area"], alpha=params["a_rd_ed_area"])
    print(Polygon(poly[0].get_xy()).area)
    ax.set_ylabel('bikeability b(m)', fontsize=params["fs_axl"])
    ax.tick_params(axis='y', labelsize=params["fs_ticks"], width=0.5)
    ax.tick_params(axis='x', labelsize=params["fs_ticks"], width=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.set_xlabel('normalized fraction of bike paths m',
                  fontsize=params["fs_axl"])
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    if params["titles"]:
        ax.set_title(f'{city}', fontsize=params["fs_title"])
    if params["legends"]:
        ax.legend(loc='lower right', fontsize=params["fs_legend"])

    fig.savefig(f'{paths["plot_folder"]}results/{save}/{save}_ba_rd_comp'
                f'.{params["plot_format"]}', bbox_inches='tight')


def plot_city(city, save, paths=None, params=None):
    if paths is None:
        paths = create_default_paths()

    if params is None:
        params = create_default_params()

    # Define city specific folders
    comp_folder = f'{paths["comp_folder"]}/'
    plot_folder = f'{paths["plot_folder"]}results/{save}/'
    input_folder = f'{paths["input_folder"]}{save}/'
    output_folder = f'{paths["output_folder"]}{save}/'

    # Create non existing folders
    Path(comp_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    hf_comp = h5py.File(f'{comp_folder}comp_{save}.hdf5', 'w')
    hf_comp.attrs['city'] = city

    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']


    hf_demand = h5py.File(f'{input_folder}{save}_demand.hdf5', 'r')
    trip_nbrs = {(int(k1), int(k2)): v[()] for k1 in list(hf_demand.keys())
                 for k2, v in hf_demand[k1].items()}
    hf_demand.close()
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

    if paths["use_base_polygon"]:
        base_save = save.split(paths["save_devider"])[0]
        polygon_path = f'{paths["polygon_folder"]}{base_save}.json'
        polygon = get_polygon_from_json(polygon_path)
    else:
        polygon_path = f'{paths["polygon_folder"]}{save}.json'
        polygon = get_polygon_from_json(polygon_path)

    remove_area = None
    if params["correct_area"]:
        if paths["use_base_polygon"]:
            base_save = save.split(paths["save_devider"])[0]
            correct_area_path = f'{paths["polygon_folder"]}{base_save}' \
                                f'_delete.json'
        else:
            correct_area_path = f'{paths["polygon_folder"]}{save}_delete.json'
        if Path(correct_area_path).exists():
            remove_area = get_polygons_from_json(correct_area_path)
        else:
            print('No polygons for area size correction found.')

    area = calc_polygon_area(polygon, remove_area)
    hf_comp.attrs['area'] = area
    print(f'Area {round(area, 1)}')
    sa_ratio = len(stations) / area
    hf_comp['ratio stations area'] = sa_ratio

    plot_used_nodes(city=city, save=save, G=nxG_plot, trip_nbrs=trip_nbrs,
                    stations=stations, plot_folder=plot_folder, params=params)

    data_now = calc_current_state(nxG_calc, trip_nbrs, params["bike_paths"])

    data = {}
    for m in params["modes"]:
        hf_in = h5py.File(f'{output_folder}{save}_data_mode_'
                          f'{m[0]:d}{m[1]}.hdf5', 'r')
        data[m] = hf_in['all']

    if params["cut"]:
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
        if params["plot_evo"] and (m in params["evo_for"]):
            evo = True
        else:
            evo = False
        bpp_now, ba_now, cost_now, nos_now, los_now = \
            plot_mode(city=city, save=save, data=d, data_now=data_now,
                      nxG_calc=nxG_calc, nxG_plot=nxG_plot, stations=stations,
                      trip_nbrs=trip_nbrs, mode=m, end=end, evo=evo,
                      hf_group=sbgrp_algo, plot_folder=plot_folder,
                      params=params)

    grp_ps = hf_comp.create_group('p+s')
    grp_ps['bpp'] = bpp_now
    grp_ps['ba'] = ba_now
    grp_ps['cost'] = cost_now
    grp_ps['nos'] = nos_now
    grp_ps['los'] = los_now


def plot_city_rand_demand(city, save, paths, params, rd_pattern='rd',
                          nbr_of_rd_sets=10):
    # Plot the results for the single rand demand sets
    for rd_i in range(nbr_of_rd_sets):
        rd_save = f'{save}_{rd_pattern}_{rd_i+1}'
        plot_city(city=city, save=rd_save, paths=paths, params=params)

    comp_folder = f'{paths["comp_folder"]}/{save}/'

    # Average the results for the rand demand over the number of sets
    bpp_rd, ba_rd = average_rand_demand(save=save, comp_folder=comp_folder,
                                        rd_pattern=rd_pattern,
                                        nbr_of_rd_sets=nbr_of_rd_sets)

    # Plot the comparison between the empirical and rand demand
    data = h5py.File(f'{comp_folder}comp_{save}.hdf5', 'r')
    data_algo = data['algorithm']
    for m in params["modes"]:
        bpp_ed = list(reversed(data_algo[f'{m[0]:d}{m[1]}']
                               ['bpp complete'][()]))
        ba_ed = data_algo[f'{m[0]:d}{m[1]}']['ba complete'][()]
        end = data_algo[f'{m[0]:d}{m[1]}']['end'][()]
        data.close()
        plot_comp_rand_demand(city=city, save=save, bpp_ed=bpp_ed,
                              bpp_rd=bpp_rd, ba_ed=ba_ed, ba_rd=ba_rd,
                              end=end, params=params, paths=paths)
