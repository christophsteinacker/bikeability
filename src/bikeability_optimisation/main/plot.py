from math import ceil
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from matplotlib.colors import rgb2hex
from matplotlib.ticker import AutoMinorLocator
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from pathlib import Path
import matplotlib.lines as mlines
import osmnx as ox

from bikeability_optimisation.helper.plot_helper import *


def plot_used_nodes(city, save, G, trip_nbrs, stations, plot_folder,
                    plot_format='png'):
    print('Plotting used nodes.')

    nodes = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if (s_node, e_node) in trip_nbrs:
                nodes[s_node] += trip_nbrs[(s_node, e_node)]
                nodes[e_node] += trip_nbrs[(s_node, e_node)]

    max_n = max(nodes.values())
    n_rel = {key: ceil((value / max_n) * 100) for key, value in nodes.items()}
    ns = [100 if n in stations else 0 for n in G.nodes()]

    for n in G.nodes():
        if n not in stations:
            n_rel[n] = 101

    cmap_name = 'cool'
    cmap = plt.cm.get_cmap(cmap_name)
    cmap = ['#999999'] + \
           [rgb2hex(cmap(n)) for n in np.linspace(1, 0, 100, endpoint=False)] \
           + ['#ffffff']
    color_n = [cmap[v] for k, v in n_rel.items()]

    fig, ax = ox.plot_graph(G, fig_height=15, fig_width=15, dpi=300,
                            edge_linewidth=2, node_color=color_n,
                            node_size=ns, node_zorder=3, show=False,
                            close=False)
    # ax.set_title('Nodes used as stations in {}'.format(city))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap_name),
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbaxes = fig.add_axes([0.1, 0.075, 0.8, 0.03])
    cbar = fig.colorbar(sm, orientation='horizontal', cax=cbaxes)
    cbar.ax.tick_params(axis='x', labelsize=18)
    cbar.ax.set_xlabel('normalised usage of stations', fontsize=24)
    fig.suptitle('Nodes used as Stations in {}'.format(city),
                 fontsize=30)
    plt.savefig(plot_folder+'{:}_stations_used.png'.format(save),
                format=plot_format)

    plt.close('all')
    # plt.show()


def plot_edited_edges(city, save, G, edited_edges, bike_path_perc, node_size,
                      rev, minmode, plot_folder, plot_format='png'):
    print('Plotting edited edges.')
    plot_folder_evo = plot_folder+'evolution/'
    Path(plot_folder_evo).mkdir(parents=True, exist_ok=True)

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
        ee_cut = ee[:idx]
        for edge in ee_cut:
            G[edge[0]][edge[1]][0]['bike path'] = True
            G[edge[1]][edge[0]][0]['bike path'] = True
        ec = ['#0000FF' if data['bike path'] else '#999999' for
              u, v, data in G.edges(keys=False, data=True)]
        fig, ax = ox.plot_graph(G, node_size=node_size, node_color='C0',
                                edge_color=ec, fig_height=6, fig_width=6,
                                node_zorder=3, dpi=300, show=False,
                                close=False)
        fig.suptitle('Bike Path Percentage: {0:.0%} in {1:}'.format(blp[idx],
                                                                    city),
                     fontsize='x-large')
        plt.savefig(plot_folder_evo+'{0:s}-edited-mode-{1:d}{2:}-{3:d}'
                    '.{4:s}'.format(save, rev, minmode, i, plot_format),
                    format=plot_format)
        plt.close(fig)


def plot_mode(city, save, data, data_now, nxG, stations, trip_nbrs,
              mode, end, plot_folder, evo=False, plot_format='png'):
    rev = mode[0]
    minmode = mode[1]

    bp_now = data_now[0]
    cost_now = data_now[1]
    blp_now = data_now[2]
    trdt_now = data_now[3]
    tfdt_now = data_now[4]
    nos_now = data_now[5]

    # edited_edges = data[0]
    edited_edges_nx = data[1]
    cost = data[2]
    bike_path_perc = data[3]
    total_real_distance_traveled = data[4]
    total_felt_distance_traveled = data[5]
    nbr_on_street = data[6]
    # len_saved = data[7]
    # nbr_of_cbc = data[8]
    # gcbc_size = data[9]
    trdt, trdt_now = total_distance_traveled_list(total_real_distance_traveled,
                                                  trdt_now, rev)
    tfdt, tfdt_now = total_distance_traveled_list(total_felt_distance_traveled,
                                                  tfdt_now, rev)
    bl_st = len_of_bikepath_by_type(edited_edges_nx, nxG, rev)
    bl_st_now = len_of_bikepath_by_type(bp_now, nxG, rev)
    bl_st_now = {st: length[-1] for st, length in bl_st_now.items()}

    if rev:
        blp = bike_path_perc
    else:
        blp = list(reversed(bike_path_perc))
    trdt_min = min(trdt['all'])
    trdt_max = max(trdt['all'])
    ba = [1 - (i - trdt_min) / (trdt_max - trdt_min) for i in trdt['all']]
    ba_now = 1 - (trdt_now['all'] - trdt_min) / (trdt_max - trdt_min)

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
    blp_now = blp_now / blp[end]

    blp_x = min(blp_cut, key=lambda x: abs(x-blp_now))
    blp_idx = next(x for x, val in enumerate(blp_cut) if val == blp_x)
    ba_y = ba[blp_idx]

    cost_y = min(cost[:end], key=lambda x: abs(x-cost_now))
    cost_idx = next(x for x, val in enumerate(cost[:end]) if val == cost_y)
    cost_x = blp_cut[cost_idx]

    nos_y = min(nos[:end], key=lambda x: abs(x-nos_now))
    nos_idx = next(x for x, val in enumerate(nos[:end]) if val == nos_y)
    nos_x = blp_cut[nos_idx]

    los_y = min(los[:end], key=lambda x: abs(x-los_now))
    los_idx = next(x for x, val in enumerate(los[:end]) if val == los_y)
    los_x = blp_cut[los_idx]

    cut = next(x for x, val in enumerate(ba) if val >= 1)
    total_cost, cost_now = sum_total_cost(cost, cost_now, rev)

    cost_now = cost_now / total_cost[end]
    # gcbc_size_normed = [i / max(gcbc_size) for i in reversed(gcbc_size)]

    ns = [30 if n in stations else 0 for n in nxG.nodes()]

    print('Mode: {:d}{:d}, ba=1 after: {:d}, blp at ba=1: {:3.2f}, '
          'blp at cut: {:3.2f}, blp big roads: {:3.2f}, edges: {:}'
          .format(rev, minmode, cut, blp[cut], blp[end], blp_now * blp[end],
                  len(edited_edges_nx)))

    # Plotting
    fig1, ax1 = plt.subplots(dpi=300)
    ax12 = ax1.twinx()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax12.set_ylim(-0.05, 1.05)

    c_ba = 'C0'

    ax1.plot(blp_cut, ba[:end], c=c_ba, label='bikeability')
    ax1.plot(blp_now, ba_now, c=c_ba, marker='D')
    xmax, ymax = coord_transf(blp_now, max([ba_y, ba_now]))
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_ba, ls='--', alpha=0.5)
    ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    ax1.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)

    ax1.set_ylabel('bikeability', fontsize=12, color=c_ba)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=c_ba)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    c_cost = 'C8'

    ax12.plot(blp_cut, [x / total_cost[end] for x in total_cost[:end]],
              c=c_cost, label='total cost')
    ax12.plot(blp_now, cost_now, c=c_cost, marker='s')
    xmin, ymax = coord_transf(min(blp_now, cost_x), cost_now)
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_cost, ls='--', alpha=0.5)
    ax1.axhline(y=cost_now, xmax=1, xmin=xmin, c=c_cost, ls='--', alpha=0.5)
    ax1.axhline(y=cost_y, xmax=xmax, xmin=0, c=c_cost, ls='--', alpha=0.5)

    ax12.set_ylabel('cost', fontsize=12, color=c_cost)
    ax12.tick_params(axis='y', labelsize=12, labelcolor=c_cost)
    ax12.yaxis.set_minor_locator(AutoMinorLocator())

    ax1.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax1.set_title('Bikeability and Cost in {}'.format(city), fontsize=14)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='x', labelsize=12)

    handles = ax1.get_legend_handles_labels()[0]
    handles.append(ax12.get_legend_handles_labels()[0][0])
    ax1.legend(handles=handles, loc='lower right')

    plt.savefig(plot_folder+'{:}_ba_tc_mode_{:d}{:}.png'
                .format(save, rev, minmode), format=plot_format)

    ax1ins = zoomed_inset_axes(ax1, 3.5, loc=1)
    x1, x2, y1, y2 = round(blp_now - 0.05, 2), round(blp_now + 0.05, 2), \
                     round(ba_now - 0.03, 2), round(ba_y + 0.03, 2)
    ax1ins.plot(blp_cut, ba[:end])
    ax1ins.plot(blp_now, ba_now, c=c_ba, marker='D')
    xmax, ymax = coord_transf(blp_now, max([ba_y, ba_now]), xmin=x1, xmax=x2,
                              ymin=y1, ymax=y2)
    ax1ins.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)

    ax1ins.set_xlim(x1, x2)
    ax1ins.set_ylim(y1, y2)
    ax1ins.tick_params(axis='y', labelsize=8, labelcolor=c_ba)
    ax1ins.tick_params(axis='x', labelsize=8)
    ax1ins.yaxis.set_minor_locator(AutoMinorLocator())
    ax1ins.xaxis.set_minor_locator(AutoMinorLocator())
    # ax1ins.set_yticklabels(labels=ax1ins.get_yticklabels(), visible=False)
    # ax1ins.set_xticklabels(labels=ax1ins.get_xticklabels(), visible=False)
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")

    plt.savefig(plot_folder+'{:}_ba_tc_zoom_mode_{:d}{:}.png'
                .format(save, rev, minmode), format=plot_format)

    fig2, ax2 = plt.subplots(dpi=300)
    ax22 = ax2.twinx()
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax22.set_ylim(-0.05, 1.05)

    c_nos = 'C1'
    c_los = 'm'

    p1, = ax2.plot(blp_cut, los[:end], label='length', c=c_los)
    ax2.plot(blp_now, los_now, c=c_los, marker='8')
    xmax, ymax = coord_transf(max(blp_now, los_x), los_now)
    ax2.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5)
    ax2.axhline(y=los_now, xmax=xmax, xmin=0, c=c_los, ls='--', alpha=0.5)
    ax2.axvline(x=los_x, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5)

    ax2.set_ylabel('length of trips', fontsize=12, color=c_los)
    ax2.tick_params(axis='y', labelsize=12, labelcolor=c_los)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    p2, = ax22.plot(blp_cut, nos[:end], label='trips', c=c_nos)
    ax22.plot(blp_now, nos_now, c=c_nos, marker='v')
    xmin, ymax = coord_transf(min(blp_now, nos_x), nos_now)
    ax22.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5)
    ax22.axhline(y=nos_now, xmax=1, xmin=xmin, c=c_nos, ls='--', alpha=0.5)
    ax22.axvline(x=nos_x, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5)

    ax22.set_ylabel('number of trips', fontsize=12, color=c_nos)
    ax22.tick_params(axis='y', labelsize=12, labelcolor=c_nos)
    ax22.yaxis.set_minor_locator(AutoMinorLocator())

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='x', labelsize=12)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax2.set_title('Number of Trips and Length on Street in {}'.format(city),
                  fontsize=14)
    ax2.legend([p1, p2], [l.get_label() for l in [p1, p2]])
    plt.savefig(plot_folder+'{:}_trips_on_street_mode_{:d}{:}.png'.
                format(save, rev, minmode), format=plot_format)

    fig3, ax3 = plt.subplots(dpi=300)
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

    ax3.set_xlabel('% of bike paths by length', fontsize=12)
    ax3.set_ylabel('length', fontsize=12)
    ax3.set_title('Length on Street in {}'.format(city), fontsize=14)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.legend()
    plt.savefig(plot_folder+'{:}_len_on_street_mode_{:d}{:}.png'
                .format(save, rev, minmode), format=plot_format)

    fig4, ax4 = plt.subplots(dpi=300)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)

    """for st, len_by_st_now in bl_st_now.items():
        xmax, ymax = coord_transf(blp_now, len_by_st_now)
        ax4.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_st[st], ls='--',
                    alpha=0.5)
        ax4.axhline(y=len_by_st_now, xmax=xmax, xmin=0, c=c_st[st], ls='--',
                    alpha=0.5)"""
    for st, len_by_st in bl_st.items():
        ax4.plot(blp_cut, [x / len_by_st[end] for x in len_by_st[:end]],
                 c=c_st[st], label='{}'.format(st))
    """for st, len_by_st in bl_st.items():
        ax4.plot(blp_now, bl_st_now[st], marker=m_st[st],  c=c_st[st])"""

    ax4.axvline(x=blp[cut]/blp[end], c='#999999', ls='--', alpha=0.5)

    ax4.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax4.set_ylabel('length', fontsize=12)
    ax4.set_title('Length of Bike Paths along Streets in {}'.format(city),
                  fontsize=14)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    ax4.legend()
    plt.savefig(plot_folder+'{:}_len_bl_mode_{:d}{:}.png'
                .format(save, rev, minmode), format=plot_format)

    plot_used_nodes(city=city, save=save, G=nxG, trip_nbrs=trip_nbrs,
                    stations=stations, plot_folder=plot_folder,
                    plot_format='png')

    if evo:
        plot_edited_edges(city=city, save=save, G=nxG,
                          edited_edges=edited_edges_nx,
                          bike_path_perc=bike_path_perc, node_size=ns,
                          rev=rev, minmode=minmode, plot_folder=plot_folder,
                          plot_format='png')

    # plt.show()
    plt.close('all')

    return blp_cut, ba[:end], total_cost[:end], nos[:end], los[:end], \
           blp_now, ba_now, cost_now, nos_now, los_now, blp[cut], trdt_min, \
           trdt_max


def compare_modes(city, save, label, blp, ba, cost, nos, los, blp_now, ba_now,
                  cost_now, nos_now, los_now, color, plot_folder,
                  plot_format='png'):
    fig1, ax1 = plt.subplots(dpi=300)
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax1.set_ylabel('bikeability', fontsize=12)
    ax1.set_title('Bikeability of {}'.format(city),
                  fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(bottom=-0.05, top=1.05)

    fig2, ax2 = plt.subplots(dpi=300)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax2.set_ylabel('integrated bikeability', fontsize=12)
    ax2.set_title('Integrated Bikeability of {}'.format(city),
                  fontsize=14)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(bottom=-0.05, top=1.05)

    fig3, ax3 = plt.subplots(dpi=300)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax3.set_ylabel('ba per cost', fontsize=12)
    ax3.set_title('CBA of {}'.format(city), fontsize=14)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.set_ylim(bottom=-0.05, top=1.05)

    fig4, ax4 = plt.subplots(dpi=300)
    ax42 = ax4.twinx()
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax42.set_ylim(-0.05, 1.05)

    ax4.set_ylabel('length on street', fontsize=12, color='orangered')
    ax4.tick_params(axis='y', labelsize=12, labelcolor='orangered')
    ax4.yaxis.set_minor_locator(AutoMinorLocator())

    ax42.set_ylabel('number of trips', fontsize=12, color='mediumblue')
    ax42.tick_params(axis='y', labelsize=12, labelcolor='mediumblue')
    ax42.yaxis.set_minor_locator(AutoMinorLocator())

    ax4.set_title('Trips and Length on Street in {}'.format(city), fontsize=14)
    ax4.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())

    ax4_hand = {}

    for m in blp.keys():
        bikeab = [np.trapz(ba[m][:idx], blp[m][:idx]) for idx in
                  range(len(blp[m]))]
        cba = [bikeab[idx] / cost[m][idx] for idx in
               range(len(blp[m]))]
        ax1.plot(blp[m], ba[m], color=color[m], label=label[m])
        ax2.plot(blp[m], bikeab, color=color[m],  label=label[m])
        ax3.plot(blp[m], cba, color=color[m], label=label[m])
        space = round(len(blp[m]) / 20)
        ax4.plot(blp[m], nos[m], color=color[m], marker='v',  markevery=space,
                 label=label[m])
        ax42.plot(blp[m], los[m], color=color[m], marker='8', markevery=space,
                  label=label[m])
        ax4_hand[m] = mlines.Line2D([], [], color=color[m], label=label[m])

    ax1.plot(blp_now, ba_now, c='#999999', marker='D')
    xmax, ymax = coord_transf(blp_now, ba_now)
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    ax4.plot(blp_now, nos_now, c='#999999', marker='v')
    ax4.plot(blp_now, los_now, c='#999999', marker='8')
    xmax, ymax = coord_transf(blp_now, nos_now)
    ax4.axvline(x=blp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    ax4.axhline(y=nos_now, xmax=1, xmin=xmax, c='#999999', ls=':', alpha=0.5)
    ax4.axhline(y=los_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    l_keys_r = []
    l_keys_b = []
    for mode, l_key in ax4_hand.items():
        if not mode[0]:
            l_keys_r.append(l_key)
        else:
            l_keys_b.append(l_key)

    l_keys = [tuple(l_keys_r)] + [tuple(l_keys_b)]
    l_labels = ['Removing', 'Building']

    ax1.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})
    ax2.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})
    ax3.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    l_keys.append(mlines.Line2D([], [], color='k', marker='v', label='trips'))
    l_labels.append('trips')
    l_keys.append(mlines.Line2D([], [], color='k', marker='8', label='length'))
    l_labels.append('length')
    ax4.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc=1)

    fig1.savefig(plot_folder+'{}-1.png'.format(save), format=plot_format)
    fig2.savefig(plot_folder+'{}-2.png'.format(save), format=plot_format)
    fig3.savefig(plot_folder+'{}-3.png'.format(save), format=plot_format)
    fig4.savefig(plot_folder+'{}-4.png'.format(save), format=plot_format)

    plt.close('all')
    # plt.show()


def plot_city(city, save, input_folder, output_folder, comp_folder,
              plot_folder, modes, comp_modes=False, bike_paths=None,
              plot_evo=False, evo_for=None, plot_format='png'):
    if evo_for is None:
        evo_for = [(False, 1)]

    Path(comp_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    trip_nbrs = np.load(input_folder+'{}_demand.npy'.format(save),
                        allow_pickle=True)[0]
    trip_nbrs_re = {trip_id: nbr_of_trips for trip_id, nbr_of_trips
                    in trip_nbrs.items() if not trip_id[0] == trip_id[1]}
    trips = sum(trip_nbrs.values())
    trips_re = sum(trip_nbrs_re.values())
    utrips = len(trip_nbrs.keys())
    utrips_re = len(trip_nbrs_re.keys())

    stations = [station for trip_id, nbr_of_trips in trip_nbrs.items() for
                station in trip_id]
    stations = set(stations)

    print('City: {}, stations: {},trips: {} (rt excl.: {}), '
          'unique trips: {} (rt excl. {})'
          .format(city, len(stations), trips, trips_re, utrips, utrips_re))

    nxG = ox.load_graphml(filename='{}.graphml'.format(save),
                          folder=input_folder, node_type=int)
    nxG = nxG.to_undirected()
    st_len = get_street_type_ratio(nxG)
    print(st_len)

    data_now = calc_current_state(nxG, trip_nbrs, bike_paths)

    data = {}
    for m in modes:
        data[m] = np.load(output_folder+'{:}_data_mode_{:d}{:}.npy'
                          .format(save, m[0], m[1]),
                          allow_pickle=True)

    end = max([get_end(d[4], data_now[3], m[0]) for m, d in data.items()])
    print('Cut after: ', end)

    c_norm = ['darkblue', 'mediumblue', 'cornflowerblue']
    c_rev = ['red', 'orangered', 'indianred']

    c = {}
    for m, d in data.items():
        if m[0]:
            c[m] = c_rev[m[1]]
        else:
            c[m] = c_norm[m[1]]

    blp = {}
    ba = {}
    cost = {}
    nos = {}
    los = {}
    blp_cut = {}
    trdt_min = {}
    trdt_max = {}
    blp_now, ba_now, cost_now, nos_now, los_now = 0, 0, 0, 0, 0

    for m, d in data.items():
        if plot_evo and (m in evo_for):
            evo = True
        else:
            evo = False
        blp[m], ba[m], cost[m], nos[m], los[m], blp_now, ba_now, cost_now, \
        nos_now, los_now, blp_cut[m], trdt_min[m], trdt_max[m] = \
            plot_mode(city=city, save=save, data=d, data_now=data_now, nxG=nxG,
                      stations=stations, trip_nbrs=trip_nbrs, mode=m, end=end,
                      evo=evo, plot_folder=plot_folder, plot_format='png')

    data_to_save = [blp, ba, cost, nos, los, blp_cut, trdt_min, trdt_max]
    np.save(comp_folder+'ba_comp_{}.npy'.format(save), data_to_save)

    if comp_modes:
        label = {m: 'Removing' if not m[0] else 'Building' for m in modes}
        compare_modes(city=city, save=save, label=label, blp=blp, ba=ba,
                      cost=cost, nos=nos, los=los, blp_now=blp_now,
                      ba_now=ba_now, cost_now=cost_now, nos_now=nos_now,
                      los_now=los_now, color=c, plot_folder=plot_folder,
                      plot_format=plot_format)


def compare_cities(cities, saves, mode, color, data_folder, plot_folder,
                   figsize=None, plot_format='png'):
    if figsize is None:
        figsize = [12, 9]
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    rev = mode[0]
    minmode = mode[1]

    blp = {}
    ba = {}
    cost = {}
    nos = {}
    los = {}
    blp_cut = {}
    trdt_min = {}
    trdt_max = {}
    trdt_rat = {}


    for city in cities:
        save = saves[city]
        data = np.load(data_folder+'ba_comp_{}.npy'.format(save),
                       allow_pickle=True)
        blp[city] = data[0][mode]
        ba[city] = data[1][mode]
        cost[city] = data[2][mode]
        nos[city] = data[3][mode]
        los[city] = data[4][mode]
        blp_cut[city] = data[5][mode]
        trdt_min[city] = data[6][mode]
        trdt_max[city] = data[7][mode]
        trdt_rat[city] = data[7][mode] / data[6][mode]

    fig1, ax1 = plt.subplots(dpi=150, figsize=figsize)
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax1.set_ylabel('bikeability', fontsize=12)
    ax1.set_title('Comparison of Bikeabilities', fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(bottom=-0.05, top=1.05)

    fig2, ax2 = plt.subplots(dpi=150, figsize=figsize)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax2.set_ylabel('integrated bikeability', fontsize=12)
    ax2.set_title('Comparison of Integrated Bikeabilities', fontsize=14)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(bottom=-0.05, top=1.05)

    fig3, ax3 = plt.subplots(dpi=150, figsize=figsize)
    ax32 = ax3.twinx()
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax32.set_ylim(-0.05, 1.05)

    ax3.set_ylabel('number of trips', fontsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.yaxis.set_minor_locator(AutoMinorLocator())

    ax32.set_ylabel('length on street', fontsize=12)
    ax32.tick_params(axis='y', labelsize=12)
    ax32.yaxis.set_minor_locator(AutoMinorLocator())

    ax3.set_title('Comaprison of Trips and Length on Street', fontsize=14)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())

    ax3_hand = {}

    for city in cities:
        bikeab = [np.trapz(ba[city][:idx], blp[city][:idx]) for idx in
                  range(len(blp[city]))]
        ax1.plot(blp[city], ba[city], color=color[city],
                 label='{}, fmax = {:3.2f}, trdt ratio = {:3.2f}'
                 .format(city, blp_cut[city], trdt_rat[city]))
        ax2.plot(blp[city], bikeab, color=color[city],
                 label='{}'.format(city))
        space = round(len(blp[city]) / 25)

        ax3.plot(blp[city], nos[city], marker='v', markevery=space,
                 color=color[city])
        ax32.plot(blp[city], los[city], marker='8', markevery=space,
                  color=color[city])
        ax3_hand[city] = mlines.Line2D([], [], color=color[city])

    l_keys = [l_key for city, l_key in ax3_hand.items()]
    l_cities = [city for city, l_key in ax3_hand.items()]
    l_keys.append(mlines.Line2D([], [], color='k', marker='v', label='trips'))
    l_cities.append('trips')
    l_keys.append(mlines.Line2D([], [], color='k', marker='8', label='length'))
    l_cities.append('length')

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax3.legend(l_keys, l_cities, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    fig1.savefig(plot_folder+'comparison-1-{:d}{:}.{}'.format(rev, minmode,
                                                              plot_format),
                 format=plot_format)
    fig2.savefig(plot_folder+'comparison-2-{:d}{:}.{}'.format(rev, minmode,
                                                              plot_format),
                 format=plot_format)
    fig3.savefig(plot_folder+'comparison-3-{:d}{:}.{}'.format(rev, minmode,
                                                               plot_format),
                 format=plot_format)

    plt.close('all')
    # plt.show()
