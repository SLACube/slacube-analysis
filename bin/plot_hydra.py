#!/usr/bin/env python

import os
import fire
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def find_arrow(next_node):
    for i, chip_id in enumerate(next_node):
        if chip_id is None:
            continue
        return i
    return -1

def draw_right_arrow(ax, x, y, color):
    ax.arrow(
        x+0.03, y+0.01, 0.04, 0,
        color=color,
        length_includes_head=True,
        head_width=0.01,
        shape='right',
    )
    
def draw_left_arrow(ax, x, y, color):
    ax.arrow(
        x-0.03, y-0.01, -0.04, 0,
        color=color,
        length_includes_head=True,
        head_width=0.01,
        shape='right',
    )
    
def draw_down_arrow(ax, x, y, color):
    ax.arrow(
        x-0.01, y-0.03, 0, -0.04,
        color=color,
        length_includes_head=True,
        head_width=0.01,
        shape='left',
    )

def draw_up_arrow(ax, x, y, color):
    ax.arrow(
        x+0.01, y+0.03, 0, 0.04,
        color=color,
        length_includes_head=True,
        head_width=0.01,
        shape='left',
    )
    
draw_arrow = [
    draw_up_arrow,
    draw_left_arrow,
    draw_down_arrow,
    draw_right_arrow
]

def main(fpath, outfile, io_group=1):
    mpl.use('agg')
    sns.set_theme('talk', 'white')

    # ==================================
    # Parse tile cfg
    # ==================================
    with open(fpath, 'r') as f:
        cfg = json.load(f)

    net = cfg['network']
    subnet = net.get(str(io_group), {})

    paths = np.zeros((10,10), dtype=int)
    arrows = np.full_like(paths, -1)
    for key, nodes_dict in subnet.items():
        io_channel = int(key)

        nodes = nodes_dict.get('nodes', {})
        for node in nodes[1:]:
            i, j = divmod(node['chip_id']-11, 10)
            k = find_arrow(node['miso_us'])
            arrows[i, j] = k
            paths[i, j] = io_channel
            
    # ===================================
    # plot hydra network
    # ===================================
    grids = np.linspace(0, 1, 11)
    mid = (grids[1:] + grids[:-1]) / 2

    colors = ['black', '#0072b2', '#d55e00', '#009e73', '#cc79a7']
    ROOT_CHIPS = [11,41,71,101]

    fig, ax = plt.subplots(figsize=(12, 12))
    for chip in range(11, 111):
        i, j = divmod(chip-11, 10)
        x = mid[j]
        y = 1 - mid[i]
        
        color = colors[paths[i,j]]
        arrow = arrows[i,j]
        
        ax.text(
            x, y, f'{chip}', 
            va='center', ha='center',
            size='large',
            color=color,
            weight='extra bold' if chip in ROOT_CHIPS else 'regular',
        )
        
        if arrow != -1:
            draw_arrow[arrow](ax, x, y, color)
            
    for chip in ROOT_CHIPS:
        i, j = divmod(chip-11, 10)
        x = grids[j] + 0.005
        y = 1 - grids[i] - 0.005
        
        io_channel = paths[i,j]
        if io_channel == 0:
            continue
            
        ax.text(
            x, y, io_channel,
            va='top', ha='left',
            color=colors[io_channel],
        )

    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.grid(linestyle='--')

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_aspect('equal')
    fig.tight_layout()

    fig.savefig(outfile)

if __name__ == '__main__':
    fire.Fire(main)
