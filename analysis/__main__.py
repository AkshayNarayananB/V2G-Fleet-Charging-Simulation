"""Plots for the paper:

M. Yuhas, R. K. Ahir, L. V. T. Hartono, M. D. D. Putranto, and A. Easwaran,
``Managing Charging Induced Grid Stress and Battery Degradation in Electric
Taxi Fleets,'' in 2025 IEEE Innovative Smart Grid Technologies - Asia
(ISGT-Asia), Guangzhou, China, Oct. 2025.

This can serve as a starting point for analyzing simulator results.
"""
from typing import List

import argparse
import csv
import statistics

import numpy
import matplotlib as mpl

from matplotlib import pyplot as plt



def plot_battery_degradation(logs: List[str], fleet_size: int, dt: int) -> None:
    """Plot battery degradation for all vehicles in a fleet.
    
    Args:
        logs: list of log files to plot
        fleet_size: number of vehicles in the fleet
        dt: simulator tick time (seconds)
    """
    soh_max = {}
    soh_min = {}
    soh_med = {}
    t = {}
    labels = []
    colors = mpl.color_sequences['tab10']
    fig, ax = plt.subplots()
    for i, log_f in enumerate(logs):
        with open(log_f, 'r') as csvfile:
            soh_max[log_f] = []
            soh_min[log_f] = []
            soh_med[log_f] = []
            reader = csv.DictReader(csvfile)
            for idx, datum in enumerate(reader):
                if idx % (24 * 7 * dt / 3600) == 0:
                    soh = []
                    for v in range(fleet_size):
                        soh.append(float(datum[f"soh{v}"]) / 72.1)
                    soh_max[log_f].append(numpy.percentile(numpy.array(soh), 75))
                    soh_min[log_f].append(numpy.percentile(numpy.array(soh), 25))
                    soh_med[log_f].append(numpy.percentile(numpy.array(soh), 50))
            t[log_f] = numpy.linspace(0, len(soh_med[log_f]) / 52, len(soh_med[log_f]))
            ax.fill_between(
                t[log_f],
                numpy.array(soh_min[log_f]),
                numpy.array(soh_max[log_f]),
                alpha=0.5,
                facecolor=colors[i],
                label='_nolegend_'
            )
            ax.plot(t[log_f], numpy.array(soh_med[log_f]), color=colors[i])
            labels.append(log_f)
    ax.set_xlabel('Years')
    ax.set_ylabel('State of Health $\\bar{Q}_v(t)/\\bar{Q}_v(0)$')
    ax.legend(labels)
    fig.tight_layout()
    plt.show()

def plot_revenue(logs: List[str], dt: int) -> None:
    revenue = {}
    t = {}
    labels = []
    colors = mpl.color_sequences['tab10']
    fig, ax = plt.subplots()
    for i, log_f in enumerate(logs):
        with open(log_f, 'r') as csvfile:
            revenue[log_f] = []
            reader = csv.DictReader(csvfile)
            r = 0
            for idx, datum in enumerate(reader):
                r += float(datum["profit"])
                if idx % (24 * 7 * dt / 3600) == 0:
                    revenue[log_f].append(r)
            t[log_f] = numpy.linspace(0, len(revenue[log_f]) / 52, len(revenue[log_f]))
            ax.plot(t[log_f], numpy.array(revenue[log_f]), color=colors[i])
            labels.append(log_f)
    ax.set_xlabel('Years')
    ax.set_ylabel('Cumulative Revenue ($)')
    ax.legend(labels)
    fig.tight_layout()
    fig.show()


def plot_charge_power_over_time(logs: List[str], week: int, day: int, dt: int) -> None:
    ch_pwr = {}
    colors = mpl.color_sequences['tab10']
    fig, ax = plt.subplots()
    ticks_per_hr = int(dt / 3600)
    t = numpy.linspace(0, 7, 24 * 7 * ticks_per_hr)
    labels = []
    for i, log_f in enumerate(logs):
        with open(log_f, 'r') as csvfile:
            ch_pwr[log_f] = []
            reader = csv.DictReader(csvfile)
            for _ in range((24 * 7 * week + 24 * day) * ticks_per_hr):
                next(reader)
            for _ in range(24 * 7 * ticks_per_hr):
                datum = next(reader)
                ch_pwr[log_f].append(float(datum['total_power']))
        ax.plot(t, ch_pwr[log_f], color=colors[i])
        labels.append(log_f)
    ax.set_xlabel('Days')
    ax.set_ylabel('Total Fleet Charging Power (kWh)')
    ax.legend(labels)
    fig.tight_layout()
    plt.show()


def plot_charge_power_distribution(logs: List[str], dt: int) -> None:
    ch_pwr = {}
    fig, ax = plt.subplots()
    labels = []
    handles = []
    for i, log_f in enumerate(logs):
        with open(log_f, 'r') as csvfile:
            ch_pwr[log_f] = []
            reader = csv.DictReader(csvfile)
            for datum in reader:
                ch_pwr[log_f].append(float(datum['total_power']))
            parts = ax.violinplot(
                [ch_pwr[log_f]],
                positions=[i],
                widths=0.75,
                vert=False,
                showmeans=False,
                showextrema=False,
                showmedians=False
            )
            handles.append(parts['bodies'][0])
            labels.append(log_f)
    ax.set_xlabel('Instantaneous Charging Power (kW)')
    ax.legend(handles, labels)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Comapare scheduling algorithms.")
    parser.add_argument(
        '-l',
        '--log-files',
        nargs='+',
        help='List of logs files (csv) from different scheduling algorithms.',
        required=True
    )
    parser.add_argument(
        '-f',
        '--fleet-size',
        type=int,
        help='Size of fleet',
        required=True
    )
    parser.add_argument(
        '--dt',
        type=int,
        help='Simulation tick time (seconds).',
        required=True
    )
    parser.add_argument(
        '--plot-battery-degradation',
        action='store_true',
        help='Plot battery degradation'
    )
    parser.add_argument(
        '--plot-revenue',
        action='store_true',
        help='Plot revenue'
    )
    parser.add_argument(
        '--plot-charge-power-distribution',
        action='store_true',
        help='Plot distribution of charging power'
    )
    parser.add_argument(
        '--plot-charge-power-over-time',
        action='store_true',
        help='Plot charging power over time for one week (Needs --week and --day)'
    )
    parser.add_argument(
        '--week',
        type=int,
        help='# of weeks into simulation to plot charge power over time'
    )
    parser.add_argument(
        '--day',
        type=int,
        help='Day of the week offset for charge power over time'
    )
    args = parser.parse_args()
    if args.plot_battery_degradation:
        plot_battery_degradation(args.log_files, args.fleet_size, args.dt)
    if args.plot_revenue:
        plot_revenue(args.log_files, args.dt)
    if args.plot_charge_power_distribution:
        plot_charge_power_distribution(args.log_files, args.dt)
    if args.plot_charge_power_over_time:
        plot_charge_power_over_time(args.log_files, args.week, args.day, args.dt)
