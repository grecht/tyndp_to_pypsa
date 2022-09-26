"""
This module contains code for loading PyPSA-Eur's grid topology based on the
data in this directory. Code extracted and modified from:
https://github.com/PyPSA/pypsa-eur/blob/master/scripts/base_network.py
"""
import os
import pandas as pd
import numpy as np
from scipy import spatial
import shapely.wkt
from shapely.geometry import LineString


_FILE_DIR = os.path.dirname(__file__)
_GRIDKIT_DIR = os.path.join(_FILE_DIR, 'entsoegridkit/')


def import_gridx_buses():
    buses_file = os.path.join(_GRIDKIT_DIR, 'buses.csv')
    return (pd.read_csv(buses_file,
                        quotechar="'",
                        true_values=['t'],
                        false_values=['f'],
                        dtype=dict(bus_id="str"))
            .set_index("bus_id")
            .drop(['station_id'], axis=1)
            .rename(columns=dict(voltage='v_nom')))


def import_gridx_lines(gridx_buses):
    lines_file = os.path.join(_GRIDKIT_DIR, 'lines.csv')
    gridx_lines = (pd.read_csv(lines_file,
                               quotechar="'",
                               true_values=['t'],
                               false_values=['f'],
                               dtype=dict(line_id='str',
                                          bus0='str',
                                          bus1='str',
                                          underground="bool",
                                          under_construction="bool"))
                   .set_index('line_id')
                   .rename(columns=dict(voltage='v_nom',
                                        circuits='num_parallel')))

    gridx_lines['length'] /= 1e3
    return gridx_lines.loc[gridx_lines.bus0.isin(gridx_buses.index)
                           & gridx_lines.bus1.isin(gridx_buses.index)]


def import_gridx_links(gridx_buses):
    links_file = os.path.join(_GRIDKIT_DIR, 'links.csv')
    gridx_links = (pd.read_csv(links_file,
                               quotechar="'",
                               true_values=['t'],
                               false_values=['f'],
                               dtype=dict(link_id='str',
                                          bus0='str',
                                          bus1='str',
                                          under_construction="bool"))
                   .set_index('link_id'))
    gridx_links['length'] /= 1e3
    # Skagerrak Link is connected to 132kV bus which is removed
    # in_load_buses_from_eg. Connect to neighboring 380kV bus
    gridx_links.loc[gridx_links.bus1 == '6396', 'bus1'] = '6398'
    gridx_links = gridx_links.loc[gridx_links.bus0.isin(gridx_buses.index)
                                  & gridx_links.bus1.isin(gridx_buses.index)]

    # merge with TYNDP links already imported in PyPSA-Eur
    return _add_links_from_tyndp(gridx_buses, gridx_links)


def _add_links_from_tyndp(buses, links):
    links_tyndp_file = os.path.join(_FILE_DIR, 'links_tyndp.csv')
    links_tyndp = pd.read_csv(links_tyndp_file)

    has_replaces_b = links_tyndp.replaces.notnull()
    oids = dict(Bus=_get_oid(buses), Link=_get_oid(links))
    keep_b = dict(
        Bus=pd.Series(True, index=buses.index),
        Link=pd.Series(True, index=links.index)
    )
    for reps in links_tyndp.loc[has_replaces_b, "replaces"]:
        for comps in reps.split(":"):
            oids_to_remove = comps.split(".")
            c = oids_to_remove.pop(0)
            keep_b[c] &= ~oids[c].isin(oids_to_remove)
    buses = buses.loc[keep_b["Bus"]]
    links = links.loc[keep_b["Link"]]

    links_tyndp["j"] = _find_closest_links(
        links, links_tyndp, distance_upper_bound=0.20
    )
    # Corresponds approximately to 20km tolerances

    if links_tyndp["j"].notnull().any():
        links_tyndp = links_tyndp.loc[links_tyndp["j"].isnull()]
        if links_tyndp.empty:
            return buses, links

    tree = spatial.KDTree(buses[["x", "y"]])
    _, ind0 = tree.query(links_tyndp[["x1", "y1"]])
    ind0_b = ind0 < len(buses)
    links_tyndp.loc[ind0_b, "bus0"] = buses.index[ind0[ind0_b]]

    _, ind1 = tree.query(links_tyndp[["x2", "y2"]])
    ind1_b = ind1 < len(buses)
    links_tyndp.loc[ind1_b, "bus1"] = buses.index[ind1[ind1_b]]

    links_tyndp_located_b = (
        links_tyndp["bus0"].notnull() & links_tyndp["bus1"].notnull()
    )
    if not links_tyndp_located_b.all():
        links_tyndp = links_tyndp.loc[links_tyndp_located_b]

    links_tyndp = links_tyndp[["bus0", "bus1"]].assign(
        carrier="DC",
        p_nom=links_tyndp["Power (MW)"],
        length=links_tyndp["Length (given) (km)"].fillna(
            links_tyndp["Length (distance*1.2) (km)"]
        ),
        under_construction=True,
        underground=False,
        geometry=(
            links_tyndp[["x1", "y1", "x2", "y2"]].apply(
                lambda s: str(LineString([[s.x1, s.y1], [s.x2, s.y2]])), axis=1
            )
        ),
        tags=(
            '"name"=>"'
            + links_tyndp["Name"]
            + '", '
            + '"ref"=>"'
            + links_tyndp["Ref"]
            + '", '
            + '"status"=>"'
            + links_tyndp["status"]
            + '"'
        ),
    )

    links_tyndp.index = "T" + links_tyndp.index.astype(str)

    links = pd.concat([links, links_tyndp], sort=True)

    return buses, links


def _get_oid(df):
    if "tags" in df.columns:
        return df.tags.str.extract('"oid"=>"(\d+)"', expand=False)
    else:
        return pd.Series(np.nan, df.index)


def _find_closest_links(links, new_links, distance_upper_bound=1.5):
    treecoords = np.asarray(
        [
            np.asarray(shapely.wkt.loads(s).coords)[[0, -1]].flatten()
            for s in links.geometry
        ]
    )
    querycoords = np.vstack(
        [new_links[["x1", "y1", "x2", "y2"]], new_links[["x2", "y2", "x1", "y1"]]]
    )
    tree = spatial.KDTree(treecoords)
    dist, ind = tree.query(querycoords, distance_upper_bound=distance_upper_bound)
    found_b = ind < len(links)
    found_i = np.arange(len(new_links) * 2)[found_b] % len(new_links)
    return (
        pd.DataFrame(
            dict(D=dist[found_b], i=links.index[ind[found_b] % len(links)]),
            index=new_links.index[found_i],
        )
        .sort_values(by="D")[lambda ds: ~ds.index.duplicated(keep="first")]
        .sort_index()["i"]
    )
