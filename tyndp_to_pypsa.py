import sys
import warnings
import datetime
import re
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from scipy import spatial

_IGNORE_PROJECTS = [
       335  # North Sea Wind Power Hub
]

# TODO: is 'investment_id' unique? Otherwise, we must check also check 'project_id'
_IGNORE_INVESTMENTS = [
       1652  # line from new Bodelwyddan converter station to Bodelwyddan 2 ('project_id'=349)
]

__SPLIT_REGEX = r'("\w+"\s*=>\s*"[^"]*"),'
_TAG_REGEX = r'"(?P<key>\w+)"\s*=>\s*"(?P<value>[^"]*)"'  # Form: '"key"=>"value"'
_TAG_PATTERN = re.compile(_TAG_REGEX)


def _tags_to_dict(row):
    """
    Convert 'tags' column containing comma-separated entries
    of the form '"key"=>"value"' to a dictionary.

    The 'tags' column in the data extracted from the ENTSO-E
    interactive grid map, which PyPSA-Eur's grid topology is based
    on, contains strings of this form. This is the PostgreSQL hstore
    format. For further info, see following links.

    https://github.com/PyPSA/pypsa-eur/tree/master/data/entsoegridkit
    https://github.com/bdw/GridKit/tree/master/entsoe
    https://www.postgresql.org/docs/current/hstore.html
    """
    if row.tags is np.nan:
        return {}
    tags = list(filter(None, [s.strip() for s in re.split(__SPLIT_REGEX,
                                                          row.loc['tags'])]))
    return dict(_TAG_PATTERN.match(t).groups() for t in tags)


def _dict_to_tags(d):
    """
    Convert dictionary to string of comma-separated entries
    of the form '"key"=>"value"'.
    """
    return ', '.join([f'"{k}"=>"{v}"' for k,v in d.items()])


def _cols_to_tags(cols, row):
    """
    Convert all values in 'row' from a column in 'col'
    to a string of comma-separated entries of the form
    '"key"=>"value"'.
    """
    return ', '.join([f'"{col}"=>"{row[col]}"' for col in cols if col in row.index])


def _splitinto_buses_lines_links(df):
    # Split into lines and buses
    tyndp_buses = df.loc[df.asset_type == 'substation']
    tyndp_lines = df.loc[df.asset_type == 'line']

    # Ignore entries with missing coordinates (various reasons:
    # incomplete project specification, outside of PyPSA-Eur area,...)
    tyndp_buses = tyndp_buses.dropna(subset=['x1', 'y1'])
    tyndp_lines_links = tyndp_lines.dropna(subset=['x1', 'y1', 'x2', 'y2'])

    tyndp_lines = tyndp_lines_links.loc[tyndp_lines_links.ac_dc == 'AC']
    tyndp_links = tyndp_lines_links.loc[tyndp_lines_links.ac_dc == 'DC']

    # Extract all buses occuring in 'tyndp_lines'
    buses_1 = tyndp_lines.loc[:, ('substation_1', 'x1', 'y1')]
    buses_1.columns = ['substation', 'x', 'y']

    buses_2 = tyndp_lines.loc[:, ('substation_2', 'x2', 'y2')]
    buses_2.columns = ['substation', 'x', 'y']

    all_buses = pd.concat([buses_1, buses_2])
    all_buses = all_buses.drop_duplicates()

    # check if there are substations with same name but different coordinates
    duplicates = all_buses.duplicated(subset=['substation'], keep=False)
    if not all_buses.loc[duplicates, :].empty:
        s = 'There are substations with different coordinate values:\n'
        s += str(all_buses.loc[duplicates, :].sort_values('substation'))
        raise ValueError(s)

    return tyndp_buses, tyndp_lines, tyndp_links


def _buses_to_pypsa(tyndp_buses):
    """
    Transform buses from TYNDP to PyPSA-Eur format.
    """
    tyndp_buses = tyndp_buses.drop([
        'substation_2',
        'x2',
        'y2',
        'asset_type',
        'specified_length_km',
        'underground',
        'p_nom_max'],
        axis=1)

    tyndp_buses = tyndp_buses.rename(columns={
        'substation_1': 'name',
        'x1': 'x',
        'y1': 'y',
        'voltage': 'v_nom',
        'status': 'tyndp_status',
        'project_id': 'tyndp2020_proj_id',
        'investment_id': 'tyndp2020_invest_id'
    })

    tyndp_buses['dc'] = tyndp_buses.ac_dc.map({'AC': False, 'DC': True})
    tyndp_buses = tyndp_buses.drop('ac_dc', axis=1)

    # TODO: extract substation name and country to 'tags' column,
    # set 'symbol' column to "Substation"

    reg = r'(?P<name>.+)\s?[\[(](?P<country>\w{2})[)\]]'
    pat = re.compile(reg)

    def extract_name(val):
        m = pat.match(val)
        return m.group('name').strip() if m else val.strip()

    def extract_country(val):
        m = pat.match(val)
        return m.group('country').strip() if m else np.nan

    tyndp_buses['country'] = tyndp_buses['name'].apply(extract_country)
    tyndp_buses['name'] = tyndp_buses['name'].apply(extract_name)

    # Create tags
    tag_cols = ['name',
                'country',
                'url',
                'tyndp2020_proj_id',
                'tyndp2020_invest_id',
                'tyndp_status']

    tyndp_buses.loc[:, 'tags'] = tyndp_buses.apply(
                                    lambda row: _cols_to_tags(tag_cols, row),
                                    axis=1)

    tyndp_buses = tyndp_buses.drop(tag_cols, axis=1)

    return tyndp_buses


def _lines_to_pypsa(tyndp_lines, all_buses):
    tyndp_lines = tyndp_lines.drop([
        'asset_type',
        'ac_dc',
        'substation_1',
        'substation_2',
        'investment_id',
        'p_nom_max',
        'remarks',
        'description'], axis=1)

    tyndp_lines = tyndp_lines.rename(columns={
        'project_id': 'tyndp2020_proj_id',
        'specified_length_km': 'length',
        'voltage': 'v_nom',
        'status': 'tyndp_status'
    })

    # Create tags
    tag_cols = ['url', 'tyndp2020_proj_id']
    tyndp_lines.loc[:, 'tags'] = tyndp_lines.apply(
                                    lambda row: _cols_to_tags(tag_cols, row),
                                    axis=1)
    tyndp_lines = tyndp_lines.drop(tag_cols, axis=1)

    tyndp_lines = tyndp_lines.join(_create_bus0_bus1(tyndp_lines, all_buses))

    bus0_coords = all_buses.loc[tyndp_lines.bus0, ('x', 'y')]
    bus1_coords = all_buses.loc[tyndp_lines.bus1, ('x', 'y')]

    bus0_coords.columns = ['x1', 'y1']
    bus1_coords.columns = ['x2', 'y2']

    bus0_coords.index = tyndp_lines.index
    bus1_coords.index = tyndp_lines.index

    tyndp_lines.loc[:, ('x1', 'y1')] = bus0_coords
    tyndp_lines.loc[:, ('x2', 'y2')] = bus1_coords

    tyndp_lines.loc[:, 'geometry'] = tyndp_lines.apply(_coords_to_linestring,
                                                       axis=1)
    tyndp_lines = tyndp_lines.drop(['x1', 'y1', 'x2', 'y2'], axis=1)

    return tyndp_lines


def _links_to_pypsa(tyndp_links, all_buses):
    tyndp_links = tyndp_links.drop([
        'asset_type',
        'ac_dc',
        'substation_1',
        'substation_2',
        'remarks',
        'description'], axis=1)

    tyndp_links = tyndp_links.rename(columns={
        'project_id': 'tyndp2020_proj_id',
        'investment_id': 'tyndp2020_invest_id',
        'specified_length_km': 'length',
        'voltage': 'v_nom',
        'status': 'tyndp_status',
        'p_nom_max': 'p_nom'
    })

    # Create tags
    tag_cols = ['url', 'tyndp2020_proj_id', 'tyndp2020_invest_id', 'tyndp_status']
    tyndp_links.loc[:, 'tags'] = tyndp_links.apply(
                                    lambda row: _cols_to_tags(tag_cols, row),
                                    axis=1)

    return tyndp_links.join(_create_bus0_bus1(tyndp_links, all_buses))


def _create_bus0_bus1(df, all_buses):
    buses_tree = spatial.KDTree(all_buses.loc[:, ('x', 'y')])
    _, ind0 = buses_tree.query(df.loc[:, ('x1', 'y1')])
    _, ind1 = buses_tree.query(df.loc[:, ('x2', 'y2')])

    ind0_b = ind0 < len(all_buses)
    ind1_b = ind1 < len(all_buses)

    bus0 = pd.DataFrame(all_buses.index[ind0[ind0_b]],
                        index=df.index,
                        columns=['bus0'])
    bus1 = pd.DataFrame(all_buses.index[ind1[ind1_b]],
                        index=df.index,
                        columns=['bus1'])

    return bus0.join(bus1)


def _coords_to_linestring(row):
    return str(LineString([[row.x1, row.y1], [row.x2, row.y2]]))


def _find_closest_gridx_buses(tyndp_buses,
                              gridx_buses,
                              distance_upper_bound=0.2):
    treecoords = gridx_buses.loc[:, ('x', 'y')]
    querycoords = tyndp_buses.loc[:, ('x', 'y')]

    tree = spatial.KDTree(treecoords)
    dist, ind = tree.query(querycoords,
                           distance_upper_bound=distance_upper_bound)
    found_b = ind < gridx_buses.index.size
    found_i = np.arange(tyndp_buses.index.size)[found_b]

    tyndp_buses['closest_gridx_bus'] = pd.DataFrame(
                                            dict(D=dist[found_b],
                                            i=gridx_buses.index[ind[found_b] % buses.index.size]),
                                            index=tyndp_buses.index[found_i]) \
                                         .sort_values(by='D')\
                                         [lambda ds: ~ds.index.duplicated(keep='first')] \
                                         .sort_index()['i']
    return tyndp_buses


def _split_buses_into_existing_new(tyndp_buses, gridx_buses):
    existing_buses = tyndp_buses.loc[~tyndp_buses['closest_gridx_bus'].isnull()]
    existing_buses = existing_buses.set_index('closest_gridx_bus')

    new_buses = tyndp_buses.loc[tyndp_buses['closest_gridx_bus'].isnull()]
    new_buses = new_buses.drop('closest_gridx_bus', axis=1)

    return existing_buses, new_buses


def _split_lines_into_existing_new(tyndp_lines, gridx_lines):
    # Get "undirected" 'tyndp_lines', as they might occur
    # in 'gridx_lines' defined in either direction.
    tyndp_lines_rev = tyndp_lines.rename(columns={'bus0': 'bus1', 'bus1': 'bus0'})
    tyndp_undir = pd.concat([tyndp_lines, tyndp_lines_rev])

    # like JOIN based on column values
    existing_lines = tyndp_undir.merge(gridx_lines.loc[:, ('bus0', 'bus1')]
                                       .drop_duplicates(),
                                       how='inner')

    # "reversed index" hack to get original index
    # from 'tyndp_undir' lost through merge
    row_to_ind = {tuple(row):ind for ind, row in
                  tyndp_undir.loc[:, ('bus0', 'bus1')].iterrows()}

    existing_ind = [row_to_ind[t] for t in
                    map(tuple, existing_lines.loc[:, ('bus0', 'bus1')].values)]
    new_ind = list(set(tyndp_lines.index) - set(existing_ind))

    new_lines = tyndp_lines.loc[new_ind]

    # Update indices of 'existing_lines' to those of the counterpart in 'lines'
    row_to_ind = {tuple(row): ind for ind, row in
                  lines.loc[:, ('bus0', 'bus1')].drop_duplicates().iterrows()}
    existing_lines.loc[:, 'line_id'] = (existing_lines.loc[:, ('bus0', 'bus1')]
                                        .apply(lambda row: row_to_ind[tuple(row)],
                                               axis=1))
    existing_lines = existing_lines.set_index('line_id')

    return existing_lines, new_lines


def _take_larger_vals(df_a, df_b, cols):
    df_a = df_a.loc[:, cols]
    df_b = df_b.loc[df_a.index, cols]

    # Replace NaN values in a with values in b.
    df_c = df_a.combine_first(df_b)

    # Take larger values (note that this prefers "True" over "False").
    return df_c.where(df_c > df_b, df_b)


def _compare_tags_buses(existing_buses, gridx_buses, new_buses):
    for index, tyndp_bus in existing_buses.loc[~gridx_buses.tags.isna()].iterrows():
        gridx_bus = gridx_buses.loc[index]

        tyndp_tags = _tags_to_dict(tyndp_bus)
        gridx_tags = _tags_to_dict(gridx_bus)
        conflicting_tags = [k for k in tyndp_tags if k in gridx_tags and tyndp_tags[k] != gridx_tags[k]]

        if conflicting_tags:
            # Edge case due to buses in different countries
            # that are geographically close to each other.
            project_id = tyndp_tags['tyndp2020_proj_id']
            investment_id = tyndp_tags['tyndp2020_invest_id']
            s = f"Inconsistent values for keys {conflicting_tags} between " \
                f"TYNDP bus with project_id={project_id}, investment_id={investment_id} " \
                f"and its geographically closest gridextract bus with index='{index}'.\n" \
                "Adding TYNDP bus as a new bus."
            warnings.warn(s)
            new_buses.loc[index] = tyndp_bus

            not_in_gridx = {k: v for k, v in tyndp_tags.items() if k not in gridx_tags}
            joined_tags = gridx_tags | not_in_gridx

            existing_buses.loc[index, 'tags'] = _dict_to_tags(joined_tags)

    return existing_buses, new_buses


def _merge_tags_lines(existing_lines, gridx_lines):
    for ind, row in existing_lines.iterrows():
        gridx_tags = _tags_to_dict(gridx_lines.loc[ind])
        existing_tags = _tags_to_dict(row)
        row.tags = _dict_to_tags(gridx_tags | existing_tags)

    return existing_lines


def _assign_index(new_buses, gridx_buses):
    max_index = max(map(int, gridx_buses.index))
    new_index = list(map(str, range(max_index + 1, max_index + 1 + len(new_buses))))
    new_buses.index = new_index

    return new_buses


def main():
    tyndp_file = sys.argv[1]  # e.g. '2020/tyndp_2020.csv'
    df = pd.read_csv(tyndp_file)
    df = df.loc[~df['project_id'].isin(_IGNORE_PROJECTS)]
    df = df.loc[~df['investment_id'].isin(_IGNORE_INVESTMENTS)]

    tyndp_buses, tyndp_lines, tyndp_links = _splitinto_buses_lines_links(df)

    # Load buses from PyPSA-Eur gridextract.
    gridx_buses = (pd.read_csv(r'entsoegridkit/buses.csv',
                               quotechar="'",
                               true_values=['t'],
                               false_values=['f'],
                               dtype=dict(bus_id="str"))
                   .set_index("bus_id")
                   .drop(['station_id'], axis=1)
                   .rename(columns=dict(voltage='v_nom')))

    tyndp_buses = _buses_to_pypsa(tyndp_buses, gridx_buses)
    tyndp_buses = _find_closest_gridx_buses(tyndp_buses)

    existing_buses, new_buses = _split_buses_into_existing_new(tyndp_buses)

    # If necessary, update voltage and DC capability of counterparts.
    buses_check_cols = ['v_nom', 'dc']
    existing_buses.loc[:, buses_check_cols] = _take_larger_vals(existing_buses,
                                                                gridx_buses,
                                                                buses_check_cols)

    # Compare tags of existing_buses to those of their counterparts.
    # Might reveal that buses were incorrectly put into 'existing_buses'.
    existing_buses, new_buses = _compare_tags_buses(
                                    existing_buses,
                                    gridx_buses,
                                    new_buses)

    new_buses = _assign_index(new_buses, gridx_buses)

    # TODO: Define 'all_buses' (merge gridx with existing, add new)

    # Load lines from PyPSA-Eur gridextract.
    gridx_lines = (pd.read_csv(r'entsoegridkit/lines.csv',
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
    gridx_lines = pd.DataFrame(gridx_lines.loc[gridx_lines.bus0.isin(gridx_buses.index)
                               & gridx_lines.bus1.isin(gridx_buses.index)])

    existing_lines, new_lines = _split_lines_into_existing_new(tyndp_lines,
                                                               gridx_lines)

    # If necessary, update voltage and 'underground' status of counterparts.
    lines_check_cols = ['v_nom', 'underground']
    existing_lines.loc[:, lines_check_cols] = _take_larger_vals(existing_lines,
                                                                gridx_lines,
                                                                lines_check_cols)
    existing_lines = _merge_tags_lines(existing_lines, gridx_lines)

    # Load links from PyPSA-Eur gridextract.


if __name__ == "__main__":
    main()