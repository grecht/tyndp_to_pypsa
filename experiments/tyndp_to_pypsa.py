import os
from geopy import *
import geopy.distance
import pandas as pd
import numpy as np
import math
import itertools
import re
import Levenshtein # get via 'pip install levenshtein' (not on conda)

# see 'util.py' for expected column names

def extract_name_country(buses_file='buses_v0.1.0.csv'):
    buses = (pd.read_csv(buses_file, quotechar="'",
                         true_values='t', false_values='f',
                         dtype=dict(bus_id="str"))
             .set_index("bus_id")
             .drop(['station_id'], axis=1)
             .rename(columns=dict(voltage='v_nom')))

    buses = buses.query('tags.notnull()', engine='python')
    buses = buses.query("symbol == 'Substation'") 

    # Form: 'key => value, key => value, ...'
    split_regex = r'("\w+"=>"[^"]*"),'

    tag_regex = r'"(?P<key>\w+)"=>"(?P<value>[^"]*)"'  # Form: 'key => value'
    tag_pattern = re.compile(tag_regex)

    rows = []

    for _, row in buses.iterrows():
        name = ''
        country = ''
        x = row['x']
        y = row['y']

        tags_string = row['tags']

        tags = re.split(split_regex, tags_string)

        # Remove whitespaces at front and end, remove None values
        tags = [s.strip() for s in tags]
        tags = list(filter(None, tags))

        for tag in tags:
            m = tag_pattern.match(tag)
            key = m.group('key')
            value = m.group('value')

            if key == 'name_eng':
                name = value.strip()
            elif key == 'country':
                country = value.strip()

        if name == 'unknown' or not name:
            continue

        rows.append((name, country, x, y))
    curated_buses = pd.DataFrame.from_records(
        rows, columns=['name', 'country', 'x', 'y'])
    return curated_buses.loc[~curated_buses.duplicated()]


# TODO: rename parameter, this does not only work on lines.
def prepare_substation_names(lines):
    # Form: 'Glorenza (IT)'
    subst_regex = r'(?P<place>.+)\s?[\[(](?P<country>\w{2})[)\]]'
    subst_pattern = re.compile(subst_regex)

    # use this if other pattern does not match to remove comments in parentheses
    # e.g. 'Molai (through Sklavouna Terminal)'
    # TODO: does it make sense to "throw away" information here?
    alt_regex = r'(?P<place>.+)\s?[\[(].*[)\]]'
    alt_pattern = re.compile(alt_regex)

    fr_names = []
    fr_countries = []
    to_names = []
    to_countries = []

    for _, row in lines.iterrows():
        fr = row['substation_1']
        to = row['substation_2']

        # default values if regex does not match
        fr_name = fr
        to_name = to
        fr_country = np.NAN
        to_country = np.NAN

        fr_match = subst_pattern.match(fr)
        to_match = subst_pattern.match(to)

        if fr_match:
            fr_name = fr_match.group('place').strip()
            fr_country = fr_match.group('country').strip()
        else:
            fr_alt_match = alt_pattern.match(fr)
            if fr_alt_match:
                fr_name = fr_alt_match.group('place')

        if to_match:
            to_name = to_match.group('place').strip()
            to_country = to_match.group('country').strip()
        else:
            to_alt_match = alt_pattern.match(to)
            if to_alt_match:
                to_name = to_alt_match.group('place')

        fr_names.append(fr_name)
        fr_countries.append(fr_country)
        to_names.append(to_name)
        to_countries.append(to_country)

    lines['substation_1'] = fr_names
    lines['substation_2'] = to_names
    lines['country_1'] = fr_countries
    lines['country_2'] = to_countries
    return lines


def tyndp_to_substation(lines, curated_buses):
    tyndp_subs = set(lines['substation_1']).union(set(lines['substation_2']))
    tyndp_to_bus = {}

    for tyndp in tyndp_subs:
        buses_subs = curated_buses.name.values

        tyndp_to_bus[tyndp] = min([(bus, Levenshtein.distance(bus.lower(), tyndp.lower())) for bus in buses_subs],
                                  key=lambda t: t[1])[0]
    return tyndp_to_bus


def _extract_coords(rows):
    coordinates = []
    for _, row in rows.iterrows():
        coordinates.append((row['x'], row['y']))
    return coordinates

def match_pair_with_length(s1_rows, s2_rows, length):
    s1_coords = _extract_coords(s1_rows)
    s2_coords = _extract_coords(s2_rows)
    
    combinations = list(itertools.product(s1_coords, s2_coords))
    # TODO: already multiply by 1.2 here?
    with_distance = [(a, b, geopy.distance.distance(a, b).km)
                     for (a, b) in combinations]
    return min(with_distance, key=lambda t: abs(length - t[2]))


def match_pair_with_length_geopy(s1_locations, s2_locations, length):
    s1_first_name = s1_locations[0][0]
    s2_first_name = s2_locations[0][0]

    # Only take locations which at least include name of the first location in list
    # (assumption: best name-based match).
    # Map location objects to (lat, lon) tuples
    s1_locations = [l for l in s1_locations if s1_first_name in l[0]]
    s2_locations = [l for l in s2_locations if s2_first_name in l[0]]

    lat_lon = lambda loc: (loc.latitude, loc.longitude)
    combinations  = list(itertools.product(s1_locations, s2_locations))
    with_distance = [(a, b, geopy.distance.distance(lat_lon(a),lat_lon(b)).km) for (a,b) in combinations]
    
    best_match = min(with_distance, key=lambda t: abs(length - t[2]))
    return best_match


def match_tyndp_with_buses(lines, tyndp_to_bus, curated_buses):
    fr_to_tuples = {}

    for index, row in lines.iterrows():
        fr = row['substation_1']
        to = row['substation_2']

        fr_country = row['country_1']
        to_country = row['country_2']

        s1 = tyndp_to_bus[fr]
        s2 = tyndp_to_bus[to]

        # Extract respective rows in buses to determine coordinates
        buses_s1 = curated_buses.loc[curated_buses.name == s1]
        buses_s2 = curated_buses.loc[curated_buses.name == s2]

        # If we were able to extract country from name, restrict chosen rows to this country.
        if not pd.isna(fr_country):
            buses_s1 = buses_s1.loc[buses_s1['country'] == fr_country]
        if not pd.isna(to_country):
            buses_s2 = buses_s2.loc[buses_s2['country'] == to_country]

        if buses_s1.empty or buses_s2.empty:
            continue

        # Choose pair which matches length best
        length = row['specified_length_km']
        (x1, y1), (x2, y2), coord_dist = match_pair_with_length(
            buses_s1, buses_s2, length)

        tpl = (s1, x1, y1, s2, x2, y2, coord_dist)

        if math.isclose(coord_dist, length, rel_tol=0.45):
            fr_to_tuples[index] = tpl

    results = pd.DataFrame(index=fr_to_tuples.keys(),
                           data=fr_to_tuples.values(),
                           columns=['s1', 'x1', 'y1', 's2', 'x2', 'y2', 'coord_dist_km'])
    
    # TODO: is there a better way?
    columns = ['commissioning_year', 'status', 'ac_dc', 'voltage', 'underground']
    columns = set(lines.columns).intersection(set(columns))

    # TODO: return lines which could not be matched
    return results.join(lines[columns])


def match_tyndp_with_geopy(lines):
    fr_to_tuples = {}

    locator = AlgoliaPlaces(user_agent='esm_group')
    geocode = locator.geocode

    for index, row in lines.iterrows():
        fr = row['substation_1']
        to = row['substation_2']
        dist = row['specified_length_km']

        fr_country = row['country_1']
        to_country = row['country_2']

        fr_locs = (geocode(fr, exactly_one=False) if pd.isna(fr_country)
                   else geocode(fr, exactly_one=False, countries=[fr_country]))
        to_locs = (geocode(to, exactly_one=False) if pd.isna(to_country)
                   else geocode(to, exactly_one=False, countries=[to_country]))

        if fr_locs is None or to_locs is None:
            continue

        (s1, (x1, y1)), (s2, (x2, y2)), coord_dist = match_pair_with_length_geopy(
            fr_locs, to_locs, dist)

        if math.isclose(coord_dist, dist, rel_tol=0.45):
            fr_to_tuples[index] = (s1, x1, y1, s2, x2, y2, coord_dist)

    results = pd.DataFrame(index=fr_to_tuples.keys(),
                           data=fr_to_tuples.values(),
                           columns=['s1', 'x1', 'y1', 's2', 'x2', 'y2', 'coord_dist_km'])
    return results.join(lines[['commissioning_year', 'status', 'ac_dc', 'voltage', 'underground']])



if __name__ == "__main__":
    pass
