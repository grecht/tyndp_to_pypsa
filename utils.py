import pandas as pd
import numpy as np
import math
import dateutil
import datetime
import itertools
import re

_column_names = [
    'investment_id',
    'project_id',
    'commissioning_year',
    'status',
    'asset_type',
    'substation_1',
    'substation_2',
    'ac_dc',
    'voltage',
    'specified_length_km',
    'description'
]

_status = [
    'under_consideration',
    'planned_not_yet_permitting',
    'in_permitting',
    'under_construction'
]


def prepare_tyndp_data(excel_file, 
                       sheet_name,
                       column_semantics,
                       status_map,
                       asset_type_map=None,
                       header_row=0,
                       base_url=None):
    if any([(v not in _column_names) for v in column_semantics.values()]):
        wrong_col_names = [v for v in column_semantics.values() if v not in _column_names]
        error_msg = f""" 
                    Argument 'column_semantics' contains wrong values {wrong_col_names}.
                    The values should correspond to those in {_column_names}.
                    """
        raise ValueError(error_msg)

    tyndp = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
    
    # 'clean up' column names
    simplify_column  = lambda s: s.replace(' ', '').replace('\n', '').lower()
    column_semantics = {simplify_column(k):v for (k,v) in column_semantics.items()}
    wanted_columns   = column_semantics.keys()
    tyndp.columns    = [simplify_column(col) for col in tyndp.columns]
    
    # reduce to wanted_columns
    tyndp = tyndp[wanted_columns]

    # map columns to specified names
    tyndp.columns = [column_semantics[c] for c in tyndp.columns]
    
    tyndp.description = tyndp.description.astype(str)

    # TODO: (cleanup) convert whitespaces to np.NAN values.

    if tyndp['status'].dtype == pd.StringDtype:
        tyndp['status'] = tyndp['status'].str.lower()

    # replace status with values as specified in status_map
    tyndp = tyndp.loc[tyndp['status'].isin(status_map.keys())]
    tyndp = tyndp.replace({'status': status_map})

    # only choose those in permitting or under construction
    # tyndp = tyndp.loc[tyndp['status'].isin('in_permitting', 'under_construction')]

    if asset_type_map:
        # map asset types to specified names
        tyndp = tyndp.loc[tyndp['asset_type'].isin(asset_type_map.keys())]
        tyndp = tyndp.replace({'asset_type': asset_type_map})
    else:
        # TODO: infer cable if e.g. description contains 'underground', 'undersea', 'cable',...
        # assumption: 2 substations specified => line, 1 substation specified => substation
        tyndp.loc[tyndp.substation_2.isna(), 'asset_type']  = 'substation'
        tyndp.loc[~tyndp.substation_2.isna(), 'asset_type'] = 'line'

    # map 'cable' to 'line' but add binary column 'underground'
    # (matches convention of lines.csv in PyPSA-Eur)
    tyndp['underground'] = tyndp['asset_type'].str.fullmatch('cable')
    tyndp                = tyndp.replace({'asset_type': {'cable': 'line'}})

    # try to extract missing values from description if not given as separate column
    if 'voltage' not in column_semantics.values() and 'description' in tyndp.columns:
        regex = r'(\d{3})\s*\-*kv' # e.g. '400 kv', '400-kv', '400kv'
        tyndp['voltage (guess)'] = _find_max_quantity_in_desc(tyndp, regex)
    
    if 'specified_length_km' not in column_semantics.values() and 'description' in tyndp.columns:
        regex = r'(\d+)[\.,]?\d*\s*\-*km' # e.g. '20km', '20.5km', '20,5 km'
        tyndp['specified_length_km (guess)'] =  _find_max_quantity_in_desc(tyndp, regex)
    
    # TODO: infer AC/DC
    
    if base_url is not None:
        tyndp['url'] = tyndp.apply(lambda r: base_url + str(r['project_id']), axis=1)

    return tyndp


def _find_max_quantity_in_desc(tyndp, regex):
    pattern  = re.compile(regex)
    find_max = lambda d: max([int(s) for s in re.findall(pattern, d.lower())], default=np.NAN)
    return tyndp.description.apply(find_max)


def _round_to_year(date_string):
    d = dateutil.parser.parse(date_string)
    f = datetime.datetime(d.year, 1, 1)
    return (f.year + 1) if ((d - f).days > (365 / 2)) else f.year


def commissioning_dates_to_year(dates: pd.Series):
    # Clean up dates
    dates = dates.apply(lambda s: s.replace(' ', ''))

    # Replace values of format '2022-2023' with the second year
    dates = dates.str.replace(r'\d{4}-(\d{4})', r'\1',regex=True)
    # Remove comma in values specified like '2,022' 
    dates = dates.str.replace(r'(\d{1}),(\d{3})', r'\1\2',regex=True)

    return dates.apply(_round_to_year)

if __name__ == "__main__":
    pass