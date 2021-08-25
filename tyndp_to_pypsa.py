import geopy
import pandas as pd

def main():
    excel = 'transmission.xlsx'
    sheet = 'Trans.Investments'

    df = pd.read_excel(excel, sheet_name=sheet)

    # TODO: might not work in later versions, use sanity check.
    df.columns = df.iloc[0]
    df         = df.drop(index=df.index[0])

    # TODO: also contains new substations!
    
    # df.loc[:, ['Project ID', 'Commissioning Year'] +  df.columns.startswith('Status') + ['']]

if __name__ == "__main__":
    main()