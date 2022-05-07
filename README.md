This project aims to extend the information contained in the Ten Year Network Development Plans (TYNDPs) made by ENTSO-E in order to incorporate them into an energy system model.
Most notably this includes adding the substation coordinates as well as making general corrections to the data based on official project descriptions.
So far, only the TYNDP 2020 has been extended.

# Data Sources & Brief Descriptions
## TYNDP data in form of excel files: https://tyndp.entsoe.eu/maps-data/
The excel file for TYNDP 20XY can be found in folder `20XY/`. Some notes about the data:
- Each file has a different structure and uses different column names
- The project descriptions in the files may contain different information than the project descriptions on the TYNDP website
- Only the substation's names are included, but not their coordinates
- Some projects are not specified correctly: For example, the file contains an entry for a line between substations `A` and `C`, but the project description indicates that there are actually two lines connecting `A` and `C` via `B`. Such information is sometimes included in the description, on the TYNDP website, or in official documents for the project.

## Coordinates of substations: https://www.openstreetmap.org
Most substations can be found in Openstreetmap, and often their names match those in the TYNDP. 
## Coordinates of Finnish substations: https://www.businessfinland.fi/globalassets/international-customers/data-center-assets/substations.kmz
Openstreetmap does not contain many Finnish substations or does not include their names. This obscure file can be opened with Google Earth and contains the coordinates of the most important substations in Finland.

# Procedure
Due to the poor data quality of the TYNDP excel files mentioned above, it is almost impossible to extend the TYNDPs without manual work.
Hence, the following procedure was used:
- Use `prepare_tyndp_data()` from `utils.py` to read in and process the excel file. This includes:
    - Conversion of data to consistent format. Semantics of column names need to be specified by user, see `manual_utils.ipynb` for examples.
    - Adding empty columns for substation coordinates and line capacities (mostly used for HVDC lines)
    - 'Guessing' of line voltage and length from the `description` column
    - Adding hyperlinks to the respective TYNDP project website, if available
    - ⚠️ Transformation of commissioning dates to years, which sometimes involves rounding up.
- Correct data if information in 'description' column differs from rest of the row.
- Manually search for substation in Openstreetmap in order to find coordinates
