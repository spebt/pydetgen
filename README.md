# PyDetGen

SPEBT project Python package for generating the detector system configuration file

## Prerequisites

- `Python 3.10` or higher, preferably `Python 3.12` or higher
- `PyTorch`
- `NumPy`
- `Pandas`
- `Matplotlib`

## Get the code

- by `git clone https://github.com/spebt/pydetgen.git`
or
- by download from the releases page

## How to run

First, resolve the prerequisites.

The final scanner geometry is generated as:
```mermaid
graph LR;
    panel.svg-->panel.csv-->scanner.csv;
```

### Quick start

#### Generate the scanner geometry and save into `csv` files

1. Parse the `panel.svg` and save into a `panel.csv` file in the `tmp` folder
    ```bash
    python load_panel_svg_and_generate_panel_csv.py
    ```
1. Load the panel geometries and perform the transformation to generate the scanner geometries.
    ```bash
    python load_panel_csv_and_generate_scanner_csv.py
    ```
   - Scanner with different aperture sizes are generated
   - For each aperture size, the `panel #0` are replicated and rotated to 6 angles
   - Apertures: 
     - 1.0 mm
     - 1.5 mm
     - 2.0 mm
     - 2.5 mm
     - 3.0 mm
     - 3.5 mm
     - 4.0 mm
     - 4.5 mm
   - Generated `csv` files are saved in `output/scanner_csv` folder.
  
#### Visualization of the scanner geometry

1. Plot `panel #0` geometry
    ```sh
    python plot_panel_csv.py
    ```
1. Plot scanner geometry
    ```sh
    python plot_scanner_csv.py <scanner_csv_file_path>
    ```
    for example:
    ```sh
    python plot_scanner_csv.py output/scanner_csv/scanner_1.5_mm_aperture.csv
    ```

The plots are saved as `png` files in `output/plots` folder

### Change the panel geometry

The `panel.svg` file is create with [Inkscape](https://inkscape.org/), you can modify it with Inkscape or similar software. 

> [!TIP]
> If you change `panel.svg`, verify the following: 
> - the plate geometries for `panel #0` is grouped in `plate_0`
> - the crystal geometries for `panel #0` is grouped in `panel_0`

## Contact

Developer: Fang Han (fhanonline@gmail.com)

