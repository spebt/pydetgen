# SC-1 panel-based SC-SPECT layout

This directory generates the stationary **SC-1 / Option 2** two-dimensional
geometry used for the ASCI experiment.

## Geometry

- 70 mm diameter FOV, represented downstream by 280 x 280 pixels at
  0.25 mm/pixel
- 36 biconical pinholes, 2 mm diameter and 27 degree opening angle
- 215 mm collimator junction radius and 8 mm total radial thickness
- 44 detector panels whose innermost crystal-layer centers are at 757 mm
- eight radial detector layers with tangential populations
  `[12, 15, 18, 21, 24, 27, 30, 32]`
- each layer's crystals are distributed evenly across the same 32 tangential
  slots; they are not packed contiguously at the panel center
- 2.4 x 2.4 mm active crystals centered in 3.36 x 3.36 mm slots
- 7,876 detector crystals in total

The layout contains exactly one stationary position. Generate rotation-only,
translation-only, and combined-motion variants from this output using
`utils/transform_scanner_multiple_positions.py`.

## Generate

From the repository root:

```bash
python layouts/scspect_panel_biconical/generate_layout.py --plot
```

Use `--output-dir` to target a cluster data directory:

```bash
python layouts/scspect_panel_biconical/generate_layout.py \
  --output-dir /vscratch/grp-rutaoyao/sid/data/scanner_layouts
```

The `applied_config` metadata records the intended 3 x 1 detector subdivision
and 3 x 3 FOV tiling. Those settings must also be set in the `pymatcal` YAML;
the layout tensor itself contains only physical detector and collimator
polygons.
