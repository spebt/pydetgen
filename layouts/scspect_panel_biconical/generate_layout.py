"""Generate stationary SC-1: the panel-based Option 2 SC-SPECT geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from helper import (
    generate_biconical_collimator,
    generate_panel_detectors,
    geometry_md5,
)


CONFIG = {
    "configuration_id": "SC-1",
    "description": "Stationary panel-based self-collimation SPECT geometry",
    "fov_diameter_mm": 70.0,
    "fov_pixels": [280, 280],
    "pixel_size_mm": 0.25,
    "pinhole_diameter_mm": 2.0,
    "pinhole_opening_angle_deg": 27.0,
    "n_pinholes": 36,
    "collimator_ring_radius_mm": 215.0,
    "collimator_thickness_mm": 8.0,
    "detector_first_layer_center_radius_mm": 757.0,
    "n_detector_panels": 44,
    "blocks_per_panel": [4, 1],
    "crystals_per_block": [8, 8],
    "crystal_slot_size_mm": [3.36, 3.36],
    "crystal_size_mm": [2.4, 2.4],
    "radial_layer_populations": [12, 15, 18, 21, 24, 27, 30, 32],
    "detector_subdivisions": [3, 1],
    "fov_subdivisions": [3, 3],
}


def build_layout(config: dict = CONFIG) -> dict:
    """Build one stationary layout compatible with pymatcal and transforms."""
    if config["fov_pixels"][0] * config["pixel_size_mm"] != config["fov_diameter_mm"]:
        raise ValueError("FOV x dimension does not match pixel count and pixel size")
    if config["fov_pixels"][1] * config["pixel_size_mm"] != config["fov_diameter_mm"]:
        raise ValueError("FOV y dimension does not match pixel count and pixel size")

    detector_units = generate_panel_detectors(
        detector_first_layer_center_radius_mm=config[
            "detector_first_layer_center_radius_mm"
        ],
        n_detector_panels=config["n_detector_panels"],
        radial_layer_populations=config["radial_layer_populations"],
        tangential_slots_per_panel=(
            config["blocks_per_panel"][0] * config["crystals_per_block"][0]
        ),
        crystal_slot_tangential_mm=config["crystal_slot_size_mm"][0],
        crystal_slot_radial_mm=config["crystal_slot_size_mm"][1],
        crystal_tangential_mm=config["crystal_size_mm"][0],
        crystal_radial_mm=config["crystal_size_mm"][1],
    )
    plate_segments = generate_biconical_collimator(
        pinhole_diameter_mm=config["pinhole_diameter_mm"],
        pinhole_opening_angle_deg=config["pinhole_opening_angle_deg"],
        n_pinholes=config["n_pinholes"],
        collimator_ring_radius_mm=config["collimator_ring_radius_mm"],
        collimator_thickness_mm=config["collimator_thickness_mm"],
    )

    expected_detectors = config["n_detector_panels"] * sum(
        config["radial_layer_populations"]
    )
    if detector_units.shape != (expected_detectors, 4, 2):
        raise RuntimeError(
            f"expected {(expected_detectors, 4, 2)} detectors, "
            f"generated {tuple(detector_units.shape)}"
        )
    if plate_segments.shape != (2 * config["n_pinholes"], 4, 2):
        raise RuntimeError("unexpected collimator segment count")

    base_layout = {
        "position": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        "detector units": detector_units,
        "plate segments": plate_segments,
    }
    return {
        "scanner MD5": geometry_md5(detector_units, plate_segments),
        "applied_config": dict(config),
        "motion_parameters": {
            "scheme": "stationary",
            "n_rotational_steps_defined": 1,
            "n_translational_shifts_grid": [1, 1],
            "translational_step_size_mm": [0.0, 0.0],
            "generated_n_positions": 1,
        },
        "layouts": {"position 000": base_layout},
    }


def plot_layout(output_data: dict, config: dict, output_path: Path) -> None:
    """Save a full-layout preview without requiring an interactive display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    layout = output_data["layouts"]["position 000"]
    figure, axes = plt.subplots(figsize=(10, 10))
    axes.add_collection(
        PolyCollection(
            layout["detector units"].tolist(),
            facecolor="lightskyblue",
            edgecolor="royalblue",
            linewidth=0.15,
            label="Detector crystals",
        )
    )
    axes.add_collection(
        PolyCollection(
            layout["plate segments"].tolist(),
            facecolor="dimgray",
            edgecolor="black",
            linewidth=0.5,
            label="Biconical collimator",
        )
    )
    axes.add_patch(
        plt.Circle(
            (0.0, 0.0),
            config["fov_diameter_mm"] / 2.0,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.5,
            label="70 mm FOV",
        )
    )
    plot_limit = config["detector_first_layer_center_radius_mm"] + (
        len(config["radial_layer_populations"]) + 2
    ) * config["crystal_slot_size_mm"][1]
    axes.set_xlim(-plot_limit, plot_limit)
    axes.set_ylim(-plot_limit, plot_limit)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("X (mm)")
    axes.set_ylabel("Y (mm)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parents[3] / "data/scanner_layouts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"output directory (default: {default_output_dir})",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also save a PNG preview beside the tensor",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_data = build_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = "scspect_panel_biconical_sc1_stationary_36p_2mm"
    tensor_path = args.output_dir / f"{stem}.tensor"
    torch.save(output_data, tensor_path)

    detector_count = output_data["layouts"]["position 000"][
        "detector units"
    ].shape[0]
    segment_count = output_data["layouts"]["position 000"][
        "plate segments"
    ].shape[0]
    print(f"Saved SC-1 layout: {tensor_path}")
    print(f"Detector crystals: {detector_count}")
    print(f"Collimator segments: {segment_count}")
    print(f"Scanner MD5: {output_data['scanner MD5']}")

    if args.plot:
        plot_path = args.output_dir / f"{stem}.png"
        plot_layout(output_data, CONFIG, plot_path)
        print(f"Saved preview: {plot_path}")


if __name__ == "__main__":
    main()
