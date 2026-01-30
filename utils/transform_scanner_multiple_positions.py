import argparse
import os
import yaml
import torch
import numpy as np
from typing import Sequence, Dict, Union, List, Optional
from torch import (
    linspace,
    arange,
    stack,
    cat,
    cos,
    sin,
    meshgrid,
    tensor,
    Tensor,
    float32,
    pi,
    deg2rad,
)

# Type hint for the output dictionary
OutDataDict = Dict[str, Union[str, Dict, Tensor]]

# --- Vectorized Transformation Functions ---

def rotation_matrix_2d_batch(angles: Tensor) -> Tensor:
    """
    Generates a batch of 2D rotation matrices for a given set of angles.
    
    Args:
        angles: A 1D Tensor of rotation angles in radians.
        
    Returns:
        A Tensor of shape (N, 2, 2) representing the rotation matrices.
    """
    c, s = cos(angles), sin(angles)
    return stack((c, -s, s, c), dim=1).view(-1, 2, 2)

def generate_master_positions(
    n_rotations: int,
    custom_angles_deg: List[float],
    translation_mode: str,
    n_shifts: Sequence[int],
    shift_step_mm: Sequence[float],
    n_transaxial_positions: int,
    ellipse_a_mm: float,
    ellipse_b_mm: float,
    outward_shift_mm: float,
    custom_translations: Optional[Sequence[Sequence[float]]] = None,
) -> Tensor:
    """
    Generates a master set of position parameters (angle, tx, ty).
    Supports: 'grid', 'elliptical', and 'custom' modes.
    
    Returns:
        A Tensor of shape (N, 3) where each row is [angle_rad, tx, ty].
    """
    # 1. Generate angles (common to all modes)
    if custom_angles_deg:
        angles = deg2rad(tensor(custom_angles_deg, dtype=float32))
    elif n_rotations > 0:
        angles = linspace(0, 2 * pi, n_rotations + 1)[:-1]
    else:
        angles = tensor([0.0], dtype=float32)

    # 2. Generate shifts based on the selected mode
    shifts = tensor([], dtype=float32)
    
    if translation_mode == 'grid':
        shift_ix, shift_iy = meshgrid(arange(n_shifts[0]), arange(n_shifts[1]), indexing="ij")
        # Serpentine pattern for efficiency in movement simulations
        shift_iy[1::2, :] = shift_iy[1::2, :].flip(1) 
        shifts = stack((shift_ix, shift_iy), dim=-1).view(-1, 2) * tensor(shift_step_mm, dtype=float32)
        if shifts.shape[0] > 1:
            shifts = shifts - shifts.mean(dim=0) # Center the grid relative to origin
            
    elif translation_mode == 'elliptical':
        thetas = linspace(0, 2 * pi, steps=n_transaxial_positions + 1)[:-1]
        xs = ellipse_a_mm * cos(thetas)
        ys = ellipse_b_mm * sin(thetas)
        shifts = stack([xs, ys], dim=1)
        if outward_shift_mm != 0.0:
            radii = torch.linalg.norm(shifts, dim=1, keepdims=True).clamp(min=1e-6)
            dirs = shifts / radii
            shifts = shifts + outward_shift_mm * dirs

    elif translation_mode == 'custom':
        if custom_translations is None:
            raise ValueError("translation_mode is 'custom' but 'custom_translations' is None")
        shifts = tensor(custom_translations, dtype=float32)

    else:
        # Default to origin if no specific mode is selected
        shifts = tensor([[0.0, 0.0]], dtype=float32)

    # 3. Combine rotations and shifts (Cartesian product)
    n_total_shifts = shifts.shape[0]
    if n_total_shifts == 0: 
        shifts = tensor([[0.0, 0.0]], dtype=float32)
        n_total_shifts = 1
        
    # Cartesian product of angles and shifts
    expanded_shifts = shifts.repeat_interleave(angles.shape[0], dim=0)
    expanded_angles = angles.repeat(n_total_shifts)

    return cat([expanded_angles.unsqueeze(1), expanded_shifts], dim=1)


def apply_transformations_batch(base_vertices: Tensor, positions: Tensor) -> Tensor:
    """
    Applies a batch of 2D rotations and translations to a set of vertices.
    
    Args:
        base_vertices: Original vertices of the scanner components.
        positions: Master positions tensor [angle, tx, ty].
        
    Returns:
        A Tensor of shape (Positions, Polygons, Vertices, XY).
    """
    m = positions.shape[0]
    n_polys, n_verts_per_poly, _ = base_vertices.shape
    n_total_vertices = n_polys * n_verts_per_poly

    # Apply Rotations
    angles = positions[:, 0]
    rotation_matrices = rotation_matrix_2d_batch(angles)
    vertices_flat = base_vertices.view(1, n_total_vertices, 2)
    rotated_vertices_flat = (
        rotation_matrices.unsqueeze(1) @ vertices_flat.unsqueeze(-1)
    ).squeeze(-1)

    # Apply Translations
    shifts = positions[:, 1:]
    transformed_vertices_flat = rotated_vertices_flat + shifts.unsqueeze(1)

    final_shape = (m, n_polys, n_verts_per_poly, 2)
    return transformed_vertices_flat.view(final_shape)

def print_active_config(config: Dict):
    """Logs the active configuration to the console."""
    print("--- Active Configuration ---")
    if config["apply_rotations"]:
        print("Rotations: ENABLED")
        if config["custom_rotation_degrees"]:
            print(f"  - Count: {len(config['custom_rotation_degrees'])}")
        else:
            print(f"  - Evenly spaced: {config['n_rotations']}")
    else:
        print("Rotations: DISABLED")

    if config["apply_translations"]:
        print("Translations: ENABLED")
        mode = config["translation_mode"]
        print(f"  - Mode: {mode.capitalize()}")
        if mode == 'grid':
            print(f"    - Dimensions: {config['n_shifts']}, Step: {config['shift_step_mm']}")
        elif mode == 'elliptical':
            print(f"    - Ellipse parameters: a={config['ellipse_a_mm']}, b={config['ellipse_b_mm']}")
        elif mode == 'custom':
            print(f"    - Custom coordinates loaded: {len(config['custom_translations'])}")
    else:
        print("Translations: DISABLED")
    print("--------------------------\n")

# --- Main Script Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate scanner layouts from a base configuration.")
    parser.add_argument("input_file", type=str, help="Path to the input .tensor file.")
    parser.add_argument("--base_pos", type=int, default=0, help="Index of the base layout.")
    args = parser.parse_args()

    # --- CONFIGURATION DICTIONARY ---
    CONFIG = {
        "apply_rotations": True,
        "apply_translations": True,
        "translation_mode": "elliptical", 
        "custom_translations": None, # Used for T4 Protocol if mode is 'custom'
        "rotate_detectors": False,
        "rotate_collimator": True,
        "n_rotations": 0,
        "custom_rotation_degrees": list(np.arange(0, 20, 4)),
        "n_shifts": [3, 3],
        "shift_step_mm": [20.0, 20.0],
        "n_transaxial_positions": 8,
        "ellipse_a_mm": 50.0,
        "ellipse_b_mm": 35.0,
        "outward_shift_mm": 0.0,
    }

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    print_active_config(CONFIG)

    # 1. Data Loading
    print(f"Loading base layout: {args.input_file}")
    scanner_data = torch.load(args.input_file)
    input_md5 = scanner_data.get("scanner MD5", "unknown_md5")
    base_layout = scanner_data["layouts"][f"position {args.base_pos:03d}"]
    base_detectors = base_layout["detector units"]
    base_collimator = base_layout["plate segments"]

    # 2. Coordinate Generation
    master_positions = generate_master_positions(
        n_rotations=CONFIG["n_rotations"] if CONFIG["apply_rotations"] else 0,
        custom_angles_deg=CONFIG["custom_rotation_degrees"] if CONFIG["apply_rotations"] else [],
        translation_mode=CONFIG["translation_mode"] if CONFIG["apply_translations"] else "none",
        n_shifts=CONFIG["n_shifts"],
        shift_step_mm=CONFIG["shift_step_mm"],
        n_transaxial_positions=CONFIG["n_transaxial_positions"],
        ellipse_a_mm=CONFIG["ellipse_a_mm"],
        ellipse_b_mm=CONFIG["ellipse_b_mm"],
        outward_shift_mm=CONFIG["outward_shift_mm"],
        custom_translations=CONFIG["custom_translations"], 
    )
    n_total_positions = master_positions.shape[0]
    print(f"Generated {n_total_positions} unique acquisition positions.")

    # 3. Transform Setup (Selective Rotation/Translation)
    detector_transforms = master_positions.clone()
    collimator_transforms = master_positions.clone()

    if not CONFIG["apply_translations"]:
        detector_transforms[:, 1:] = 0.0
        collimator_transforms[:, 1:] = 0.0
        
    if not CONFIG["apply_rotations"]:
        detector_transforms[:, 0] = 0.0
        collimator_transforms[:, 0] = 0.0
    else:
        if not CONFIG["rotate_detectors"]:
            detector_transforms[:, 0] = 0.0
        if not CONFIG["rotate_collimator"]:
            collimator_transforms[:, 0] = 0.0

    # 4. Batch Processing
    print("Applying batch transformations to geometry...")
    transformed_detectors = apply_transformations_batch(base_detectors, detector_transforms)
    transformed_collimator = apply_transformations_batch(base_collimator, collimator_transforms)

    layouts = {
        f"position {i:03d}": {
            "position": master_positions[i],
            "detector units": transformed_detectors[i],
            "plate segments": transformed_collimator[i],
        } for i in range(n_total_positions)
    }

    # 5. Output Preparation and Serialization
    output_data: OutDataDict = {
        "scanner MD5": input_md5,
        "source_file": os.path.basename(args.input_file),
        "applied_config": CONFIG,
        "layouts": layouts,
    }

    # Filename Generation based on active modes
    config_tags = []
    if CONFIG["apply_rotations"]:
        config_tags.append("rotated")
    if CONFIG["apply_translations"]:
        mode_tag = CONFIG["translation_mode"] if CONFIG["translation_mode"] != 'grid' else f"{CONFIG['n_shifts'][0]}x{CONFIG['n_shifts'][1]}grid"
        config_tags.append(mode_tag)
    
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = "../data/scanner_layouts/"
    os.makedirs(output_dir, exist_ok=True)
    out_filename = os.path.join(output_dir, f"{input_basename}_{'_'.join(config_tags)}.tensor")

    print(f"Saving layouts to: {out_filename}")
    torch.save(output_data, out_filename)
    print("Process complete.")

if __name__ == "__main__":
    main()