from lxml import etree
import torch
import re
import pandas as pd


def parse_svg_path_command(command: str) -> list:

    # command = "m 323.17999,225.39334 h 3 v -3 h -3 z"
    # command = "m 288.93524,143.95963 0.5,-0.86603 -6.92822,-4 -0.5,0.86603 z"
    pattern = re.compile(r"([a-zA-Z])\s*([0-9\.\-eE]+(?:[\s,][0-9\.\-eE]+)*)")
    matches = pattern.findall(command)
    verts = []
    for match in matches:
        if match[0].lower() == "m":
            verts_str = match[1].split(" ")
            start_coords = verts_str[0].split(",")
            verts.append([float(start_coords[0]), float(start_coords[1])])
            for vert_str in verts_str[1:]:
                verts.append(
                    [
                        verts[-1][0] + float(vert_str.split(",")[0]),
                        verts[-1][1] + float(vert_str.split(",")[1]),
                    ]
                )
        elif match[0] == "L":
            verts_str = match[1].split(" ")
            for vert_str in verts_str:
                verts.append(
                    [
                        float(vert_str.split(",")[0]),
                        float(vert_str.split(",")[1]),
                    ]
                )
        elif match[0] == "h":
            verts.append([verts[-1][0] + float(match[1]), verts[-1][1]])
        elif match[0] == "v":
            verts.append([verts[-1][0], verts[-1][1] + float(match[1])])
        elif match[0] == "V":
            verts.append([verts[-1][0], float(match[1])])
        elif match[0] == "H":
            verts.append([float(match[1]), verts[-1][1]])
        elif match[0] == "l":
            verts_str = match[1].split(" ")
            for vert_str in verts_str:
                verts.append(
                    [
                        verts[-1][0] + float(vert_str.split(",")[0]),
                        verts[-1][1] + float(vert_str.split(",")[1]),
                    ]
                )
    return verts


def get_verts_from_svg_path(path) -> list[list[float]]:

    path_commands = path.get("d")
    path_id = path.get("id")
    try:
        verts = parse_svg_path_command(path_commands)
    except Exception as e:
        raise ValueError(f"Error parsing path {path_id}: {e}")
    return verts


def get_width_height_from_svg(svg) -> torch.Tensor:
    width = re.search(r"\d+", svg.get("width")).group()
    height = re.search(r"\d+", svg.get("height")).group()
    return torch.tensor([float(width), float(height)])


def get_polygons_from_svg_group(svg, group_name: str) -> torch.Tensor:
    group = svg.xpath(
        "//svg:g[@id='{}']".format(group_name),
        namespaces={"svg": "http://www.w3.org/2000/svg"},
    )
    polygons = []
    for element in group:
        for path in element.xpath(
            "svg:path", namespaces={"svg": "http://www.w3.org/2000/svg"}
        ):
            try:
                verts = get_verts_from_svg_path(path)
                if len(verts) != 4:
                    raise ValueError(
                        f"Error parsing path {path.get('id')}: {verts}"
                    )
            except Exception as e:
                raise ValueError(f"Error parsing path {path.get('id')}: {e}")

            polygons.append(verts)
    return torch.tensor(polygons)


def get_scanner_geometry_from_svg(
    svg_fname: str, group_names: list[str]
) -> dict[str, torch.Tensor]:
    # Load the scanner SVG file and read the path into vertices
    xml_data_bytes = open(svg_fname, "rb").read()
    parser = etree.XMLParser(remove_blank_text=True)
    svg = etree.XML(xml_data_bytes, parser)
    offset = get_width_height_from_svg(svg) / 2
    out = {}
    for group_name in group_names:
        try:
            raw_verts = get_polygons_from_svg_group(svg, group_name)
            out[group_name] = raw_verts - offset
        except Exception as e:
            raise ValueError(f"Error parsing group {group_name}: {e}")
    return out


def get_polygon_groups_from_csv(grp_names: list[str], csv_path: str):
    df = pd.read_csv(csv_path)
    for grp_name in grp_names:
        grp_verts = df.loc[df["polygon group"] == grp_name]
        grp_tensor = torch.empty(
            grp_verts.shape[0] // 4, 4, 2, dtype=torch.float
        )
        # print(grp_verts["polygon id"].unique())
        grp_tensor[
            torch.tensor(grp_verts["polygon id"].values, dtype=torch.int),
            torch.tensor(grp_verts["vertice id"].values, dtype=torch.int),
            :,
        ] = torch.tensor(grp_verts[["x", "y"]].values, dtype=torch.float)
        yield grp_tensor


def load_scanner_geometry_csv(csv_path: str):
    """
    Load the scanner geometry from a CSV file.
    Arguments:
        csv_path (str): Path to the CSV file containing the scanner geometry.
    Returns:
        tuple: A tuple containing:
            - plate_polygon_tensor (torch.Tensor): Tensor of shape (num_polygons, 4, 2)
                representing the plate polygons.
            - xtal_polygon_tensor (torch.Tensor): Tensor of shape (num_polygons, 4, 2)
                representing the crystal polygons.
    """
    # Get the plate polygons from the CSV file
    group_names = ["plate_{}".format(i) for i in range(6)]
    plate_polygon_tensor = torch.cat(
        list(get_polygon_groups_from_csv(group_names, csv_path))
    )
    group_names = ["crystals_{}".format(i) for i in range(6)]
    xtal_polygon_tensor = torch.cat(
        list(get_polygon_groups_from_csv(group_names, csv_path))
    )
    return plate_polygon_tensor, xtal_polygon_tensor
