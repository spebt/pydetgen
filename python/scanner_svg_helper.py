from lxml import etree
import torch
import re


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


def get_verts_sorted_by_angle_2d(
    vertices: torch.Tensor, ref_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # sort the vertices by angle to point ref_point
    rads = torch.atan2(
        vertices[:, 1] - ref_point[1], vertices[:, 0] - ref_point[0]
    )
    rads = (rads + 2 * torch.pi) % (2 * torch.pi)
    order = torch.argsort(rads)
    return vertices[order], order


def get_verts_sorted_by_xy_2d(
    verts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # sort the vertices by x
    indices_on_x = torch.argsort(verts[:, 0])
    verts = verts[indices_on_x]
    # sort the vertices again by y if x is the same
    indices_on_y = torch.argsort(verts[:, 1])
    verts = verts[indices_on_y]
    return verts, indices_on_x[indices_on_y]


def create_svg_polygon_group(
    group_attrib: dict, group_nsmap: dict, polygons: torch.Tensor, **kwargs
):
    group = etree.Element(
        "g",
        attrib=group_attrib,
        nsmap=group_nsmap,
    )
    prefix = "polygon"
    try:
        prefix = kwargs["polygon_prefix"]
    except KeyError:
        prefix = "polygon"
    for id, polygon in enumerate(polygons):
        vertices, _ = get_verts_sorted_by_xy_2d(polygon)
        vertices = get_verts_sorted_by_angle_2d(vertices, vertices[0])[0]

        # flip y axis
        vertices[:, 1] = -vertices[:, 1]
        vertices = vertices + kwargs["global_offset"]

        polygon_attrib = {
            "id": f"{prefix}_{id}",
            "points": " ".join(f"{x},{y}" for x, y in vertices),
            # "fill": "gray",
            # "stroke": "none",
        }
        polygon_element = etree.Element(
            "polygon",
            attrib=polygon_attrib,
            nsmap=group_nsmap,
        )
        group.append(polygon_element)
    return group


def create_svg_path(
    id: int, vertices: torch.Tensor, prefix: str, namespace: dict
):

    commands = ["M"]
    commands.append(f"{vertices[0][0]},{vertices[0][1]}")
    for vertex in vertices[1:]:
        commands.append(f"L{vertex[0]},{vertex[1]}")
    commands.append("Z")

    path_attrib = {
        "id": f"{prefix}_{id}",
        "d": " ".join(commands),
    }
    return etree.Element(
        "path",
        attrib=path_attrib,
        nsmap=namespace,
    )


def create_svg_path_group(
    group_attrib: dict, group_nsmap: dict, polygons: torch.Tensor, **kwargs
):
    group = etree.Element(
        "g",
        attrib=group_attrib,
        nsmap=group_nsmap,
    )
    prefix = "path"
    try:
        prefix = kwargs["prefix"]
    except KeyError:
        prefix = "path"
    for id, polygon in enumerate(polygons):
        vertices = get_verts_sorted_by_xy_2d(polygon)
        vertices, _ = get_verts_sorted_by_angel_2d(vertices, vertices[0])

        # flip y axis
        vertices[:, 1] = -vertices[:, 1]
        vertices = vertices + kwargs["global_offset"]

        path_element = create_svg_path(id, vertices, prefix, group_nsmap)
        group.append(path_element)
    return group


def create_svg_scanner(
    outfname: str,
    **kwargs,
) -> None:
    try:
        crystal_polygons = kwargs["crystal_polygons"]
        plate_polygons = kwargs["plate_polygons"]
    except KeyError as e:
        raise ValueError(f"Missing required argument: {e}")

    width = kwargs.get("width", 400)
    height = kwargs.get("height", 400)
    plate_color = kwargs.get("plate_color", "gray")
    crystal_color = kwargs.get("crystal_color", "orange")
    root_attrib = {
        "width": "%s" % width,
        "height": "%s" % height,
        "viewBox": f"0 0 {width} {height}",
    }
    global_offset = torch.tensor([width, height]) / 2
    root_namespace = {None: "http://www.w3.org/2000/svg"}
    root = etree.Element("svg", attrib=root_attrib, nsmap=root_namespace)
    background_attrib = {
        "id": "background",
        "x": "0",
        "y": "0",
        "width": "%s" % width,
        "height": "%s" % height,
        "fill": "white",
    }
    background_element = etree.Element(
        "rect",
        attrib=background_attrib,
        nsmap=root_namespace,
    )
    root.append(background_element)
    root.append(
        create_svg_path_group(
            {
                "id": "crystal group",
                "fill": crystal_color,
            },
            root_namespace,
            crystal_polygons,
            global_offset=global_offset,
            prefix="crystal",
        )
    )
    root.append(
        create_svg_path_group(
            {
                "id": "plate group",
                "fill": plate_color,
            },
            root_namespace,
            plate_polygons,
            global_offset=global_offset,
            prefix="plate",
        )
    )
    tree = etree.ElementTree(root)
    tree.write(
        outfname, pretty_print=True, xml_declaration=False, encoding="UTF-8"
    )
    return None
