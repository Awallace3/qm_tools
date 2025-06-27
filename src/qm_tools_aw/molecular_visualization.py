"""
Molecular visualization functions for qcelemental molecules using py3dmol.
"""

import py3Dmol
import qcelemental as qcel
import webbrowser
from typing import Optional, Dict, Any, List


def visualize_molecule(
    molecule: qcel.models.Molecule,
    style: str = "stick",
    width: int = 800,
    height: int = 600,
    background: str = "white",
    show_labels: bool = False,
    label_style: Optional[Dict[str, Any]] = None,
    fragment_colors: Optional[List[str]] = None,
    bond_radius: float = 0.15,
    sphere_radius: float = 0.30,
    stick_radius: float = 0.25,
    cartoon_color: str = "spectrum",
    surface_opacity: float = 0.7,
    surface_color: str = "blue",
    zoom_factor: float = 1.0,
    spin: dict = None,
    custom_style: Optional[Dict[str, Any]] = None,
    auto_open: bool = True,
    temp_filename: str = "molecule_viz.html",
    title: str = None,
) -> py3Dmol.view:
    """
    Create a 3D visualization of a qcelemental molecule using py3dmol.

    Parameters:
    -----------
    molecule : qcel.models.Molecule
        The qcelemental molecule object to visualize
    style : str, default="stick"
        Visualization style: "stick", "sphere", "ball_and_stick", "cartoon",
        "surface", "line", "cross"
    width : int, default=800
        Width of the visualization window
    height : int, default=600
        Height of the visualization window
    background : str, default="white"
        Background color
    show_labels : bool, default=False
        Whether to show atom labels
    label_style : dict, optional
        Custom styling for atom labels
    fragment_colors : list, optional
        List of colors for different fragments (for multi-fragment molecules)
    bond_radius : float, default=0.15
        Radius for bonds in stick representation
    sphere_radius : float, default=0.3
        Radius for atoms in sphere representation
    stick_radius : float, default=0.25
        Radius for sticks in ball_and_stick representation
    cartoon_color : str, default="spectrum"
        Color scheme for cartoon representation
    surface_opacity : float, default=0.7
        Opacity for surface representation
    surface_color : str, default="blue"
        Color for surface representation
    zoom_factor : float, default=1.0
        Zoom factor for the view
    custom_style : dict, optional
        Custom py3dmol style dictionary
    auto_open : bool, default=True
        Whether to automatically open the visualization in a web browser
    temp_filename : str, default="molecule_viz.html"
        Name for the temporary HTML file
    title : str, optional
        Title for the visualization

    Returns:
    --------
    py3Dmol.view
        The py3dmol view object
    """

    # Create viewer
    viewer = py3Dmol.view(width=width, height=height)
    viewer.setBackgroundColor(background)

    # Convert molecule to XYZ format for py3dmol
    xyz_string = _molecule_to_xyz(molecule)
    viewer.addModel(xyz_string, "xyz")

    # Apply styling based on fragments if present
    if (
        hasattr(molecule, "fragments")
        and molecule.fragments
        and len(molecule.fragments) > 1
        and fragment_colors
    ):
        _style_fragments(
            viewer,
            molecule,
            style,
            fragment_colors,
            bond_radius,
            sphere_radius,
            stick_radius,
            cartoon_color,
            surface_opacity,
            surface_color,
            custom_style,
        )
    else:
        _apply_single_style(
            viewer,
            style,
            bond_radius,
            sphere_radius,
            stick_radius,
            cartoon_color,
            surface_opacity,
            surface_color,
            custom_style,
        )

    # Add atom labels if requested
    if show_labels:
        _add_atom_labels(viewer, molecule, label_style)

    # Set zoom and center
    viewer.zoomTo()
    if zoom_factor != 1.0:
        viewer.zoom(zoom_factor)
    if spin:
        # ensure spin is dict with 'axis' and 'frequency'
        if isinstance(spin, dict) and "axis" in spin and "frequency" in spin:
            viewer.spin(spin["axis"], spin["frequency"])
        else:
            raise ValueError("Spin must be a dict with 'axis' and 'frequency' keys")

    # Allow atoms to be selected by clicking
    viewer.setClickable(True)

    if title:
        # compute position for the label to be to the side of the molecule
        title_position = molecule.geometry.mean(axis=0) * qcel.constants.bohr2angstroms
        title_position[1] -= 6.0  # Shift label to the right
        title_position[0] -= 6.0  # Shift label to the right
        viewer.addLabel(
            title,
            {
                "fontSize": 24,
                "fontColor": "black",
                "backgroundOpacity": "0.0",
                "position": {
                    "x": title_position[0],
                    "y": title_position[1],
                    "z": title_position[2],
                },
            },
        )
    # Auto-open in web browser for non-interactive environments
    if auto_open:
        viewer.write_html(temp_filename)
        # Open the HTML file in a web browser
        webbrowser.open(temp_filename, new=2)

    return viewer


def _molecule_to_xyz(molecule: qcel.models.Molecule) -> str:
    """Convert qcelemental molecule to XYZ string format."""
    lines = [str(len(molecule.atomic_numbers)), ""]

    symbols = molecule.symbols
    geometry = molecule.geometry.reshape(-1, 3)

    for i, (symbol, coord) in enumerate(zip(symbols, geometry)):
        # Convert from bohr to angstrom
        coord_ang = coord * qcel.constants.bohr2angstroms
        lines.append(
            f"{symbol} {coord_ang[0]:.6f} {coord_ang[1]:.6f} {coord_ang[2]:.6f}"
        )

    return "\n".join(lines)


def _apply_single_style(
    viewer: py3Dmol.view,
    style: str,
    bond_radius: float,
    sphere_radius: float,
    stick_radius: float,
    cartoon_color: str,
    surface_opacity: float,
    surface_color: str,
    custom_style: Optional[Dict[str, Any]],
) -> None:
    """Apply styling to the entire molecule."""

    if custom_style:
        viewer.setStyle(custom_style)
        return

    style_dict = _get_style_dict(
        style,
        bond_radius,
        sphere_radius,
        stick_radius,
        cartoon_color,
        surface_opacity,
        surface_color,
    )

    viewer.setStyle(style_dict)


def _style_fragments(
    viewer: py3Dmol.view,
    molecule: qcel.models.Molecule,
    style: str,
    fragment_colors: Optional[List[str]],
    bond_radius: float,
    sphere_radius: float,
    stick_radius: float,
    cartoon_color: str,
    surface_opacity: float,
    surface_color: str,
    custom_style: Optional[Dict[str, Any]],
) -> None:
    """Apply different styling to different fragments."""

    default_colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "magenta",
    ]
    colors = fragment_colors or default_colors

    for frag_idx, fragment in enumerate(molecule.fragments):
        color = colors[frag_idx % len(colors)]

        # Create atom selection for this fragment
        atom_indices = fragment.tolist()
        selection = {"index": atom_indices}

        if custom_style:
            style_dict = custom_style.copy()
            if "color" not in style_dict:
                style_dict["color"] = color
        else:
            style_dict = _get_style_dict(
                style,
                bond_radius,
                sphere_radius,
                stick_radius,
                cartoon_color,
                surface_opacity,
                surface_color,
            )
            style_dict["stick"]["color"] = color
        viewer.setStyle(selection, style_dict)


def _get_style_dict(
    style: str,
    bond_radius: float,
    sphere_radius: float,
    stick_radius: float,
    cartoon_color: str,
    surface_opacity: float,
    surface_color: str,
) -> Dict[str, Any]:
    """Get the style dictionary for py3dmol based on style name."""

    style_map = {
        "stick": {"stick": {"radius": bond_radius}},
        "sphere": {"sphere": {"radius": sphere_radius}},
        "ball_and_stick": {
            "stick": {"radius": stick_radius},
            "sphere": {"radius": sphere_radius},
        },
        "line": {"line": {}},
        "cross": {"cross": {"radius": 0.1}},
        "cartoon": {"cartoon": {"color": cartoon_color}},
        "surface": {"surface": {"opacity": surface_opacity, "color": surface_color}},
    }

    return style_map.get(style, {"stick": {"radius": bond_radius}})


def _add_atom_labels(
    viewer: py3Dmol.view,
    molecule: qcel.models.Molecule,
    label_style: Optional[Dict[str, Any]],
) -> None:
    """Add atom labels to the visualization."""

    default_label_style = {
        "font": "sans-serif",
        "fontSize": 12,
        "fontColor": "white",
        "backgroundColor": "black",
        "backgroundOpacity": 0.8,
        "borderThickness": 1.0,
        "borderColor": "black",
        "borderOpacity": 1.0,
    }

    if label_style:
        default_label_style.update(label_style)

    geometry = molecule.geometry.reshape(-1, 3) * qcel.constants.bohr2angstroms

    for i, (symbol, coord) in enumerate(zip(molecule.symbols, geometry)):
        default_label_style["position"] = {"x": coord[0], "y": coord[1], "z": coord[2]}
        viewer.addLabel(f"{symbol}{i + 1}", default_label_style)


def _add_unit_cell(viewer: py3Dmol.view, molecule: qcel.models.Molecule) -> None:
    """Add unit cell visualization (placeholder for future implementation)."""
    # This would require additional information about the unit cell
    # which is not typically available in standard qcelemental molecules
    pass


def visualize_multiple_molecules(
    molecules: List[qcel.models.Molecule],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    **kwargs,
) -> py3Dmol.view:
    """
    Visualize multiple molecules in the same view.

    Parameters:
    -----------
    molecules : list of qcel.models.Molecule
        List of molecules to visualize
    labels : list of str, optional
        Labels for each molecule
    colors : list of str, optional
        Colors for each molecule
    **kwargs
        Additional arguments passed to visualize_molecule

    Returns:
    --------
    py3Dmol.view
        The py3dmol view object
    """

    if not molecules:
        raise ValueError("At least one molecule must be provided")

    # Use the first molecule to set up the viewer
    viewer = visualize_molecule(molecules[0], **kwargs)

    default_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    colors = colors or default_colors

    # Add additional molecules
    for i, mol in enumerate(molecules[1:], 1):
        xyz_string = _molecule_to_xyz(mol)
        viewer.addModel(xyz_string, "xyz")

        color = colors[i % len(colors)]
        style = kwargs.get("style", "stick")
        style_dict = _get_style_dict(
            style,
            kwargs.get("bond_radius", 0.15),
            kwargs.get("sphere_radius", 0.3),
            kwargs.get("stick_radius", 0.25),
            kwargs.get("cartoon_color", "spectrum"),
            kwargs.get("surface_opacity", 0.7),
            kwargs.get("surface_color", "blue"),
        )
        style_dict["color"] = color

        viewer.setStyle({"model": i}, style_dict)

    viewer.zoomTo()
    return viewer


def save_visualization(
    viewer: py3Dmol.view, filename: str, format: str = "png"
) -> None:
    """
    Save the visualization to a file.

    Parameters:
    -----------
    viewer : py3Dmol.view
        The py3dmol view object
    filename : str
        Output filename
    format : str, default="png"
        Output format ("png" or "html")
    """

    if format.lower() == "png":
        viewer.png()
    elif format.lower() == "html":
        with open(filename, "w") as f:
            f.write(viewer._make_html())
    else:
        raise ValueError(f"Unsupported format: {format}")


# Example usage and demonstration
if __name__ == "__main__":
    # Example with the water dimer from the original file
    mol_dimer = qcel.models.Molecule.from_data("""
    0 1
    O 0.000000 0.000000  0.000000
    H 0.758602 0.000000  0.504284
    H 0.260455 0.000000 -0.872893
    --
    0 1
    O 3.000000 0.000000  0.000000
    H 3.758602 0.000000  0.504284
    H 3.260455 0.000000 -0.872893
    """)

    # Basic visualization
    print("Creating basic stick visualization...")
    viewer1 = visualize_molecule(mol_dimer, style="stick")
    viewer1.show()

    # Ball and stick with fragment colors
    print("Creating ball and stick visualization with fragment colors...")
    viewer2 = visualize_molecule(
        mol_dimer,
        style="ball_and_stick",
        # fragment_colors=["red", "blue"],
        show_labels=True,
        spin={"axis": "x", "frequency": 0.1},
    )
    viewer2.show()


def create_latex_table_pymol(
    filename, # latex filename
    df, # pandas DataFrame with qcel molecules
    df_qcel_column="qcel_molecule",
    df_err_column=None,
    df_id_column="system_id",
    output_directory="mol_viz",
    title_include_id=True,
    visualize=False,
    zoom=5,
):
    """
    Create a LaTeX table with PyMOL molecular visualizations.
    
    This function generates a LaTeX table that includes molecular visualizations
    created using PyMOL from the molecular data in the provided DataFrame.
    
    Parameters
    ----------
    filename : str
        The name of the output LaTeX file (e.g., 'mol_vis.tex').
    dataframe : pandas.DataFrame
        DataFrame containing molecular data with 'qcel_molecule' column
        and other molecular properties.
    df_id_column : str
        The column name in the DataFrame to use as identifiers for the
        molecular systems (e.g., 'system_id').
    output_directory : str
        Directory path where the output files and molecular visualizations
        will be saved.
    zoom : float
        Zoom factor for the PyMOL visualization.
    title_include_id : bool
        Whether to include the identifier in the title of each molecule
        in the LaTeX table.
    visualize : bool
        Whether to generate molecular visualizations using PyMOL.
    df_qcel_column : str
        The column name in the DataFrame that contains the qcelemental
        molecule objects (default is 'qcel_molecule').
    df_err_column : str, optional
        The column name in the DataFrame that contains error values
        for each molecule (default is None, meaning no error values are included).
        
    Returns
    -------
    None
        The function saves the LaTeX table to the specified filename
        and molecular visualization files to the output directory.
        After running this, cd to output_directory and run `bash make-images.sh`
        followed by `pdflatex filename.tex` to generate the PDF with images.
    """
    if df_id_column is None and title_include_id:
        print(
            "Warning: df_id_column is None, but title_include_id is True. Setting title_include_id to False."
        )
        title_include_id = False
    with open(f"./{output_directory}/{filename}", "w") as tex:
        tex.write(r"""
\documentclass{article}
\usepackage{longtable}
\usepackage{adjustbox} % For adjusting image sizes, needed for ion-table
\usepackage{makecell}
\usepackage{setspace}
\usepackage[margin=1in,footskip=0.25in]{geometry}

\begin{document}
\begin{longtable}{|c|c|c|c|}
""")
        set_of_four = []
        cnt = 0
        for i, row in df.iterrows():
            error_value = ""
            if df_err_column:
                error_value = f", {row[df_err_column]:.2f}"
            if title_include_id:
                title = f"{i},{row[df_id_column]}{error_value}"
            else:
                title = f"{i}{error_value}"
            if visualize:
                visualize_molecule(
                    row[df_qcel_column],
                    style="ball_and_stick",
                    title=title,
                    temp_filename=f"./{output_directory}/{row[df_id_column]}.html",
                )
            xyz = row[df_qcel_column].to_string("xyz")
            with open(f"./{output_directory}/{row[df_id_column]}.xyz", "w") as f:
                f.write(xyz)
            set_of_four.append(
                [
                    row[df_id_column]
                    .replace("_", "\\_")
                    .replace("[", "\\[")
                    .replace("]", "\\]"),
                    f"./{row[df_id_column]}.png",
                    error_value,
                ]
            )
            cnt += 1
            if cnt == 4:
                tex.write(
                    "\\adjustbox{valign=t}{\\includegraphics[width=0.22\\textwidth]{"
                    + set_of_four[0][1]
                    + "}}& "
                    + "\\adjustbox{valign=t}{\\includegraphics[width=0.22\\textwidth]{"
                    + set_of_four[1][1]
                    + "}} & "
                    + "\\adjustbox{valign=t}{\\includegraphics[width=0.22\\textwidth]{"
                    + set_of_four[2][1]
                    + "}} & "
                    + "\\adjustbox{valign=t}{\\includegraphics[width=0.22\\textwidth]{"
                    + set_of_four[3][1]
                    + "}} \\\\\n"
                )
                e0 = f"\\\\Err={set_of_four[0][-1]:.2f}" if error_value != "" else ""
                e1 = f"\\\\Err={set_of_four[1][-1]:.2f}" if error_value != "" else ""
                e2 = f"\\\\Err={set_of_four[2][-1]:.2f}" if error_value != "" else ""
                e3 = f"\\\\Err={set_of_four[3][-1]:.2f}" if error_value != "" else ""
                tex.write(
                    "\\tiny \\makecell{"
                    + set_of_four[0][0]
                    + e0
                    + "} & "
                    + "\\tiny \\makecell{"
                    + set_of_four[1][0]
                    + e1
                    + "} & "
                    + "\\tiny \\makecell{"
                    + set_of_four[2][0]
                    + e2
                    + "} & "
                    + "\\tiny \\makecell{"
                    + set_of_four[3][0]
                    + e3
                    + "} \\\\\n"
                )
                tex.write("\\hline\n")
                set_of_four = []
                cnt = 0

        tex.write(r"""
\end{longtable}

\end{document}
""")
    make_images = f"./{output_directory}/make-images.sh"
    with open(make_images, "w") as f:
        f.write("""#!/bin/bash
direct=.
format='xyz'
for file in *.$format;
do
	fname=$(basename "$file" .$format)
	echo "Processing $fname"
	sed "s/INSERT/$fname/g" script.pml > "$fname"-script.pml
	pymol -cqr "$fname"-script.pml
	# rm "$file"-script.pml
done""")
    with open(f"./{output_directory}/script.pml", "w") as f:
        f.write(f"""
load INSERT.xyz

color grey, elem c
label all, elem
show spheres,*
set sphere_scale, 0.25
show stick,*
set_bond stick_radius, 0.2, v.
set stick_h_scale,1
zoom center,{zoom}
center v.
rotate x, 5
set ray_opaque_background, off
set opaque_background, off

ray 2000,2000
png INSERT.png, dpi=200
                """)
    return
