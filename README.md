# QM-Tools-AW

A collection of quantum mechanics and computational chemistry tools for data analysis, visualization, and manipulation.

## Installation

```bash
pip install qm-tools-aw
```

## Features

### General Utilities
- File I/O operations (JSON, pickle)
- Molecular geometry manipulation and conversion
- Handling of various file formats (XYZ, PDB, Psi4 input)
- Distance calculations and molecular structure analysis
- Visualization helpers (e.g., PyMOL script generation)

### Plot Utilities
- Violin plots for error analysis with customizable aesthetics
- Support for statistical visualizations of computational chemistry data

### SAPT Analysis
- Symmetry-Adapted Perturbation Theory (SAPT) term computation for higher-order
SAPT mimicking Psi4 but on a pandas DataFrame

### Software Access
- Interface with external chemistry software
- Tools for working with `checkmol` for functional group analysis

## Usage Examples

### Create Violin Plots for Error Analysis

```py
import pandas as pd
import numpy as np
from qm_tools_aw.plot import violin_plot

# Create sample data
df = pd.DataFrame({
    'MP2': np.random.normal(0.5, 5, 1000),
    'HF': np.random.normal(-0.5, 5, 1000),
})

# Create violin plot
violin_plot(
    df,
    {'MP2 Method': 'MP2', 'HF Method': 'HF'},
    output_filename="method_comparison",
    ylim=[-25, 25],
    plt_title="Method Error Comparison"
)
```

*Results:* `Plotting method_comparison`

### Process Molecular Geometries

```py
from qm_tools_aw.tools import read_xyz_to_pos_carts, write_cartesians_to_xyz, print_cartesians_pos_carts

# Read XYZ file
pos, carts = read_xyz_to_pos_carts("molecule.xyz")

# Manipulate coordinates
carts[:, 0] += 1.0  # Shift x coordinates

# Write modified structure
write_cartesians_to_xyz(pos, carts, "modified.xyz")
print_cartesians_pos_carts(pos, carts)
```

*Results:*
```

8	0.2978039460	-0.0560602560	0.0099422620
1	-0.0221932240	0.8467757820	-0.0114887140
1	1.2575210620	0.0421214960	0.0052189990

```

## Documentation

For detailed documentation on all functions and classes, refer to the docstrings in the source code.

## Dependencies

- NumPy
- Pandas
- Matplotlib
- QCElemental
- Other standard Python libraries

## License

This project is licensed under the MIT License - see the LICENSE file for details.
