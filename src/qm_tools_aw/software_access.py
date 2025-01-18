import numpy as np
import qcelemental as qcel
import os
from pathlib import Path


def run_checkmol(mol_file, checkmol_path='checkmol'):
    """Runs Checkmol on a given MOL file and returns the output file path."""
    p = Path(mol_file)
    mol_base_path = p.parent / p.stem
    print(mol_base_path)
    checkmol_file = str(mol_base_path) +  ".checkmol"
    os.system(f"{checkmol_path} -vxpe {mol_file} > {checkmol_file}")
    return checkmol_file


def compute_closest_hydrogens(active_ind_distances, H_atom_indices, Htot):
    """Computes the closest H_atoms to a given atom index based on the distance matrix."""
    H_distances = active_ind_distances[H_atom_indices]
    closest_H_indices = np.argsort(H_distances)[:int(Htot)]
    return H_atom_indices[closest_H_indices]


def process_checkmol_data(checkmol_file, verbose=False):
    """
    Processes the output of a Checkmol file and returns a dictionary with the following keys:
    - atomic_numbers
    - atomic_types
    - aromatic
    - heavy_atom_neighbors
    - Hexps
    - Htots
    - functional_group_indices
    - functional_group_labels
    """
    with open(checkmol_file, "r") as f:
        lines = f.readlines()
    if verbose:
        print("".join(lines))
    xyz = []
    atomic_numbers = []
    atomic_types = []
    aromatic_label = []
    functional_group_labels = []
    functional_group_indices = []
    heavy_atom_neighbors = []
    Hexps = []
    Htots = []
    H_atom_indices = []
    for line in lines:
        if "heavy-atom" in line:
            line = line.split()
            ind, atom, atom_type, x, y, z, hatom_neighbors, Hexp, Htot = line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[10], line[12]
            aromatic = "aromatic" in line
            xyz.append([float(x), float(y), float(z)])
            atomic_numbers.append(qcel.periodictable.to_Z(atom))
            atomic_types.append(atom_type)
            aromatic_label.append(aromatic)
            heavy_atom_neighbors.append(int(hatom_neighbors.replace("(", "")))
            Hexps.append(int(Hexp))
            Htots.append(int(Htot.replace(")", "")))
            if "H" in atom:
                H_atom_indices.append(int(ind) - 1)
        elif line.startswith("#"):
            split_line = line.split(":")
            functional_group_label = ":".join(split_line[0:2])
            indices = ":".join(split_line[2:]).strip()
            for n, p in enumerate(indices.split(",")):
                output_indices = []
                explicit_indices = p.split("-")
                if explicit_indices[-1] == '':
                    explicit_indices = explicit_indices[:-1]
                # checkmol uses 1-indexing but need 0-indexing for np
                for i in explicit_indices:
                    if ":" in i:
                        start, end = i.split(":")
                        output_indices += list(range(int(start) - 1, int(end)))
                    else:
                        output_indices.append(int(i) - 1)
                if n > 0:
                    # check to see if any output_indices overlap with previous output_indices, if so merge
                    if np.any(np.isin(output_indices, functional_group_indices[-1])):
                        functional_group_indices[-1] = np.concatenate((functional_group_indices[-1], output_indices))
                        continue
                functional_group_indices.append(output_indices)
                functional_group_labels.append(functional_group_label)
    H_atom_indices = np.array(H_atom_indices)
    xyz = np.array(xyz)
    atomic_numbers = np.array(atomic_numbers)
    distance_matrix = np.linalg.norm(xyz[:, None] - xyz, axis=-1)
    # H_distances = distance_matrix[H_atom_indices]
    # Now we have all information to identify which hydrogens belong to certain
    # functional groups. For every non-zero Htots element, we will use Htot
    # index to identify the closest H_atoms to the current atom. Then grab the
    # Htot number of closest H_atoms indices to append to the functional group
    # indices.
    for n, f_indices in enumerate(functional_group_indices):
        if verbose:
            print(n, f_indices)
            print(f"Functional group: {functional_group_labels[n]}")
            print(f"Functional group indices: {functional_group_indices[n]}")
        for i in f_indices:
            Htot = Htots[i]
            if Htot != 0:
                ind_distances = distance_matrix[i]
                closest_H_indices = compute_closest_hydrogens(ind_distances, H_atom_indices, Htot)
                functional_group_indices[n] = np.concatenate((functional_group_indices[n], closest_H_indices))
                functional_group_indices[n] = np.unique(functional_group_indices[n])
                if verbose:
                    print("Active index:", i, "Htot:", Htot)
                    print(closest_H_indices)
        if verbose:
            print(f"Functional group indices: {functional_group_indices[n]}")
    return {
        "atomic_numbers": atomic_numbers,
        "atomic_types": atomic_types,
        "aromatic": aromatic_label,
        "heavy_atom_neighbors": heavy_atom_neighbors,
        "Hexps": Hexps,
        "Htots": Htots,
        "functional_group_indices": functional_group_indices,
        "functional_group_labels": functional_group_labels,
    }
