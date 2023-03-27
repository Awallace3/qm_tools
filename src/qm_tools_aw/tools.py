import pickle
import numpy as np
from periodictable import elements
import qcelemental as qcel


def create_pt_dict():
    """
    create_pt_dict creates dictionary for string elements to atomic number.
    """
    el_dc = {}
    for el in elements:
        el_dc[el.symbol] = el.number
    return el_dc


def create_el_num_to_symbol():
    """
    create_pt_dict creates dictionary for string elements to atomic number.
    """
    el_dc = {}
    for el in elements:
        el_dc[el.number] = el.symbol
    return el_dc


def np_carts_to_string(carts):
    w = ""
    for n, r in enumerate(carts):
        e, x, y, z = r
        line = "{:d}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(int(e), x, y, z)
        w += line
    return w


def string_carts_to_np(geom):
    geom = geom.split("\n")
    if geom[0] == "":
        geom = geom[1:]
    mols = []
    m = []
    charges = [[0, 1]]
    new_mol = False
    el_dict = create_pt_dict()
    monA, monB = [], []
    for n, i in enumerate(geom):
        if n == 0:
            cnt = 0
            m_ind = []
            i = [int(k) for k in i.strip().split(" ")]
            charges.append(i)
        elif new_mol:
            i = [int(k) for k in i.strip().split(" ")]
            charges.append(i)
            new_mol = False
        elif "--" in i:
            new_mol = True
            # mols.append(m)
            monA = m_ind
            m_ind = []
        else:
            i = (
                i.replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
            ).rstrip()
            i = i.split(" ")
            if i[1].isnumeric():
                el = int(i[1])
            else:
                el = el_dict[i[1]]
            r = [
                el,
                float(i[2]),
                float(i[3]),
                float(i[4]),
            ]
            m.append(np.array(r))
            m_ind.append(cnt)
            cnt += 1
    m = np.array(m)
    monB = m_ind
    charges = np.array(charges)
    monA = np.array(monA)
    monB = np.array(monB)
    return m, charges, monA, monB


def print_cartesians(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for i, elem in enumerate(a):
            if i == 0:
                print("{} ".format(int(elem)), end="\t")
            else:
                print("{:.10f} ".format(elem).rjust(3), end="\t")
        print(end="\n")


def print_cartesians_dimer(geom, monAs, monBs, charges) -> str:
    """
    print_cartesians_dimer takes in dimer geometry and splits
    by monAs and monBs slicing to produce monomers. The
    charges are mult and charge.
    """
    m1 = geom[monAs]
    m2 = geom[monBs]
    c1, c2 = charges[1], charges[2]
    print(*c1)
    print_cartesians(m1)
    print(f"--")
    print(*c2)
    print_cartesians(m2)
    return


def print_cartesians_pos_carts(pos: np.array, carts: np.array):
    """
    prints a 2-D numpy array in a nicer format
    """
    print()
    lines = ""
    for n, r in enumerate(carts):
        x, y, z = r
        line = "{}\t{:.10f}\t{:.10f}\t{:.10f}".format(int(pos[n]), x, y, z)
        lines += line + "\n"
        print(line)
    print()
    return lines


def write_cartesians_to_xyz(pos: np.array, carts: np.array, fn="out.xyz"):
    """
    creates xyz file from pos and carts
    """
    el_dc = create_el_num_to_symbol()
    out = ""
    with open(fn, "w") as f:
        f.write(f"{len(pos)}\n\n")
        for n, r in enumerate(carts):
            x, y, z = r
            line = "{}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(el_dc[int(pos[n])], x, y, z)
            out += line
            f.write(line)
    return out


def write_pickle(data, fname="data.pickle"):
    with open(fname, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fname="data.pickle"):
    with open(fname, "rb") as handle:
        return pickle.load(handle)


def read_xyz_to_pos_carts(xyz_path="mol.xyz") -> (np.array, np.array):
    """
    read_xyz_to_pos_carts reads xyz file and returns pos and carts
    """
    el_dc = create_pt_dict()
    with open(xyz_path, "r") as f:
        d = f.readlines()[2:]

    pos, carts = [], []
    for l in d:
        l = l.split()
        el = el_dc[l[0]]
        x, y, z = float(l[1]), float(l[2]), float(l[3])
        pos.append(el)
        carts.append([x, y, z])
    return np.array(pos), np.array(carts)


def convert_geom_str_to_dimer_splits(geom, units_angstroms=True) -> [np.array, np.array, np.array, np.array]:

    """
    convert_str_to_dimer_splits takes in geom as a STRING as a list or single string
    and makes Molecule objects

    returning order [ZA, ZB, RA, RB]
    """
    m = 1
    if units_angstroms:
        m = qcel.constants.conversion_factor("bohr", "angstrom")
    if type(geom) == str:
        mol = qcel.models.Molecule.from_data(geom)
        RA = mol.geometry[mol.fragments[0]] * m
        RB = mol.geometry[mol.fragments[1]] * m
        ZA = mol.atomic_numbers[mol.fragments[0]]
        ZB = mol.atomic_numbers[mol.fragments[1]]
        TQA = mol.fragment_charges[0]
        TQB = mol.fragment_charges[1]
        # MA = mol.fragment_multiplicity[mol.fragments[0]]
        # MB = mol.fragment_multiplicity[mol.fragments[1]]
        return [ZA, ZB, RA, RB, TQA, TQB]
    elif type(geom) == list:
        out = []
        for i in geom:
            mol = qcel.models.Molecule.from_data(i)
            RA = mol.geometry[mol.fragments[0]] * m
            RB = mol.geometry[mol.fragments[1]] * m
            ZA = mol.atomic_numbers[mol.fragments[0]]
            ZB = mol.atomic_numbers[mol.fragments[1]]
            TQA = mol.fragment_charges[0]
            TQB = mol.fragment_charges[1]
            # MA = mol.fragment_multiplicity[mol.fragments[0]]
            # MB = mol.fragment_multiplicity[mol.fragments[1]]
            out.append([ZA, ZB, RA, RB, TQA, TQB])
        return out
    else:
        print("Type not supported")
        return []
