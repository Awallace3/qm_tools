import pickle
import numpy as np
from periodictable import elements
import qcelemental as qcel
import pandas as pd


def save_pkl(file_name, obj):
    with open(file_name, "wb") as fobj:
        pickle.dump(obj, fobj)


def load_pkl(file_name):
    with open(file_name, "rb") as fobj:
        return pickle.load(fobj)


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
        line = "{:d}\t{:.10f}\t{:.10f}\t{:.10f}".format(int(e), x, y, z)
        if n != len(carts) - 1:
            line += "\n"
        w += line
    return w


def generate_p4input_from_df(geometry, charges, monAs, monBs, units="angstrom"):
    ma, mb = [], []
    for i in monAs:
        ma.append(geometry[i, :])
    for i in monBs:
        mb.append(geometry[i, :])
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
    geom = f"{charges[1][0]} {charges[1][1]}\n{ma}"
    geom += f"\n--\n{charges[2][0]} {charges[2][1]}\n{mb}"
    if units == "angstrom":
        geom += "\nunits angstrom"
    elif units == "bohr":
        geom += "\nunits bohr"
    else:
        raise ValueError("units must be either angstrom or bohr")
    return geom



def convert_schr_row_to_mol(r) -> qcel.models.Molecule:
    """
    convert_schr_row_to_mol
    """
    ma, mb = r["monAs"], r["monBs"]
    g1, g2 = r["Geometry"][ma], r["Geometry"][mb]
    m1 = f"{r['charges'][1][0]} {r['charges'][1][1]}\n"
    m1 += print_cartesians_pos_carts(g1[:, 0], g1[:, 1:], only_results=True)
    m2 = f"{r['charges'][2][0]} {r['charges'][2][1]}\n"
    m2 += print_cartesians_pos_carts(g2[:, 0], g2[:, 1:], only_results=True)
    mol = qcel.models.Molecule.from_data(m1 + "--\n" + m2)
    return mol


def convert_pos_carts_to_mol(pos, carts):
    m1 = ""
    for i in range(len(pos)):
        if i > 0:
            m1 += "--\n"
        m1 += f"0 1\n"
        m1 += print_cartesians_pos_carts(pos[i], carts[i], only_results=True)
    mol = qcel.models.Molecule.from_data(m1)
    return mol


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


def print_cartesians(arr, symbols=False):
    """
    prints a 2-D numpy array in a nicer format
    """
    shape = np.shape(arr)
    if shape[1] != 4:
        raise ValueError("Array must be Nx4")

    l = ""
    for a in arr:
        for i, elem in enumerate(a):
            if i == 0:
                elem = qcel.periodictable.to_E(int(elem))
                print("{} ".format(elem), end="\t")
                l += "{} \t".format(elem)
            else:
                print("{:.10f} ".format(elem).rjust(3), end="\t")
                l += "{:.10f} ".format(elem).rjust(3)
                l += "\t"
        print(end="\n")
        l += "\n"
    return l


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


def print_cartesians_pos_carts(
    pos: np.array, carts: np.array, only_results=False, el_attach=None
):
    """
    prints a 2-D numpy array in a nicer format
    """
    if not only_results:
        print()
    lines = ""
    for n, r in enumerate(carts):
        x, y, z = r
        el = str(int(pos[n]))
        if el_attach is not None:
            el += el_attach
        line = "{}\t{:.10f}\t{:.10f}\t{:.10f}".format(el, x, y, z)
        lines += line + "\n"
        if not only_results:
            print(line)
    if not only_results:
        print()
    return lines


def print_cartesians_pos_carts_symbols(
    pos: np.array,
    carts: np.array,
    only_results=False,
    el_attach=None,
    el_dc=create_el_num_to_symbol(),
):
    """
    prints a 2-D numpy array in a nicer format
    """
    if not only_results:
        print()
    lines = ""
    for n, r in enumerate(carts):
        x, y, z = r
        el = el_dc[int(pos[n])]
        if el_attach is not None:
            el += el_attach
        line = "{}\t{:.10f}\t{:.10f}\t{:.10f}".format(el, x, y, z)
        lines += line + "\n"
        if not only_results:
            print(line)
    if not only_results:
        print()
    return lines


# def return_cartesians_pos_carts(pos: np.array, carts: np.array):
#     """
#     prints a 2-D numpy array in a nicer format
#     """
#     print()
#     lines = ""
#     for n, r in enumerate(carts):
#         x, y, z = r
#         line = "{}\t{:.10f}\t{:.10f}\t{:.10f}".format(int(pos[n]), x, y, z)
#         lines += line + "\n"
#         print(line)
#     print()
#     return lines


def carts_to_xyz(pos: np.array, carts: np.array, el_dc=create_el_num_to_symbol()):
    """
    creates xyz file from pos and carts
    """
    out = ""
    start = f"{len(pos)}\n\n"
    out += start
    for n, r in enumerate(carts):
        x, y, z = r
        line = "{}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(el_dc[int(pos[n])], x, y, z)
        out += line
    return out


def write_cartesians_to_xyz(
    pos: np.array, carts: np.array, fn="out.xyz", charge_multiplicity=None
):
    """
    creates xyz file from pos and carts
    """
    el_dc = create_el_num_to_symbol()
    out = ""
    with open(fn, "w") as f:
        cm = ""
        if charge_multiplicity is not None:
            cm = f"{charge_multiplicity[0]} {charge_multiplicity[1]}\n"
        start = f"{len(pos)}\n{cm}\n"
        out += start
        f.write(start)
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


def read_xyz_to_pos_carts(
    xyz_path="mol.xyz",
) -> (np.array, np.array):
    """
    read_xyz_to_pos_carts reads xyz file and returns pos and carts
    """
    el_dc = create_pt_dict()
    with open(xyz_path, "r") as f:
        d = f.readlines()[2:]

    pos, carts = [], []
    for l in d:
        l = l.split()
        if l[0].isnumeric():
            el = int(l[0])
        else:
            el = el_dc[l[0]]
        x, y, z = float(l[1]), float(l[2]), float(l[3])
        pos.append(el)
        carts.append([x, y, z])
    return np.array(pos), np.array(carts)


def read_geom_to_pos_carts_nmers(
    xyz_path="mol.xyz",
    start=4,
) -> [np.ndarray, np.ndarray]:
    """
    read_xyz_to_pos_carts reads xyz file and returns pos and carts for AP-Net prediction
    """
    el_dc = create_pt_dict()
    with open(xyz_path, "r") as f:
        d = f.readlines()[start:]

    pos, carts = [[]], [[]]
    pc_ind = 0
    for l in d:
        if "--" in l:
            pc_ind += 1
            pos.append([])
            carts.append([])
            continue

        l = l.split()
        if l[0].isnumeric():
            el = int(l[0])
        else:
            try:
                el = el_dc[l[0]]
            except (KeyError, IndexError):
                el = l[0][0] + l[0][1:].lower()
                el = el_dc[el]
                print("Warning: element not in periodic table, using first letter")
                return [], []
        x, y, z = float(l[1]), float(l[2]), float(l[3])
        pos[pc_ind].append(el)
        carts[pc_ind].append([x, y, z])
    for i in range(len(pos)):
        pos[i] = np.array(pos[i])
        carts[i] = np.array(carts[i])
    if len(pos) == 1:
        pos = pos[0]
        carts = carts[0]
    return pos, carts


def convert_geom_str_to_dimer_splits(
    geom, units_angstroms=True
) -> [np.array, np.array, np.array, np.array]:
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


def mol_to_pos_carts_ma_mb(mol, units_angstroms=True):
    cD = mol.geometry
    if units_angstroms:
        cD = cD * qcel.constants.conversion_factor("bohr", "angstrom")
    pD = mol.atomic_numbers
    geom = np.hstack((pD.reshape(-1, 1), cD))
    # tools.print_cartesians_pos_carts(pD, cD)
    ma = list(mol.fragments[0])
    mb = list(mol.fragments[1])
    charges = np.array(
        [
            [int(mol.molecular_charge), int(mol.molecular_multiplicity)],
            [int(mol.fragment_charges[0]), (mol.fragment_multiplicities[0])],
            [int(mol.fragment_charges[1]), (mol.fragment_multiplicities[1])],
        ]
    )
    qcel.models.Molecule.from_data
    return geom, pD, cD, ma, mb, charges


def mol_qcdb_to_pos_carts_ma_mb(mol, units_angstroms=True):
    from psi4.driver import qcdb

    p4_input = mol.format_molecule_for_psi4()
    p4_input = "\n".join(p4_input.split("\n")[1:-2])
    geom = mol.format_molecule_for_numpy()
    cD = geom[:, 1:]
    pD = geom[:, 0]
    start = False
    skip_twice = False
    fragment_inds = []
    fragment_ind = 0
    total_index = 0
    lines = p4_input.split("\n")
    for n, l in enumerate(lines):
        if not start and "--" in l:
            start = True
            skip_twice = True
            fragment_inds.append([])
            continue
        if "--" in l:
            skip_twice = True
            fragment_inds.append([])
            fragment_ind += 1
            continue
        if skip_twice:
            skip_twice = False
            continue
        if "".join(l.split()) == "":
            break
        if start:
            if " X " in l:
                continue
            else:
                fragment_inds[fragment_ind].append(total_index)
                total_index += 1
    ma, mb = fragment_inds[0], fragment_inds[1]

    total_charge = 0
    for i in mol.fragment_charges:
        total_charge += i
    charges = np.array(
        [
            [
                int(total_charge),
                int(mol.PYmultiplicity),
            ],
            [int(mol.fragment_charges[0]), (mol.fragment_multiplicities[0])],
            [int(mol.fragment_charges[1]), (mol.fragment_multiplicities[1])],
        ]
    )
    qcel.models.Molecule.from_data
    return p4_input, geom, pD, cD, ma, mb, charges


def remove_extra_wb(line: str):
    """
    Removes extra whitespace in a string for better splitting.
    """
    line = (
        line.replace("    ", " ")
        .replace("   ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("\n ", "\n")
    )
    return line


