import pickle
import numpy as np
from periodictable import elements


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


def print_cartesians_pos_carts(pos: np.array, carts: np.array):
    """
    prints a 2-D numpy array in a nicer format
    """
    print()
    for n, r in enumerate(carts):
        x, y, z = r
        line = "{}\t{:.10f}\t{:.10f}\t{:.10f}".format(int(pos[n]), x, y, z)
        print(line)
    print()


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
