import mdtraj as md
import pandas as pd
import numpy as np
from collections import namedtuple

Res = namedtuple(
    "Res",
    "name index",
)


def min_max_normalize(data, drange):
    data = np.array(data)
    return (data - drange[0]) / (drange[1] - drange[0])


def mean_normalize(data, drange):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))


def z_score(data):
    return (data - np.mean(data)) / np.std(data)


def transform_data(data, drange, a=-1, b=1):
    return (b - a) * ((data - drange[0]) / (drange[1] - drange[0])) + a


class NMR_PDB_Parser:
    """
    Class to handle indexing of UCBShift data with pdbs read by mdtraj.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe consisting of shifts calculated by UCBShifts
    pdb : str
        name of a pdb file
    hrange: tuple
        hrange to normalize to -1, 1
    nrange: tuple
        nrange to normalzie to -1, 1
    corange: tuple
        corange to normalize to -1, 1


    Attributes
    ----------
    df: pandas.DataFrame
        NMR data
    traj : md.Trajectory
        pdb loaded into mdtraj
    residue_to_hsqc : dict[Res] -> [float, float, float]
        Mapping from residue name and index to hsqc data.
        If the hsqc is not calculated, the list will be [None, None, None]
    resindex_to_res: dict[Res] -> int
        Mapping from residue index to res
    resindex_to_resnode: dict[int] -> int
        Mapping from residue index to resnode
    resnode_to_res: dict[int] -> Res
        Mapping from residue index to resnode
    hsqc: numpy.array
        array of hsqc values
    contacts_seq : List[Res, Res]
        List of contacts between Res objects
    contacts : List[int, int]
        List of contacts between residue indexes
    Notes
    -----
    When UCBShift fails to calculate a shift for a residue, it is simply omitted from the DataFrame. This could cause indexing errors
    when combined with structural data where that residue is not omitted.
    This class's purpose is to make sure the indexing between the NMR data and the pdb data is consistent.

    """

    def __init__(
        self, df, pdb, hrange=(7.3, 9.3), nrange=(103.3, 129.7), corange=(169.8, 181.8)
    ):
        self.df = df
        self.df["RESIND"] = self.df["RESNUM"] - 1
        self.hrange = hrange
        self.nrange = nrange
        self.corange = corange

        # Load trajectory
        self.traj = md.load(pdb)

        # Create dictionaries mapping between structure and NMR
        self._create_residue_dicts()

        # Create hsqc for node data input into model
        self._create_hsqc()

        # Create contact edges
        self._create_contacts_seq()

    def _create_residue_dicts(self):
        self.residue_to_hsqc = {}
        self.resindex_to_res = {}
        self.resindex_to_resnode = {}
        self.resnode_to_res = {}
        resnode_idx = 0

        for residue in self.traj.topology.residues:
            name = residue.name
            index = residue.index
            res = Res(name=name, index=index)
            self.resindex_to_res[res.index] = res

            df = self.df.loc[
                (self.df["RESIND"] == res.index) & (self.df["RESNAME"] == res.name)
            ]

            if df.empty:
                self.residue_to_hsqc[res] = None
                self.resindex_to_resnode[res.index] = None
                print(f"{res.name}{res.index} has no calculated hsqc values")
            else:
                h = df["H_UCBShift"].to_numpy()[0]
                n = df["N_UCBShift"].to_numpy()[0]
                co = df["C_UCBShift"].to_numpy()[0]
                self.residue_to_hsqc[res] = [h, n, co]
                self.resindex_to_resnode[res.index] = resnode_idx
                self.resnode_to_res[resnode_idx] = res
                resnode_idx = resnode_idx + 1

    def _create_hsqc(self):
        h_shifts = []
        n_shifts = []
        co_shifts = []
        for res in self.residue_to_hsqc.keys():
            if self.residue_to_hsqc[res] != None:
                h = self.residue_to_hsqc[res][0]
                n = self.residue_to_hsqc[res][1]
                co = self.residue_to_hsqc[res][2]
                h_shifts.append(h)
                n_shifts.append(n)
                co_shifts.append(co)
        h_shifts = transform_data(np.array(h_shifts), self.hrange)
        n_shifts = transform_data(np.array(n_shifts), self.nrange)
        co_shifts = transform_data(np.array(co_shifts), self.corange)
        self.hsqc = np.vstack((n_shifts, h_shifts, co_shifts)).T

    def _create_contacts_seq(self):
        pairs = gen_contacts(self.traj)
        self.contacts_seq = []
        self.contacts = []
        for pair in pairs:
            i = self.resindex_to_resnode[pair[0]]
            if i == None:
                print(
                    f"No residue node exists for {self.resindex_to_res[pair[0]]}, skipping contact "
                )
                continue
            j = self.resindex_to_resnode[pair[1]]
            if j == None:
                print(
                    f"No residue node exists for {self.resindex_to_res[pair[1]]}, skipping contact "
                )
                continue
            self.contacts.append([i, j])

        for contact in self.contacts:
            i = self.resnode_to_res[contact[0]]
            j = self.resnode_to_res[contact[1]]
            self.contacts_seq.append([f"{i.name}{i.index}", f"{j.name}{j.index}"])


def process_whiten_ucbshift_data(df, hrange, nrange, corange):
    h_shifts = df["H_UCBShift"].to_numpy()
    n_shifts = df["N_UCBShift"].to_numpy()
    co_shifts = df["C_UCBShift"].to_numpy()

    return (
        transform_data(h_shifts, hrange),
        transform_data(n_shifts, nrange),
        transform_data(co_shifts, corange),
    )


def gen_contacts(pdb):
    # get all of the non-proline residues
    H_indices = []
    H_residues = []
    for residue in pdb.topology.residues:
        N = list(residue.atoms_by_name("N"))
        H = list(residue.atoms_by_name("H"))
        if not N or not H:
            print(
                f"Skipped residue {residue.name}{residue.index}."
            )  # Need to do something else for prolines eventually.
        else:
            N = N[0]
            H = H[0]
            H_indices.append(H.index)
            H_residues.append(residue.index)
    src = []
    dst = []
    for i, resi in zip(H_indices, H_residues):
        for j, resj in zip(H_indices, H_residues):
            if j <= i:
                continue
            xyz1 = pdb.xyz[0, i, :]
            xyz2 = pdb.xyz[0, j, :]
            dist = np.linalg.norm(xyz1 - xyz2)
            if dist < 0.5:
                src.append(resi)
                dst.append(resj)
    pairs = []
    for resi, resj in zip(src, dst):
        pairs.append((resi, resj))
    return pairs
    # return torch.tensor([src, dst], dtype=torch.long)


def gen_pre(pdb_filename, probe_residue_indices, probe_residue_names, cutoff=1.75):
    pdb = md.load(pdb_filename)

    # Make sure specified probes and residues exist in input pdb
    probe_atom_indices = []
    for probe, name in zip(probe_residue_indices, probe_residue_names):
        N = pdb.topology.select(f"resid {probe} and resname {name} and name N")
        assert len(N) == 1
        probe_atom_indices.append(N)

    assert len(probe_atom_indices) == len(probe_residue_indices)

    # Measure distances between N atoms in pdb.
    # This is not the most ideal solution since it does not account for probe.
    N_indices = []
    for residue in pdb.topology.residues:
        N = list(residue.atoms_by_name("N"))
        H = list(residue.atoms_by_name("H"))
        if not N or not H:
            print(
                f"Skipped residue {residue}."
            )  # Need to do something else for prolines eventually.
        else:
            N = N[0]
            H = H[0]
            N_indices.append(N.index)

    # Loop over specified probes and N atom indices
    # Measure distances between N atom indices and probes
    # src/dst account for edges between each probe residue and every other residue
    # dists is a list of distance lists. So with 3 probes, dists has a length of 3 where
    # each element list has a length of protein residues (minus proline residues)
    src = []
    dst = []
    dists = []
    for probe_res_index, probe_atom_index in zip(
        probe_residue_indices, probe_atom_indices
    ):
        dists_by_probe = []
        for residue_index, atom_index in enumerate(N_indices):
            xyz1 = pdb.xyz[0, probe_atom_index, :]
            xyz2 = pdb.xyz[0, atom_index, :]
            dist = np.linalg.norm(xyz1 - xyz2)
            dists_by_probe.append(dist)
            src.append(probe_res_index)
            dst.append(residue_index)

        dists.append(dists_by_probe)

    # Combine src/dst into pairs to be inputted into data.py
    pairs = []
    for i, j in zip(src, dst):
        pairs.append((i, j))

    # PRE measurements are fuzzy rulers
    # If the distance is greater than cutoff, set the distance to cutoff since no attenuation would be observed.
    for idx, dists_by_probe in enumerate(dists):
        for i in range(len(dists_by_probe)):
            if dists_by_probe[i] > cutoff:
                dists_by_probe[i] = cutoff
        # Transform distances into [-1, 1]
        dists[idx] = transform_data(np.array(dists[idx]), [0, cutoff])

    return pairs, dists
