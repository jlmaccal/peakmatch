import torch
from collections import namedtuple
import random
from dataclasses import dataclass
from copy import copy

ResidueData = namedtuple("ResidueData", "shift_h shift_n shift_co")


@dataclass
class PeakNoiseAndMatchParams:
    # these are calculated from avg shift values for each residue and atom provided by bmrb
    hrange : tuple = (7.3, 9.3)
    nrange : tuple = (103.3, 129.7)
    corange : tuple = (169.8, 181.8)

    # From UCBShift RMSE for H, N, C is 0.45, 2.61, 1.14 ppm, respectively.
    # The values we give the dataset are those values divided by hrange, nrange, corange.
    noise_h: float = 0.45 / (hrange[1] - hrange[0])  # noise added in h dimension
    noise_n: float = 2.61 / (nrange[1] - nrange[0])  # noise added in n dimension
    noise_co: float = 1.14 / (corange[1] - corange[0]) # noise added in co dimension

    noise_peak_factor: float = 3.0  # scale up noise added for extra peaks
    threshold_h1: float = 0.05  # tolerance for matching in h1 dimension
    threshold_n1: float = 0.05  # tolerance for matching in n1 dimension
    threshold_h2: float = 0.05  # tolerance for matching in h2 dimension


def generate_sample(
    residues,
    contacts,
    noise_and_match_params,
    min_hsqc_completeness=1.0,
    max_hsqc_noise=0.0,
    min_noe_completeness=1.0,
    max_noe_noise=0.0,
):
    """
    Generate a sample dataset.

    Parameters
    ----------
    residues : dict[index] -> ResidueData
        Dictionary of residues in the system. Indexing is based on
        what is in NMR data / structure.
    contacts : list[(index, index)]
        List of close contacts expected to give NOE peaks. Indexing is
        the same as for residues.
    noise_and_match_params : PeakNoiseAndMatchParams
        Parameters for peak noise and matching
    min_hsqc_completeness : float
        Minimum fraction of HSQC peaks to keep
    max_hsqc_noise : float
        Maximum fraction of HSQC peaks to add
    min_noe_completeness : float
        Minimum fraction of NOE contacts to keep
    max_noe_noise : float
        Maximum fraction of NOE contacts to add

    Returns
    -------
    PeakHandler
        PeakHandler object with the generated data
    """
    assert 0.0 <= min_hsqc_completeness <= 1.0
    assert 0.0 <= max_hsqc_noise <= 1.0
    assert 0.0 <= min_noe_completeness <= 1.0
    assert 0.0 <= max_noe_noise <= 1.0

    hsqc_completeness = random.uniform(min_hsqc_completeness, 1.0)
    peaks_to_remove = int((1.0 - hsqc_completeness) * len(residues))
    residues_after_remove = copy(residues)
    for _ in range(peaks_to_remove):
        del residues_after_remove[random.choice(list(residues_after_remove.keys()))]

    hsqc_noise = random.uniform(0.0, max_hsqc_noise)
    peaks_to_add = int(hsqc_noise * len(residues))

    noe_completeness = random.uniform(min_noe_completeness, 1.0)
    noes_to_remove = int((1.0 - noe_completeness) * len(contacts))

    noe_noise = random.uniform(0.0, max_noe_noise)
    noes_to_add = int(noe_noise * len(contacts))

    return PeakHandler(
        residues_after_remove,
        contacts,
        noise_and_match_params,
        peaks_to_add,
        noes_to_remove,
        noes_to_add,
    )


class PeakHandler:
    """
    Class to handle peaks and contacts.

    Parameters
    ----------
    residues : dict[index] -> ResidueData
        Dictionary of residues in the system. Indexing is based on
        what is in NMR data / structure.
    contacts : list[(index, index)]
        List of close contacts expected to give NOE peaks. Indexing is
        the same as for residues.
    noise_and_match_params : PeakNoiseAndMatchParams
        Parameters for peak noise and matching
    n_noise_peaks : int
        Number of noise peaks to add to hsqc
    noes_to_drop : int
        Number of NOEs to drop
    noes_to_add : int
        Number of NOEs to add

    Attributes
    ----------
    residue_mapping : dict[index] -> int
        Mapping from residue index to index in hsqc and noe data
    pred_hsqc : torch.Tensor
        Predicted HSQC peaks, including dummy residue
    pred_noe : torch.Tensor
        Predicted NOE edges
    fake_hsqs : torch.Tensor
        Fake HSQC peaks
    fake_noe : torch.Tensor
        Fake NOE edges
    virtual_edges : torch.Tensor
        Virtual edges

    Notes
    -----
    All atributes are torch tensors, except for residue_mapping. Indexing
    is based on the node index, with the predicted hsqcs first, then the
    dummy residue, then the fake_hsqcs, then the virtual node.

    """

    def __init__(
        self,
        residues,
        contacts,
        noise_and_match_params,
        n_noise_peaks=0,
        noes_to_drop=0,
        noes_to_add=0,
    ):
        self.noise_peak_factor = noise_and_match_params.noise_peak_factor
        self._noise_h = noise_and_match_params.noise_h
        self._noise_n = noise_and_match_params.noise_n
        self._noise_co = noise_and_match_params.noise_co
        self._threshold_h1 = noise_and_match_params.threshold_h1
        self._threshold_n1 = noise_and_match_params.threshold_n1
        self._threshold_h2 = noise_and_match_params.threshold_h2
        self._n_noise_peaks = n_noise_peaks
        self.noes_to_drop = noes_to_drop
        self.noes_to_add = noes_to_add

        # Dict from residue index to pred_hsqc peak index
        self.residue_mapping = self._compute_residue_mapping(residues)
        # A tensor of predicted hsqcs, including dummy residue
        self.pred_hsqc = self._compute_pred_hsqc(residues)
        # A tensor that maps each fake_hsqc peak to the correpsonding pred_hsqc peak
        self.correspondence = self._compute_correspondence()
        # A tensor of predicted noe edges
        self.pred_noe = self._compute_pred_noe(contacts)
        # A tensor of fake hsqcs, including extra peaks
        self.fake_hsqc = self._compute_fake_hsqc()
        # A tensor of fake_noe edges
        self.fake_noe = self._compute_fake_noe()
        # A tensor of virtual edges
        self.virtual_edges = self._compute_virtual_edges()

    @property
    def n_pred_hsqc(self):
        return self.n_pred_hsqc_nodes - 1

    @property
    def n_pred_hsqc_nodes(self):
        return self.pred_hsqc.shape[0]

    @property
    def n_fake_hsqc(self):
        return self.n_fake_hsqc_nodes

    @property
    def n_fake_hsqc_nodes(self):
        return self.fake_hsqc.shape[0]

    @property
    def dummy_residue_index(self):
        return self.n_pred_hsqc_nodes - 1

    @property
    def fake_hsqc_offset(self):
        return self.n_pred_hsqc_nodes

    @property
    def virtual_node_index(self):
        return self.n_pred_hsqc_nodes + self.n_fake_hsqc_nodes
    
    @property
    def n_nodes(self):
        return self.n_pred_hsqc_nodes + self.n_fake_hsqc_nodes + 1

    def _compute_residue_mapping(self, residues):
        if not residues:
            raise ValueError("No residues specified")

        mapping = {}
        for i, key in enumerate(residues.keys()):
            mapping[key] = i
        return mapping

    def _compute_correspondence(self):
        base = torch.arange(len(self.residue_mapping))
        if self._n_noise_peaks:
            noise = torch.tensor([self.dummy_residue_index] * self._n_noise_peaks)
            return torch.cat((base, noise))
        else:
            return base

    def _compute_pred_hsqc(self, residues):
        pred_hsqc = []
        for key in residues.keys():
            residue = residues[key]
            pred_hsqc.append((residue.shift_h, residue.shift_n, residue.shift_co))
        # Append a final row for the dummy residue
        pred_hsqc.append((-3.0, -3.0, -3.0))
        return torch.tensor(pred_hsqc)

    def _compute_pred_noe(self, contacts):
        e1 = []
        e2 = []
        for i, j in contacts:
            if i not in self.residue_mapping or j not in self.residue_mapping:
                continue
            e1.append(self.residue_mapping[i])
            e2.append(self.residue_mapping[j])
            e1.append(self.residue_mapping[j])
            e2.append(self.residue_mapping[i])

        return torch.tensor([e1, e2], dtype=torch.long)

    def _compute_fake_hsqc(self):
        # We chop off the last entry, which is the dummy residue
        means = self.pred_hsqc[:-1, :]
        std = (
            torch.tensor([self._noise_h, self._noise_n, self._noise_co])
            .unsqueeze(0)
            .expand_as(means)
        )
        peaks = torch.normal(means, std)
        if self._n_noise_peaks == 0:
            return peaks
        else:
            noise_peaks = self._sample_noise_peaks()
            return torch.cat([peaks, noise_peaks], dim=0)

    def _sample_noise_peaks(self):
        noise_peaks = []
        for _ in range(self._n_noise_peaks):
            # choose a random peak as the source
            source_index = random.randrange(0, self.n_pred_hsqc)
            source = self.pred_hsqc[source_index, :]
            # sample a new peak from a normal distribution
            noise_peak = torch.normal(
                source,
                self.noise_peak_factor
                * torch.tensor([self._noise_h, self._noise_n, self._noise_co]),
            ).unsqueeze(0)
            noise_peaks.append(noise_peak)
        return torch.cat(noise_peaks, dim=0)

    def _compute_fake_noe(self):
        fake_noe_edges = []
        for i, j in self.pred_noe.t():
            shift_h1 = self.fake_hsqc[i, 0]
            shift_n1 = self.fake_hsqc[i, 1]
            shift_h2 = self.fake_hsqc[j, 0]
            matches = _compute_noe_matches(
                self.fake_hsqc,
                shift_h1,
                shift_n1,
                shift_h2,
                self._threshold_h1,
                self._threshold_n1,
                self._threshold_h2,
            )
            fake_noe_edges.append(matches)

        if not fake_noe_edges:
            edges_to_keep = torch.tensor([[], []], dtype=torch.long)
        else:
            fake_noe_edges = torch.cat(fake_noe_edges, dim=1) + self.fake_hsqc_offset
            n_noes = fake_noe_edges.shape[1]
            indices_to_keep = sorted(
                random.sample(list(range(n_noes)), n_noes - self.noes_to_drop)
            )

            edges_to_keep = fake_noe_edges[:, indices_to_keep]

        # Add the extra noes by choosing random pairs of peaks
        # and addition noise
        extra_noes = []
        for _ in range(self.noes_to_add):
            # choose a random peak as the source
            source_index1 = random.randrange(0, self.n_pred_hsqc)
            source_index2 = random.randrange(0, self.n_pred_hsqc)

            source1 = self.fake_hsqc[source_index1, :]
            source2 = self.fake_hsqc[source_index2, :]

            # sample new peaks from a normal distribution
            noise_peak1 = torch.normal(
                source1,
                self.noise_peak_factor
                * torch.tensor([self._noise_h, self._noise_n, self._noise_co]),
            ).unsqueeze(0)
            noise_peak2 = torch.normal(
                source2,
                self.noise_peak_factor
                * torch.tensor([self._noise_h, self._noise_n, self._noise_co]),
            ).unsqueeze(0)

            shift_h1 = noise_peak1[0]
            shift_n1 = noise_peak1[1]
            shift_h2 = noise_peak2[0]

            matches = _compute_noe_matches(
                self.fake_hsqc,
                shift_h1,
                shift_n1,
                shift_h2,
                self._threshold_h1,
                self._threshold_n1,
                self._threshold_h2,
            )
            extra_noes.append(matches)

        if not extra_noes:
            extra_noes = torch.tensor([[], []], dtype=torch.long)
        else:
            extra_noes = torch.cat(extra_noes, dim=1) + self.fake_hsqc_offset

        return torch.cat([edges_to_keep, extra_noes], dim=1)

    def _compute_virtual_edges(self):
        edges_m = []
        edges_n = []
        for i in range(self.virtual_node_index):
            edges_m.append(i)
            edges_n.append(self.virtual_node_index)
        return torch.tensor([edges_m, edges_n], dtype=torch.long)


def _compute_noe_matches(
    shifts, shift_h1, shift_n1, shift_h2, threshold_h1, threshold_n1, threshold_h2
):
    edges_m = []
    edges_n = []

    possible1 = torch.argwhere(
        torch.logical_and(
            (torch.abs(shifts[:, 0] - shift_h1) < threshold_h1),
            (torch.abs(shifts[:, 1] - shift_n1) < threshold_n1),
        )
    )
    possible2 = torch.argwhere(torch.abs(shifts[:, 0] - shift_h2) < threshold_h2)

    for m in possible1:
        for n in possible2:
            edges_m.append(m)
            edges_n.append(n)

    return torch.tensor([edges_m, edges_n], dtype=torch.long)
