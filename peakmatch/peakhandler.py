import torch
from collections import namedtuple
import random
from dataclasses import dataclass
from copy import copy

ResidueData = namedtuple("ResidueData", "shift_h shift_n shift_co")


@dataclass
class PeakNoiseAndMatchParams:
    # these are calculated from avg shift values for each residue and atom provided by bmrb
    hrange: tuple = (7.3, 9.3)
    nrange: tuple = (103.3, 129.7)
    corange: tuple = (169.8, 181.8)

    # # From UCBShift RMSE for H, N, C is 0.45, 2.61, 1.14 ppm, respectively.
    # # The values we give the dataset are those values divided by hrange, nrange, corange.
    noise_h: float = 0.45 / (hrange[1] - hrange[0])  # noise added in h dimension
    noise_n: float = 2.61 / (nrange[1] - nrange[0])  # noise added in n dimension
    noise_co: float = 1.14 / (corange[1] - corange[0])  # noise added in co dimension

    noise_peak_factor: float = 3.0  # scale up noise added for extra peaks
    threshold_h1: float = 0.05 / (
        hrange[1] - hrange[0]
    )  # tolerance for matching in h1 dimension
    threshold_n1: float = 0.50 / (
        nrange[1] - nrange[0]
    )  # tolerance for matching in n1 dimension
    threshold_h2: float = 0.50 / (
        corange[1] - corange[0]
    )  # tolerance for matching in h2 dimension


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
    n_hsqc_peaks_to_drop = int((1.0 - hsqc_completeness) * len(residues))
    hsqc_noise = random.uniform(0.0, max_hsqc_noise)
    n_hsqc_peaks_to_add = int(hsqc_noise * len(residues))

    noe_completeness = random.uniform(min_noe_completeness, 1.0)
    n_noes_to_remove = int((1.0 - noe_completeness) * len(contacts))

    noe_noise = random.uniform(0.0, max_noe_noise)
    n_noes_to_add = int(noe_noise * len(contacts))

    return PeakHandler(
        residues,
        contacts,
        noise_and_match_params,
        n_hsqc_peaks_to_drop,
        n_hsqc_peaks_to_add,
        n_noes_to_remove,
        n_noes_to_add,
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
    hsqcs_to_drop : int
        Number of peaks to drop from hsqc
    hsqcs_to_add : int
        Number of noise peaks to add to hsqc
    noes_to_drop : int
        Number of NOEs to drop
    noes_to_add : int
        Number of NOEs to add

    Attributes
    ----------
    residue_to_pred_hsqc_mapping : dict[index] -> int
        Mapping from residue index to index in hsqc and noe data
    residue_to_fake_hsqc_mapping : dict[index] -> int
        Mapping from residue index to index in fake hsqc data
    pred_hsqc_to_residue_mapping : dict[index] -> int
        Mapping from pred hsqc index to residue index
    fake_hsqc_to_residue_mapping : dict[index] -> int
        Mapping from fake hsqc index to residue index
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
    All atributes are torch tensors, except for residue_to_peak_mapping. Indexing
    is based on the node index, with the predicted hsqcs first, then the
    dummy residue, then the fake_hsqcs, then the virtual node.

    """

    def __init__(
        self,
        residues,
        contacts,
        noise_and_match_params,
        hsqc_peaks_to_drop=0,
        hsqc_peaks_to_add=0,
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
        self._hsqc_peaks_to_drop = hsqc_peaks_to_drop
        self._hsqc_peaks_to_add = hsqc_peaks_to_add
        self.noes_to_drop = noes_to_drop
        self.noes_to_add = noes_to_add

        # A tensor of predicted hsqcs, including dummy residue
        self.pred_hsqc = self._compute_pred_hsqc(residues)

        # Set up initial mappings
        self._setup_residue_mapping(residues)

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

    def _setup_residue_mapping(self, residues):
        if not residues:
            raise ValueError("No residues specified")

        peaks_to_drop = self._choose_peaks_to_drop()
        self.residue_to_pred_hsqc_mapping = {}
        self.residue_to_fake_hsqc_mapping = {}
        offset = 0
        for i, key in enumerate(residues.keys()):
            if key in peaks_to_drop:
                self.residue_to_pred_hsqc_mapping[key] = i
                self.residue_to_fake_hsqc_mapping[key] = None
                offset = offset - 1
            else:
                self.residue_to_pred_hsqc_mapping[key] = i
                self.residue_to_fake_hsqc_mapping[key] = i + offset

        self.pred_hsqc_to_residue_mapping = {}
        for key, value in self.residue_to_pred_hsqc_mapping.items():
            self.pred_hsqc_to_residue_mapping[value] = key

        self.fake_hsqc_to_residue_mapping = {}
        for key, value in self.residue_to_fake_hsqc_mapping.items():
            if value is not None:
                self.fake_hsqc_to_residue_mapping[value] = key

        offset = len(self.fake_hsqc_to_residue_mapping)
        for i in range(self._hsqc_peaks_to_add):
            self.fake_hsqc_to_residue_mapping[i + offset] = None

    def _compute_correspondence(self):
        correspondence = []
        for fake_ind in self.fake_hsqc_to_residue_mapping:
            residue = self.fake_hsqc_to_residue_mapping[fake_ind]
            if residue is None:
                pred_ind = self.dummy_residue_index
            else:
                pred_ind = self.residue_to_pred_hsqc_mapping[residue]
            correspondence.append(pred_ind)
        return torch.tensor(correspondence, dtype=torch.long)

    def _compute_pred_hsqc(self, residues):
        # Compute the predicted hsqc. This is just re-indexing things
        # to be sequential and adding dummy residues. There is nothing
        # stochastic in this function.
        pred_hsqc = []
        for key in residues.keys():
            residue = residues[key]
            pred_hsqc.append((residue.shift_h, residue.shift_n, residue.shift_co))
        # Append a final row for the dummy residue
        pred_hsqc.append((-3.0, -3.0, -3.0))
        return torch.tensor(pred_hsqc)

    def _compute_pred_noe(self, contacts):
        # Compute the expected noes between the predicted hsqc peaks
        # for the given contacts. There is nothing stochastic in this
        # function.
        e1 = []
        e2 = []
        for i, j in contacts:
            e1.append(self.residue_to_pred_hsqc_mapping[i])
            e2.append(self.residue_to_pred_hsqc_mapping[j])
            e1.append(self.residue_to_pred_hsqc_mapping[j])
            e2.append(self.residue_to_pred_hsqc_mapping[i])

        return torch.tensor([e1, e2], dtype=torch.long)

    def _compute_fake_hsqc(self):
        # Generate the fake hsqc. There are two sources of randomness
        # in this function. First, we add noise to the predicted
        # hsqc peaks. Second, we add extra peaks. The extra peaks
        # come from fake_hsqc_to_residue_mapping.
        fake_hsqcs = []
        # Loop over all of the peaks
        for peak in self.fake_hsqc_to_residue_mapping.keys():
            residue = self.fake_hsqc_to_residue_mapping[peak]

            # There are two cases. If this peak does not correspond
            # to a real residue, we sample a random noise peak.
            if residue is None:
                fake_hsqc = self._sample_noise_peak()
            # Otherwise, we add noise to the corresponding real peak.
            else:
                pred_peak = self.residue_to_pred_hsqc_mapping[residue]
                means = self.pred_hsqc[pred_peak, :].unsqueeze(0)
                std = torch.tensor(
                    [self._noise_h, self._noise_n, self._noise_co]
                ).unsqueeze(0)
                fake_hsqc = torch.normal(means, std)
            fake_hsqcs.append(fake_hsqc)

        return torch.cat(fake_hsqcs, dim=0)

    def _sample_noise_peak(self):
        source_index = random.randrange(0, self.n_pred_hsqc)
        source = self.pred_hsqc[source_index, :]
        # sample a new peak from a normal distribution
        noise_peak = torch.normal(
            source,
            self.noise_peak_factor
            * torch.tensor([self._noise_h, self._noise_n, self._noise_co]),
        ).unsqueeze(0)
        return noise_peak

    def _compute_fake_noe(self):
        # Generate the fake noes. There are two sources of randomness
        # in this function. First, we drop some of the predicted noes.
        # Second, we add extra noes.
        fake_noe_edges = []

        # First we find the chemical shifts for all of the expected noes.
        shifts = []
        for i, j in self.pred_noe.t():
            # Map from the predicted hsqc peak to the fake hsqc peak.
            i = i.item()
            i = self.residue_to_fake_hsqc_mapping[self.pred_hsqc_to_residue_mapping[i]]
            j = j.item()
            j = self.residue_to_fake_hsqc_mapping[self.pred_hsqc_to_residue_mapping[j]]

            # If either of the peaks are missing from the fake hsqc, we skip
            # this noe.
            if i is None or j is None:
                continue

            # Gather the chemical shifts for the two peaks.
            shift_h1 = self.fake_hsqc[i, 0]
            shift_n1 = self.fake_hsqc[i, 1]
            shift_h2 = self.fake_hsqc[j, 0]
            shifts.append((shift_h1, shift_n1, shift_h2))

        # Now we randomly select a subset of the shifts to keep.
        if shifts:
            shifts = torch.tensor(shifts)
            n_shifts = shifts.shape[0]
            indices_to_keep = sorted(
                random.sample(list(range(n_shifts)), n_shifts - self.noes_to_drop)
            )
            shifts_to_keep = shifts[indices_to_keep, :]

            # Now we compute the matches for all of the expected noes.
            for shift_h1, shift_n1, shift_h2 in shifts_to_keep:
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
        else:
            fake_noe_edges = []

        # Turn it into a tensor.
        if not fake_noe_edges:
            edges_to_keep = torch.tensor([[], []], dtype=torch.long)
        else:
            edges_to_keep = torch.cat(fake_noe_edges, dim=1) + self.fake_hsqc_offset

        # Add the extra noes by choosing random pairs of peaks
        # and addition noise
        extra_noes = []
        for _ in range(self.noes_to_add):
            # choose a random peak as the source
            source_index1 = random.randrange(0, self.n_pred_hsqc)
            source_index2 = random.randrange(0, self.n_pred_hsqc)

            source1 = self.pred_hsqc[source_index1, :]
            source2 = self.pred_hsqc[source_index2, :]

            # sample new peaks from a normal distribution
            noise_peak1 = torch.normal(
                source1,
                self.noise_peak_factor
                * torch.tensor([self._noise_h, self._noise_n, self._noise_co]),
            )
            noise_peak2 = torch.normal(
                source2,
                self.noise_peak_factor
                * torch.tensor([self._noise_h, self._noise_n, self._noise_co]),
            )

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
            edges_n.append(i)
            edges_m.append(self.virtual_node_index)
            edges_m.append(i)
            edges_n.append(self.virtual_node_index)
        return torch.tensor([edges_m, edges_n], dtype=torch.long)

    def _choose_peaks_to_drop(self):
        if self._hsqc_peaks_to_drop > 0:
            return random.sample(
                # peaks to remove should correspond to real predicted hsqc and not added
                # therefoer use self.n_pred_hsqc and not self.n_fake_hsqc, which includes noisified and spurious peaks
                range(0, self.n_pred_hsqc),
                k=self._hsqc_peaks_to_drop,
            )
        else:
            return []


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
