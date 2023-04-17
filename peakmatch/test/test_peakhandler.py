import torch
from ..peakhandler import (
    PeakHandler,
    ResidueData,
    PeakNoiseAndMatchParams,
    generate_sample,
    _compute_noe_matches,
)
import numpy as np
import unittest
from unittest.mock import patch


class TestGenerateSample(unittest.TestCase):
    def setUp(self):
        self.residues = {
            2: ResidueData(0.0, 1.0, 2.0),
            4: ResidueData(3.0, 4.0, 5.0),
        }
        self.contacts = [(2, 4)]
        self.params = PeakNoiseAndMatchParams()

    def test_should_work_with_defaults(self):
        # This is a stupid test... it just makes sure
        # there isn't an exception
        generate_sample(self.residues, self.contacts, self.params)
        self.assertTrue(True)

    def test_should_work_with_hsqc_noise(self):
        # This is a stupid test... it just makes sure
        # there isn't an exception
        generate_sample(
            self.residues,
            self.contacts,
            self.params,
            min_hsqc_completeness=0.8,
            max_hsqc_noise=0.2,
        )
        self.assertTrue(True)

    def test_should_work_with_noe_noise(self):
        # This is a stupid test... it just makes sure
        # there isn't an exception
        generate_sample(
            self.residues,
            self.contacts,
            self.params,
            min_noe_completeness=0.8,
            max_noe_noise=0.2,
        )
        self.assertTrue(True)


class TestEmptyResidues(unittest.TestCase):
    def test_should_raise_with_empty_residues(self):
        residues = {}
        contacts = [(1, 2)]
        params = PeakNoiseAndMatchParams()
        with self.assertRaises(ValueError):
            PeakHandler(residues, contacts, params)


class TestEmptyContacts(unittest.TestCase):
    def setUp(self):
        self.residues = {1: ResidueData(0.0, 1.0, 2.0)}
        self.contacts = []
        params = PeakNoiseAndMatchParams()
        self.peak_handler = PeakHandler(self.residues, self.contacts, params)

    def test_pred_noe_should_be_empty_with_empty_contacts(self):
        expected = torch.tensor([[], []], dtype=torch.long)
        actual = self.peak_handler.pred_noe
        torch.testing.assert_close(actual, expected)

    def test_fake_noe_should_be_empty_with_empty_contacts(self):
        expected = torch.tensor([[], []], dtype=torch.long)
        actual = self.peak_handler.fake_noe
        torch.testing.assert_close(actual, expected)


class TestPeakHandlerWithoutMissingOrExtraPeaks(unittest.TestCase):
    def setUp(self):
        self.residues = {
            2: ResidueData(0.0, 1.0, 2.0),
            4: ResidueData(3.0, 4.0, 5.0),
        }
        self.contacts = [(2, 4)]
        self.params = PeakNoiseAndMatchParams(noise_h=0.0, noise_n=0.0, noise_co=0.0)

    def test_residue_to_pred_hsqc_mapping_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = {
            2: 0,
            4: 1,
        }
        actual = peak_handler.residue_to_pred_hsqc_mapping
        self.assertEqual(actual, expected)

    def test_residue_to_fake_hsqc_mapping_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = {
            2: 0,
            4: 1,
        }
        actual = peak_handler.residue_to_fake_hsqc_mapping
        self.assertEqual(actual, expected)

    def test_pred_hsqc_to_residue_mapping_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = {
            0: 2,
            1: 4,
        }
        actual = peak_handler.pred_hsqc_to_residue_mapping
        self.assertEqual(actual, expected)

    def test_fake_hsqc_to_residue_mapping_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = {
            0: 2,
            1: 4,
        }
        actual = peak_handler.fake_hsqc_to_residue_mapping
        self.assertEqual(actual, expected)

    def test_pred_hsqc_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [-3.0, -3.0, -3.0],  # dummy residue
            ]
        )
        actual = peak_handler.pred_hsqc
        torch.testing.assert_close(actual, expected)

    def test_pred_noe_should_be_correct(self):
        # Note: 5 is not an included peak, so (4, 5) and (5, 4)
        #       should note be included in the final set of contacts
        contacts = [(2, 4), (4, 5), (5, 4)]
        peak_handler = PeakHandler(self.residues, contacts, self.params)
        expected = torch.tensor([[0, 1], [1, 0]])
        actual = peak_handler.pred_noe
        torch.testing.assert_close(actual, expected)

    def test_fake_hsqc_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ]
        )
        actual = peak_handler.fake_hsqc
        torch.testing.assert_close(actual, expected)

    def test_correspondence_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor([0, 1])
        actual = peak_handler.correspondence
        torch.testing.assert_close(actual, expected)

    def test_fake_noe_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        # + 3 due to 2 pred_hsqc peaks and 1 dummy residue
        expected = torch.tensor([[0, 1], [1, 0]]) + 3
        actual = peak_handler.fake_noe
        torch.testing.assert_close(actual, expected)

    def test_n_pred_hsqc_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 2  # +1 due to dummy residue
        actual = peak_handler.n_pred_hsqc
        self.assertEqual(actual, expected)

    def test_n_pred_hsqc_nodes_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 3  # +1 due to dummy residue
        actual = peak_handler.n_pred_hsqc_nodes
        self.assertEqual(actual, expected)

    def test_n_fake_hsqc_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 2
        actual = peak_handler.n_fake_hsqc
        self.assertEqual(actual, expected)

    def test_n_fake_hsqc_nodes_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 2
        actual = peak_handler.n_fake_hsqc_nodes
        self.assertEqual(actual, expected)

    def test_dummy_hsqc_index_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 2
        actual = peak_handler.dummy_residue_index
        self.assertEqual(actual, expected)

    def test_fake_hsqc_offset_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 3
        actual = peak_handler.fake_hsqc_offset
        self.assertEqual(actual, expected)

    def test_virtual_node_index_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 2 + 2 + 1
        actual = peak_handler.virtual_node_index
        self.assertEqual(actual, expected)

    def test_virtual_edges_should_be_correct(self):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor(
            [[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]],
            dtype=torch.long,
        )
        actual = peak_handler.virtual_edges
        torch.testing.assert_close(actual, expected)


class TestNOEMatch(unittest.TestCase):
    def test_should_match(self):
        peaks = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ]
        )
        h1 = 0.0
        n1 = 1.0
        h2 = 3.0
        actual = _compute_noe_matches(peaks, h1, n1, h2, 0.1, 0.1, 0.1)
        expected = torch.tensor([[0], [1]])
        torch.testing.assert_close(actual, expected)

    def test_should_match_with_first_peak_duplicate(self):
        peaks = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [6.0, 7.0, 8.0],
            ]
        )
        h1 = 0.0
        n1 = 1.0
        h2 = 6.0
        actual = _compute_noe_matches(peaks, h1, n1, h2, 0.1, 0.1, 0.1)
        expected = torch.tensor([[0, 1], [2, 2]])
        torch.testing.assert_close(actual, expected)

    def test_should_match_with_second_peak_duplicate(self):
        peaks = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0],
            ]
        )
        h1 = 0.0
        n1 = 1.0
        h2 = 3.0
        actual = _compute_noe_matches(peaks, h1, n1, h2, 0.1, 0.1, 0.1)
        expected = torch.tensor([[0, 0], [1, 2]])
        torch.testing.assert_close(actual, expected)


class TestPeakHandlerWithExtraPeaks(unittest.TestCase):
    def setUp(self):
        self.residues = {
            2: ResidueData(0.0, 1.0, 2.0),
            4: ResidueData(3.0, 4.0, 5.0),
        }
        self.contacts = [(2, 4)]
        self.params = PeakNoiseAndMatchParams(noise_h=0.0, noise_n=0.0, noise_co=0.0)
        self.peak_handler = PeakHandler(
            self.residues, self.contacts, self.params, hsqc_peaks_to_add=3
        )

    def test_fake_hsqc_should_have_correct_shape(self):
        expected = (5, 3)
        actual = self.peak_handler.fake_hsqc.shape
        self.assertEqual(actual, expected)

    def test_correspondence_should_be_correct(self):
        expected = torch.tensor([0, 1, 2, 2, 2])
        actual = self.peak_handler.correspondence
        torch.testing.assert_close(actual, expected)

    def test_fake_hsqc_nodes_should_be_correct(self):
        expected = 5
        actual = self.peak_handler.n_fake_hsqc_nodes
        self.assertEqual(actual, expected)

    def test_virtual_node_index_should_be_correct(self):
        expected = 2 + 5 + 1
        actual = self.peak_handler.virtual_node_index
        self.assertEqual(actual, expected)

    def test_virtual_edges_should_be_correct(self):
        expected = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 8, 8, 8, 8, 8, 8, 8]],
            dtype=torch.long,
        )
        actual = self.peak_handler.virtual_edges
        torch.testing.assert_close(actual, expected)

    def test_residue_to_fake_hsqc_mapping_should_be_correct(self):
        expected = {
            2: 0,
            4: 1,
        }
        actual = self.peak_handler.residue_to_fake_hsqc_mapping
        self.assertEqual(actual, expected)

    def test_residue_to_pred_hsqc_mapping_should_be_correct(self):
        expected = {
            2: 0,
            4: 1,
        }
        actual = self.peak_handler.residue_to_pred_hsqc_mapping
        self.assertEqual(actual, expected)

    def test_fake_hsqc_to_residue_mapping_should_be_correct(self):
        expected = {
            0: 2,
            1: 4,
            2: None,
            3: None,
            4: None,
        }
        actual = self.peak_handler.fake_hsqc_to_residue_mapping
        self.assertEqual(actual, expected)

    def test_pred_hsqc_to_residue_mapping_should_be_correct(self):
        expected = {
            0: 2,
            1: 4,
        }
        actual = self.peak_handler.pred_hsqc_to_residue_mapping
        self.assertEqual(actual, expected)


class TestPeakHandlerWithPeaksMissing(unittest.TestCase):
    def setUp(self):
        self.residues = {
            2: ResidueData(0.0, 1.0, 2.0),
            4: ResidueData(3.0, 4.0, 5.0),
            6: ResidueData(6.0, 7.0, 8.0),
        }
        self.contacts = [(2, 4), (2, 6)]
        self.params = PeakNoiseAndMatchParams(noise_h=0.0, noise_n=0.0, noise_co=0.0)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_residue_to_pred_hsqc_mapping_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(
            self.residues, self.contacts, self.params, hsqc_peaks_to_drop=1
        )
        expected = {
            2: 0,
            4: 1,
            6: 2,
        }
        actual = peak_handler.residue_to_pred_hsqc_mapping
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_residue_to_fake_hsqc_mapping_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(
            self.residues, self.contacts, self.params, hsqc_peaks_to_drop=1
        )
        expected = {
            2: 0,
            4: None,
            6: 1,
        }
        actual = peak_handler.residue_to_fake_hsqc_mapping
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_pred_hsqc_to_pred_mapping_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(
            self.residues, self.contacts, self.params, hsqc_peaks_to_drop=1
        )
        expected = {
            0: 2,
            1: 4,
            2: 6,
        }
        actual = peak_handler.pred_hsqc_to_residue_mapping
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_fake_hsqc_to_pred_mapping_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(
            self.residues, self.contacts, self.params, hsqc_peaks_to_drop=1
        )
        expected = {
            0: 2,
            1: 6,
        }
        actual = peak_handler.fake_hsqc_to_residue_mapping
        self.assertEqual(actual, expected)


    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_pred_hsqc_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [-3.0, -3.0, -3.0],  # dummy residue
            ]
        )
        actual = peak_handler.pred_hsqc
        torch.testing.assert_close(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_pred_noe_should_be_correct(self, mock_method):
        # Note: 5 is not an included peak, so (2, 5) and (5, 2)
        #       should note be included in the final set of contacts.
        #       Peak 4 is being dropped, so (2, 4) should not be included.
        contacts = [(2, 4), (2, 6), (2, 5), (5, 2)]
        peak_handler = PeakHandler(self.residues, contacts, self.params)
        expected = torch.tensor([[0, 1], [1, 0]])
        actual = peak_handler.pred_noe
        torch.testing.assert_close(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_fake_hsqc_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [6.0, 7.0, 8.0],
            ]
        )
        actual = peak_handler.fake_hsqc
        torch.testing.assert_close(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_correspondence_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor([0, 2])
        actual = peak_handler.correspondence
        torch.testing.assert_close(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_fake_noe_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        # + 4 due to 3 pred_hsqc peaks and 1 dummy residue
        expected = torch.tensor([[0, 1], [1, 0]]) + 4
        actual = peak_handler.fake_noe
        torch.testing.assert_close(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_n_pred_hsqc_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 3
        actual = peak_handler.n_pred_hsqc
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_n_pred_hsqc_nodes_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 3 + 1  # +1 due to dummy residue
        actual = peak_handler.n_pred_hsqc_nodes
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_n_fake_hsqc_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 2
        actual = peak_handler.n_fake_hsqc
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_n_fake_hsqc_nodes_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 2
        actual = peak_handler.n_fake_hsqc_nodes
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_dummy_hsqc_index_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 3
        actual = peak_handler.dummy_residue_index
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_fake_hsqc_offset_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 4
        actual = peak_handler.fake_hsqc_offset
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_virtual_node_index_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = 3 + 2 + 1
        actual = peak_handler.virtual_node_index
        self.assertEqual(actual, expected)

    @patch.object(PeakHandler, "_choose_peaks_to_drop", return_value=[4])
    def test_virtual_edges_should_be_correct(self, mock_method):
        peak_handler = PeakHandler(self.residues, self.contacts, self.params)
        expected = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [6, 6, 6, 6, 6, 6]],
            dtype=torch.long,
        )
        actual = peak_handler.virtual_edges
        torch.testing.assert_close(actual, expected)
