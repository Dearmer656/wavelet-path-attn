import math

import torch

from analyze_k1_k4_wavelet_amp import causal_centered_rms_amp, component_weighted_upper_bound_amp


def test_causal_centered_rms_known_values_and_future_exclusion():
    x = torch.tensor(
        [
            [
                [1.0, 999.0, 999.0],
                [1.0, 3.0, 999.0],
                [2.0, 4.0, 6.0],
            ]
        ]
    )

    amp = causal_centered_rms_amp(x, q0=0)

    assert torch.allclose(amp[0, 0], torch.tensor(0.0))
    assert torch.allclose(amp[0, 1], torch.tensor(1.0))
    assert torch.allclose(amp[0, 2], torch.tensor(math.sqrt(8.0 / 3.0)))

    x_changed_future = x.clone()
    x_changed_future[0, 1, 2] = -99999.0
    amp_changed = causal_centered_rms_amp(x_changed_future, q0=0)
    assert torch.allclose(amp[0, 1], amp_changed[0, 1])


def test_causal_centered_rms_translation_invariance_per_row():
    x = torch.tensor([[[1.0, 3.0, 5.0], [2.0, 6.0, 10.0]]])
    shifted = x.clone()
    shifted[:, 0, :] += 17.0
    shifted[:, 1, :] -= 12.0

    assert torch.allclose(causal_centered_rms_amp(x, q0=0), causal_centered_rms_amp(shifted, q0=0))


def test_convexity_inequality_for_weighted_components():
    basis_a = torch.tensor([[[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]])
    basis_b = -basis_a
    basis = torch.stack([basis_a, basis_b], dim=-1)
    pi = torch.full((1, 2, 2), 0.5)

    mixture = (basis * pi.unsqueeze(-2)).sum(dim=-1)
    amp_mixture = causal_centered_rms_amp(mixture, q0=0)
    upper = component_weighted_upper_bound_amp(basis, pi, q0=0)

    assert torch.all(amp_mixture <= upper + 1e-6)
    assert torch.allclose(amp_mixture, torch.zeros_like(amp_mixture))
    assert torch.any(upper > 0)


def test_component_weighted_upper_bound_hand_computed():
    basis = torch.tensor(
        [
            [
                [[1.0, 10.0], [3.0, 14.0], [999.0, 999.0]],
                [[2.0, 5.0], [6.0, 9.0], [10.0, 13.0]],
            ]
        ]
    )
    pi = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])

    upper = component_weighted_upper_bound_amp(basis, pi, q0=0)

    # q=0 has one valid key, so both component amplitudes are zero.
    assert torch.allclose(upper[0, 0], torch.tensor(0.0))
    # q=1 valid keys:
    # scale0 [2, 6] has centered RMS 2; scale1 [5, 9] has centered RMS 2.
    assert torch.allclose(upper[0, 1], torch.tensor(2.0))
