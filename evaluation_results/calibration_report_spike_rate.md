# Calibration Report -- readout=spike_rate

**Caveat**: temperature was fit on a split carved out of the original TRAIN set, which the checkpoint's cbl/head were already trained on. This likely UNDERSTATES the true calibration benefit (see calibration_ece.py module docstring). Treat this as a directional result, not the paper-final number; the rigorous fix is a from-scratch retrain with a proper 4-way split.

## Per-concept temperature
mean=1.0662  min=0.3338  max=2.2012

## ECE on calib_eval (disjoint from calib_fit)
Uncalibrated: 0.1784
Calibrated: 0.1692
Delta: +0.0092

## ECE on main test split
Uncalibrated: 0.1742
Calibrated: 0.1652
Delta: +0.0090

Reliability diagram: C:\Users\palag\Research\SpikingResformer\evaluation_results\reliability_diagram_spike_rate.png
