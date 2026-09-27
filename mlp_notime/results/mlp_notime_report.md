# MLP-without-time control (review fix #3c)

Question: the GRU `learned_decoder` beats the fair `spike_rate` readout by about 14 pts, but it also adds ~1.77M trainable parameters. Does the gain come from using **time**, or just from the **extra parameters**?

Control: same frozen SpikingResformer-Ti backbone, same hooked LIF (`layers.2.6.down.0`), same per-timestep spatially pooled spike features the GRU receives, but **averaged over T=4** (order and per-step values discarded), then an MLP with the same parameter budget, then the same CBL + head. Existing checkpoints and reports are unchanged.

## Decoder parameter match

| Decoder | Architecture | Params |
|:---|:---|:---:|
| MLP-no-time | Linear(1536->576) -> GELU -> Linear(576->1536) | 1,771,584 |
| learned_decoder | GRU(1536->256) + Linear(256->1536) | 1,772,544 |

Relative difference: 0.054% (required <= 2%).

## Recipe

Train(fit) 5,095 / held-out 899 / test 5,794; 50 epochs, AdamW lr 0.001, wd 0.0001, batch 32, cosine LR, grad clip 5.0, concept dropout 0.25, train_cbm.py augmentation, backbone frozen in eval mode, seed 0. Identical to `train_spike_rate_fair.py` and the learned_decoder run.

Selected epoch: **33** (held-out ClassAcc 55.06%, ConceptAUC 0.8749). Training time 152.4 min.

## Test results (n=5,794, one shared backbone pass)

| Model | Uses time? | Extra params | Species acc | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|:---:|:---:|
| spike_rate (fair, epoch 48) | no | 0 | 45.44% | 0.8652 | 0.1193 |
| **MLP-no-time (this run)** | no | 1,771,584 | **56.51%** | **0.8782** | **0.1069** |
| learned_decoder (GRU, epoch 35) | yes | 1,772,544 | 59.48% | 0.9194 | 0.0788 |

Concept ECE = mean per-concept ECE of the raw (uncalibrated) sigmoid scores, 15 equal-width bins (`calibration_ece.expected_calibration_error`).

## Paired comparisons (10,000 bootstrap resamples, seed 20260826; exact McNemar)

| Comparison | Gap (pts) | 95% CI | McNemar p | first right / second wrong | first wrong / second right | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| GRU - MLP-no-time | +2.97 | [+1.90, +4.04] | 8.52e-08 | 598 | 426 | first significantly better |
| MLP-no-time - spike_rate | +11.06 | [+9.80, +12.34] | 1.48e-67 | 1025 | 384 | first significantly better |
| GRU - spike_rate | +14.03 | [+12.72, +15.36] | 1.81e-95 | 1210 | 397 | first significantly better |

## Conclusion

**Mixed: the GRU is significantly better than MLP-no-time (+2.97 pts, CI [+1.90, +4.04]), but the extra parameters alone already recover 79% of the +14.03-pt GRU gain over spike_rate. Most of the gain is from PARAMETERS; time adds a smaller, real part.**

Of the +14.03-pt GRU gain over spike_rate, +11.06 pts (79%) are recovered by time-blind extra parameters and +2.97 pts (21%) remain attributable to per-timestep information.

## Caveats

- Averaging over T removes *all* per-timestep information (order and the spread of values across steps), not only order. A GRU advantage therefore means per-step information helps; it does not by itself show that *order* matters. The timing-shuffle test (3a) probes order directly.
- One training seed per arm (seed 0). Bootstrap CIs cover test-set sampling, not training-run variance.
- The two decoders have matched parameter counts but different inductive biases (recurrence vs. a single hidden layer), so this is a capacity control, not an exact architectural twin.

## Training history (held-out validation)

| Epoch | TrainLoss | ValLoss | ConceptAUC | ClassAcc(%) | LR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 5.5822 | 5.3219 | 0.8150 | 1.78 | 1.00e-03 |
| 2 | 4.8416 | 4.6548 | 0.8478 | 6.45 | 9.99e-04 |
| 3 | 4.2413 | 4.2419 | 0.8626 | 13.13 | 9.96e-04 |
| 4 | 3.7911 | 3.9230 | 0.8656 | 16.24 | 9.91e-04 |
| 5 | 3.4257 | 3.6402 | 0.8652 | 25.14 | 9.84e-04 |
| 6 | 3.1052 | 3.4523 | 0.8667 | 28.59 | 9.76e-04 |
| 7 | 2.8486 | 3.2547 | 0.8684 | 31.03 | 9.65e-04 |
| 8 | 2.6256 | 3.1138 | 0.8711 | 33.59 | 9.52e-04 |
| 9 | 2.4301 | 2.9539 | 0.8714 | 38.71 | 9.38e-04 |
| 10 | 2.2566 | 2.8324 | 0.8714 | 40.93 | 9.22e-04 |
| 11 | 2.1137 | 2.7436 | 0.8755 | 41.49 | 9.05e-04 |
| 12 | 1.9831 | 2.6669 | 0.8725 | 43.27 | 8.85e-04 |
| 13 | 1.8660 | 2.6113 | 0.8696 | 43.94 | 8.65e-04 |
| 14 | 1.7656 | 2.5270 | 0.8713 | 46.16 | 8.42e-04 |
| 15 | 1.6816 | 2.4881 | 0.8713 | 46.61 | 8.19e-04 |
| 16 | 1.5924 | 2.4776 | 0.8735 | 47.05 | 7.94e-04 |
| 17 | 1.5145 | 2.4168 | 0.8731 | 48.28 | 7.68e-04 |
| 18 | 1.4598 | 2.3804 | 0.8711 | 49.17 | 7.41e-04 |
| 19 | 1.4104 | 2.3325 | 0.8738 | 50.28 | 7.13e-04 |
| 20 | 1.3521 | 2.3504 | 0.8739 | 50.17 | 6.84e-04 |
| 21 | 1.3085 | 2.2614 | 0.8767 | 49.17 | 6.55e-04 |
| 22 | 1.2738 | 2.2859 | 0.8737 | 51.39 | 6.25e-04 |
| 23 | 1.2155 | 2.2761 | 0.8755 | 50.50 | 5.94e-04 |
| 24 | 1.1804 | 2.1966 | 0.8744 | 51.72 | 5.63e-04 |
| 25 | 1.1512 | 2.1872 | 0.8737 | 52.39 | 5.32e-04 |
| 26 | 1.1240 | 2.2201 | 0.8742 | 51.39 | 5.01e-04 |
| 27 | 1.0891 | 2.1683 | 0.8745 | 52.28 | 4.69e-04 |
| 28 | 1.0680 | 2.1744 | 0.8728 | 52.73 | 4.38e-04 |
| 29 | 1.0446 | 2.2044 | 0.8750 | 50.61 | 4.07e-04 |
| 30 | 1.0152 | 2.1985 | 0.8732 | 52.17 | 3.76e-04 |
| 31 | 1.0062 | 2.1254 | 0.8753 | 54.51 | 3.46e-04 |
| 32 | 0.9805 | 2.1075 | 0.8750 | 53.62 | 3.17e-04 |
| 33 | 0.9634 | 2.1104 | 0.8749 | 55.06 | 2.88e-04 |
| 34 | 0.9504 | 2.1351 | 0.8747 | 53.73 | 2.60e-04 |
| 35 | 0.9364 | 2.0832 | 0.8744 | 54.39 | 2.33e-04 |
| 36 | 0.9138 | 2.1078 | 0.8748 | 53.50 | 2.07e-04 |
| 37 | 0.9087 | 2.0923 | 0.8740 | 53.62 | 1.82e-04 |
| 38 | 0.9009 | 2.1098 | 0.8746 | 53.50 | 1.59e-04 |
| 39 | 0.8855 | 2.0785 | 0.8749 | 53.73 | 1.36e-04 |
| 40 | 0.8779 | 2.0746 | 0.8746 | 54.51 | 1.16e-04 |
| 41 | 0.8694 | 2.0673 | 0.8745 | 54.95 | 9.64e-05 |
| 42 | 0.8667 | 2.0764 | 0.8755 | 54.95 | 7.88e-05 |
| 43 | 0.8498 | 2.0607 | 0.8752 | 54.28 | 6.28e-05 |
| 44 | 0.8476 | 2.0637 | 0.8749 | 54.51 | 4.85e-05 |
| 45 | 0.8531 | 2.0584 | 0.8755 | 54.17 | 3.61e-05 |
| 46 | 0.8459 | 2.0567 | 0.8750 | 54.51 | 2.54e-05 |
| 47 | 0.8444 | 2.0567 | 0.8751 | 54.62 | 1.67e-05 |
| 48 | 0.8344 | 2.0586 | 0.8752 | 54.39 | 9.85e-06 |
| 49 | 0.8445 | 2.0571 | 0.8752 | 54.17 | 4.94e-06 |
| 50 | 0.8422 | 2.0566 | 0.8752 | 54.28 | 1.99e-06 |
