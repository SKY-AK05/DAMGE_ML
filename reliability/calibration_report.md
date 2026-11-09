> [!WARNING] **Calibration Provenance & Selection Bias Notice**
> The 652 calibration images (`calib_652.json`) were partitioned from the original 1,304 validation
> pool on which Teacher A and Teacher B checkpoints were selected. These reliability metrics must
> be treated as exploratory and potentially selection-biased until independent re-calibration is conducted.

# Teacher Reliability Calibration Report

Held-out validation evidence for offline multi-teacher knowledge distillation.

## Summary of Evaluated Teachers

| Teacher | Global Precision | Global Recall | Global Mean TP IoU | r_global | Total TPs |
|---|---:|---:|---:|---:|---:|
| teacher_a_m2f | 0.6591 | 0.9248 | 0.8809 | **0.8146** | 2570 |
| teacher_b_m2f | 0.5860 | 0.8908 | 0.8735 | **0.7781** | 2471 |

---

## Per-Class Shrunk Reliability Scores (r_hat)

| Class ID | Class Name | teacher_a_m2f r_hat (CI) | teacher_b_m2f r_hat (CI) | Flag |
|---:|---| ---: | ---: |---|
| 0 | wheel | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 1 | wheel_cap | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 2 | front_left_light | 0.8047 [0.78, 0.82] | 0.7905 [0.78, 0.81] | OK |
| 3 | front_right_light | 0.8559 [0.85, 0.89] | 0.8413 [0.84, 0.88] | OK |
| 4 | door_glass | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 5 | left_mirror | 0.8163 [0.80, 0.83] | 0.8052 [0.80, 0.83] | OK |
| 6 | right_mirror | 0.8251 [0.81, 0.85] | 0.7825 [0.76, 0.81] | OK |
| 7 | front_bumper | 0.8445 [0.84, 0.85] | 0.7893 [0.78, 0.80] | OK |
| 8 | grille | 0.5235 [0.48, 0.51] | 0.5871 [0.53, 0.57] | OK |
| 9 | front_left_door | 0.8634 [0.87, 0.90] | 0.8605 [0.88, 0.91] | OK |
| 10 | front_right_door | 0.8356 [0.83, 0.87] | 0.8472 [0.86, 0.91] | OK |
| 11 | hood | 0.9080 [0.91, 0.92] | 0.8942 [0.89, 0.91] | OK |
| 12 | pillar | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 13 | back_left_door | 0.8757 [0.89, 0.92] | 0.8616 [0.89, 0.92] | OK |
| 14 | back_right_door | 0.8357 [0.82, 0.88] | 0.8344 [0.84, 0.90] | OK |
| 15 | front_glass | 0.9178 [0.92, 0.93] | 0.9114 [0.91, 0.92] | OK |
| 16 | front_left_fender | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 17 | front_right_fender | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 18 | back_left_light | 0.8120 [0.78, 0.84] | 0.7009 [0.63, 0.69] | OK |
| 19 | back_right_light | 0.8055 [0.77, 0.82] | 0.7542 [0.71, 0.76] | OK |
| 20 | rear_quarter_panel | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 21 | roof | 0.6770 [0.65, 0.68] | 0.6443 [0.62, 0.64] | OK |
| 22 | side_skirt_rocker_panel | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 23 | fog_lamp | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 24 | back_bumper | 0.7398 [0.72, 0.75] | 0.7247 [0.71, 0.74] | OK |
| 25 | indicator | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 26 | number_plate | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 27 | back_glass | 0.8263 [0.81, 0.84] | 0.7947 [0.78, 0.81] | OK |
| 28 | tailgate | 0.8146 [0.00, 1.00] | 0.7781 [0.00, 1.00] | Low-N,Low-N |
| 29 | fuel_lid | 0.6909 [0.58, 0.66] | 0.6959 [0.61, 0.69] | OK |
| 30 | antenna | 0.5402 [0.32, 0.38] | 0.4964 [0.27, 0.33] | OK |
| 31 | trunk | 0.7533 [0.72, 0.76] | 0.7191 [0.68, 0.73] | OK |

> [!NOTE]
> **Asymmetry Rule:** On training examples, GT matching can veto teacher guidance (setting omega=0), but can never inflate trust beyond the validation-derived r_hat ceiling shown above.
