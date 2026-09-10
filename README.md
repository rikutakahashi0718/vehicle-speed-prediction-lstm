# Vehicle Speed Prediction with LSTM

Short-term (5 s) vehicle speed forecasting from driving simulator data, built as the
input signal for a model predictive controller in a hybrid powertrain.

**3 km/h RMSE on unseen drivers, against an 8 km/h persistence baseline (62% reduction).**

## Setup

- **Task** — predict vehicle speed at *t* + 5 s from history up to *t*
- **Inputs** — vehicle speed, throttle, steering angle, brake
- **Data** — collected by the author on a driving simulator, 6 drivers
- **Model** — LSTM, hyperparameters tuned by Bayesian optimization

## Evaluation

Evaluated on held-out drivers rather than held-out windows, and on driving regimes absent
from the training distribution (constant-speed cruising, hard acceleration and braking).

Undergraduate thesis, Dept. of Electrical and Electronic Engineering, Tokyo Denki
University, Mar 2025 – Mar 2026.

*Full write-up in progress.*
