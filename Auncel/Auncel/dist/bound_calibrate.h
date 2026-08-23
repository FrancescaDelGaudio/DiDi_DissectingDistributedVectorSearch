#pragma once

namespace auncel_dist {

// §4.5 error-bound calibration (NSDI paper).
// global_epsilon: allowed global error rate ε in [0, 1] (e.g. 0.1 for 10% error).
// k: top-k workload size.
// w: number of workers (shards).
// gamma: confidence threshold γ (default 0.999).
// Returns calibrated local error bound εc (≤ ε).
double calibrate_local_epsilon(
        double global_epsilon,
        int k,
        int w,
        double gamma = 0.999);

// P(Hk = i) per Formula 12.
double probability_hk_equals(int i, int k, int w, double epsilon_c);

// P(Hk ≥ k·(1−ε)) per Formula 13.
double probability_meets_global_bound(
        double global_epsilon,
        int k,
        int w,
        double epsilon_c);

}  // namespace auncel_dist
