# Model Card: Adaptive Bayesian Black-Box Optimisation

> **Project:** Bayesian Black-Box Optimisation Capstone  
> **Objective:** Maximise eight unknown objective functions under a restricted evaluation budget  
> **Programme:** Professional Certificate in Machine Learning and Artificial Intelligence, Imperial College London
> **Datasheet framework:** Adapted from *Model Cards for Model Reporting* (Mitchell et al., 2019)

---

## 1. Executive Summary

This model card documents the optimisation system used to select sequential query points for eight unknown black-box functions. It is not a single static predictive model. The system combines:

- **For Global Bayesian search:** a separate Gaussian Process (GP) surrogate for each function;
- **For acquisition function:** Upper Confidence Bound (UCB), Expected Improvement (EI), and Probability of Improvement (PI);
- **For candidate selection:** meshed grid, Sobol, and trust-region candidate generation;
- **Targetted human intervention:** human review and occasional manual probes.

At each round, the GP is refitted using all observations available for the relevant function. Candidate inputs are scored using an acquisition function, reviewed against the search history, and one point is submitted to the hidden evaluator.

The workflow evolved from broad exploration to local refinement. Early rounds mainly used RBF-kernel GPs and UCB over Cartesian grids. Later rounds introduced Matérn kernels, scrambled Sobol candidates, incumbent-centred trust regions, lower exploration pressure, and local PI for narrow or boundary-sensitive regions.

The final dataset and its provenance, composition, collection process, biases, and permitted uses are documented separately in [`DATASHEET.md`](DATASHEET.md). This card focuses only on the optimisation methodology, performance, model risks, and reproducibility.

---

## 2. Model Scope

| Field | Description |
|---|---|
| **Model family** | Sequential model-based optimisation |
| **Primary surrogate** | `sklearn.gaussian_process.GaussianProcessRegressor` |
| **Objective** | Maximise each unknown function over `[0,1]^d` |
| **Dimensionality** | 2 to 8 continuous inputs |
| **Acquisition functions** | UCB, EI, PI |
| **Evaluator Input** | One recommended query vector per function and round |
| **Evaluator Output** | One float scalar |
| **Training approach** | Refit independently on each function’s cumulative observations |
| **Human involvement** | Candidate review and occasional manual override |
| **Deployment status** | Educational capstone; not validated for production or high-stakes use |

The documented “model” is the complete decision process:
→ diagnostics
→ GP fit
→ candidate generation
→ acquisition scoring
→ human review
→ submitted query
→ new observation

No information is shared across functions. Each function has its own surrogate, acquisition settings, and search history.

---

## 3. Intended use

The system is appropriate for low-data, continuous, bounded, single-objective black-box optimisation where evaluations are expensive and a human can inspect recommendations.

It requires redesign for categorical inputs, hard constraints, multi-objective optimisation, high-dimensional problems, strongly stochastic evaluators, batch experimentation, or safety-critical decisions.


**Model inputs and outputs**

For a function at a given round, the model receives its cumulative input matrix, response vector, unit-cube bounds, current incumbent, and function-specific search settings. It returns one candidate query and, where required for review, the GP posterior mean, posterior standard deviation, acquisition score, and alternative EI/UCB proposals.

Inputs are not pooled across functions. The different response scales and dimensionalities make a shared unadjusted model inappropriate. The dataset itself—including observation counts, ranges, collection order, and validation—is documented in the datasheet rather than repeated here.

---

## 4. Model Architecture

### 4.1 Gaussian Process surrogate

The notebooks use two main GP configurations.

#### Early global-search configuration

```python
RBF(length_scale=0.2) + WhiteKernel(noise_level=1e-6)
```

Typical settings include:

```python
alpha=1e-10
normalize_y=True
```

This provides a stable baseline for sparse observations but assumes a smooth, stationary surface. It can over-smooth narrow peaks and perform poorly when nearby points produce sharply different responses.

#### Later local-search configuration

```python
ConstantKernel * Matern(nu=2.5) + WhiteKernel
```

typically with multiple optimiser restarts, output normalisation, and fixed random seeds.

The Matérn kernel was introduced when the RBF/grid workflow became too rigid, particularly for higher-dimensional or locally volatile functions. F1 used a shorter-scale Matérn-3/2 model for a very narrow near-zero signal.

### 2.2 Predictive uncertainty

For each candidate `x`, the GP estimates:

- `mu(x)`: predicted objective value;
- `sigma(x)`: posterior predictive uncertainty.

These estimates guide the acquisition policy. The reported uncertainty is operational rather than formally calibrated: the project does not include repeated evaluator measurements, held-out coverage tests, or a dedicated noise model.

---

## 5. Acquisition and Candidate Selection

### 5.1 Upper Confidence Bound

```text
UCB(x) = mu(x) + kappa × sigma(x)
```

UCB was used primarily during exploration. Larger `kappa` values favour uncertain regions; smaller values concentrate on predicted high-value areas.

In later local searches, exploration pressure was reduced as the budget was consumed, including schedules of the form:

```text
kappa_t = 2.5 / sqrt(t)
```

### 5.2 Expected Improvement

EI scores both the probability and magnitude of improvement over the best observed value. It was used after promising regions had been identified and the priority shifted towards exploitation.

Typical improvement margins `xi` ranged from approximately `0.01` in broader searches to `0.001` or lower in late local refinement.

### 5.3 Probability of Improvement

PI was used selectively for late-stage local exploitation on F1 and F2. Because PI ignores the size of a potential improvement, it was restricted to narrow local regions and was not used as the main global-search policy.

### 5.4 Adaptive acquisition choice

Several notebooks calculated both EI and UCB proposals. The choice between them was based on:

- the current incumbent;
- the remaining predictive upside;
- whether unexplored regions still appeared valuable;
- whether recent queries had improved the result;
- the risk of repeatedly selecting near-identical points.

This is best described as **adaptive acquisition comparison with human review**, rather than a completely autonomous policy.


### 5.5 Acquisition-selection logic

| Search condition | Preferred response |
|---|---|
| Coverage remains poor or posterior upside is broad | UCB with stronger exploration |
| A credible high-value region has emerged | EI |
| The remaining task is very local confirmation | PI within a restricted region |
| EI and UCB propose materially different points | Review both against recent outcomes and coverage |
| The model repeats a previous or near-duplicate query | Apply the distance filter or redesign the candidate set |
| A boundary or ridge hypothesis is unresolved | Submit a documented diagnostic probe |

The policy was deliberately not identical across all functions. Uniform settings would have ignored clear differences between flat, volatile, boundary-seeking, and high-dimensional surfaces.

---

## 6. Candidate Generation

| Method | Role | Main limitation |
|---|---|---|
| **Cartesian grids** | Deterministic global search and easy visualisation in low dimensions | Exponential growth and poor resolution in higher dimensions |
| **Geometric max-distance search** | Model-free coverage diagnostic, used especially for F1 | Ignores predicted objective values |
| **Scrambled Sobol sequences** | Space-filling global candidates for F6–F8 | Finite samples may still miss narrow peaks |
| **Trust regions** | Local candidates around the best observed point | Can lock the search into the wrong basin |
| **Duplicate-distance filters** | Avoid repeated or near-repeated queries | Overly large thresholds can block useful confirmation |
| **Manual diagnostic probes** | Test boundaries, ridges, and local hypotheses | Introduces human judgement and reduces full automation |

The later workflow combined global and local Sobol candidates, then filtered candidates by distance from existing observations. Trust-region radii and distance thresholds were reduced as the search became more exploitative.

Manual probes were used transparently where the model alone was insufficient. Examples include:

- micro-probes around F1’s narrow positive region;
- testing whether F2’s lower-boundary ridge extended into the interior;
- confirming F5’s upper-corner behaviour through one-coordinate perturbations.

---

## 7. Decision Process

For each function and round:

1. Append the latest query and evaluator response.
2. Inspect the cumulative history and current best point.
3. Refit the function-specific GP.
4. Generate global, local, or mixed candidates.
5. Calculate posterior mean and uncertainty.
6. Score candidates with UCB, EI, or PI.
7. Remove invalid and near-duplicate candidates.
8. Compare model proposals with recent search behaviour.
9. Submit one model-generated or explicitly documented manual query.
10. Repeat after receiving the next response.

The process moved through three broad phases:

| Phase | Main behaviour |
|---|---|
| **Exploration** | Coverage diagnostics, UCB, broad grids |
| **Adaptive search** | EI/UCB comparison, Sobol candidates, function-specific settings |
| **Local refinement** | Matérn GPs, trust regions, lower `kappa`/`xi`, PI, diagnostic probes |

---

## 8. Function-Specific Strategy

| Function | Main challenge | Final strategy emphasis | Key finding |
|---|---|---|---|
| **F1** | Almost-flat surface with an extremely narrow positive signal | Short-scale Matérn GP, local PI, micro-probes | Broad exploration added little; only a very small numerical improvement was found |
| **F2** | Multiple promising regions and a lower-boundary ridge | Local EI/PI and boundary-to-interior tests | Best observed region lay slightly inside the lower boundary |
| **F3** | Negative, shallow landscape | UCB followed by EI around a productive interior basin | Improved substantially towards zero, but later exploration regressed |
| **F4** | Strong local volatility and sharp response changes | Increasingly local GP refinement | Crossed from a negative initial incumbent to a positive best-observed result |
| **F5** | Strong upper-corner trend | EI and explicit boundary validation | `[1,1,1,1]` remained the best observed point after nearby perturbations |
| **F6** | Sparse 5D search with negative outputs | Matérn GP, Sobol candidates, trust region | Later global-local candidate design materially improved the incumbent |
| **F7** | 6D landscape with a better basin discovered mid-search | Sobol exploration followed by local refinement | Strong sequential improvement, although gains were not monotonic |
| **F8** | 8D search with limited headroom from a strong starting point | Sobol plus trust-region refinement | Improved further, then entered a diminishing-return regime |

Detailed query histories, plots, and round-by-round reasoning remain in the function notebooks rather than being repeated here.

---

## 9. Performance

The primary evaluation measure is the **best objective observed so far** for each function. Raw values should not be compared across functions because the objective scales differ.

Final observation counts, response ranges, provenance, and coverage are maintained in [`DATASHEET.md`](DATASHEET.md).

| Function | Initial best | Final best observed | Absolute improvement |
|---|---:|---:|---:|
| F1 | `7.710875e-16` | `8.190854e-16` | `4.79979e-17` |
| F2 | `0.611205` | `0.632404` | `0.021199` |
| F3 | `-0.034835` | `-0.00626387` | `0.02857113` |
| F4 | `-4.025542` | `0.585186` | `4.610727` |
| F5 | `1088.859618` | `8662.4825` | `7573.622882` |
| F6 | `-0.714265` | `-0.103406` | `0.610859` |
| F7 | `1.364968` | `2.854626` | `1.489658` |
| F8 | `9.598482` | `9.99309` | `0.394608` |

### Interpretation

- The strategy improved the best observed value on all eight functions.
- F4–F7 show the clearest practical gains.
- F5 provided strong empirical evidence of an upper-corner optimum within the tested resolution.
- F8 improved from an already strong starting point and then showed diminishing returns.
- F1 remained effectively flat at ordinary numerical scales, so percentage improvement would be misleading.
- These results are **best observed values**, not proof of global optimality.

No held-out test set exists because the hidden evaluator is available only through submitted queries. Performance therefore measures optimisation progress rather than predictive generalisation.


### What the performance table does and does not show

The table measures realised optimisation progress. It does not measure the GP’s predictive accuracy independently of the acquisition policy, because every additional observation was selected adaptively. It also cannot separate the contribution of the surrogate, acquisition function, candidate set, and human review.

A stronger future evaluation would replay fixed baselines on repeatable benchmark functions and compare best-so-far curves, simple regret, query efficiency, and uncertainty calibration. Those tests were not available for the hidden capstone evaluator.

---

## 10. Assumptions and Limitations

### Core assumptions

- Inputs are continuous and correctly bounded within `[0,1]^d`.
- Nearby points contain useful information about one another.
- A stationary GP kernel is an adequate local approximation.
- The evaluator is sufficiently stable for one observation per query to be useful.
- Candidate-set maximisation is an acceptable approximation to continuous acquisition optimisation.
- The current incumbent is informative enough to justify late-stage local refinement.

### Main limitations

1. **Small samples in moderate dimensions.** GP length scales and uncertainty can be unstable, especially for F6–F8.
2. **Kernel misspecification.** RBF and Matérn kernels can smooth over spikes, cliffs, or non-stationary behaviour.
3. **Uncalibrated uncertainty.** Posterior standard deviations were used for decisions but were not independently validated.
4. **Boundary effects.** UCB may prefer boundaries because uncertainty is high, while some functions genuinely perform best there.
5. **Candidate approximation.** Grids and finite Sobol sets do not guarantee that the acquisition maximum is found.
6. **Trust-region lock-in.** Local refinement can miss a better distant basin.
7. **Manual hyperparameters.** `kappa`, `xi`, radii, and distance thresholds were chosen heuristically rather than benchmark-tuned.
8. **Human-selection bias.** Manual interpretation improves flexibility but may reinforce an incorrect narrative about the surface.
9. **Partial procedural reproducibility.** Seeded model-generated proposals are reproducible; human overrides require an explicit decision log.

---

## 11. Design Trade-offs

| Design decision | Benefit | Risk |
|---|---|---|
| GP surrogate | Effective in small-data settings and provides uncertainty | Sensitive to kernel assumptions |
| UCB | Preserves exploration | Can waste evaluations on uncertain boundaries |
| EI | Targets expected gain | Can over-concentrate near the incumbent |
| PI | Useful for terminal local search | Can converge prematurely |
| Sobol candidates | More scalable than full grids | Still an approximate acquisition search |
| Trust regions | Efficient use of late evaluations | May exclude better unexplored regions |
| Human review | Uses diagnostics the acquisition score may miss | Adds subjectivity and limits automation |

---


## 12. Responsible Interpretation

Recommendations should be presented as uncertain experimental choices, not authoritative predictions. The user should be able to identify:

- the current incumbent and proposed query;
- the acquisition policy and its parameters;
- whether the point is global, local, or boundary-constrained;
- its distance from previous observations;
- whether it was model-generated or manually selected.

The terms **global optimum**, **solved**, and **converged** should not be used without external ground truth. The appropriate claim for this project is **best observed value in the recorded query history**.

The workflow has not been validated for operational, financial, medical, or safety-critical deployment. Real-world adaptation would require domain constraints, calibrated uncertainty, explicit noise handling, monitoring, and independent validation.

---

## 13. Reproducibility

A reproducible release should preserve:

- cumulative observations in query order;
- notebook or code version for each round;
- kernel definition and fitted settings;
- acquisition type and parameters;
- candidate method, sample size, bounds, and seed;
- trust-region centre and radius;
- duplicate-distance threshold;
- model-generated alternatives;
- final submitted query;
- justification for any manual override;
- Python and package versions.

The notebooks contain much of this information, but it is distributed across eight files. A future release should centralise it in a structured per-round decision log.

Dataset provenance, observation counts, quality controls, and release requirements are documented in [`DATASHEET.md`](DATASHEET.md).

---

## 14. Recommended Improvements

The highest-priority improvements are:

1. centralise all model and acquisition settings in configuration files;
2. generate one machine-readable decision record per function and round;
3. compare isotropic and ARD kernels using leave-one-out diagnostics;
4. test uncertainty calibration against realised query outcomes;
5. retain a small global candidate allocation during local refinement;
6. benchmark against random search, max-distance sampling, and a fixed BO baseline;
7. generate final performance tables automatically from the canonical dataset.

---

## Conclusion

The project developed from a basic RBF/UCB baseline into a more adaptive human-in-the-loop Bayesian optimisation workflow. Its strongest methodological improvements were the move to Matérn surrogates, Sobol candidate generation, incumbent-centred trust regions, declining exploration pressure, and transparent diagnostic probes.

The strategy improved the best observed value on every function, but the results remain constrained by small samples, kernel assumptions, approximate acquisition optimisation, uncalibrated uncertainty, and human-selected hyperparameters.

Within those limits, the system provides an interpretable and technically credible implementation of sequential black-box optimisation under a restricted evaluation budget.

