# Experimental Bayesian Black-Box Optimisation

**Researcher-in-the-loop sequential optimisation across eight hidden objective functions**

**Project:** Bayesian Black-Box Optimisation Capstone <br>
**Programme:** Machine Learning and Artificial Intelligence, Imperial College London <br>

---

*An applied Bayesian optimisation project for learning how to explore and maximise unknown black-box functions. Across 13 sequential rounds, predictive modelling, uncertainty-aware decision-making, and practical machine-learning heuristics were used to learn from each result and determine where to search next..*

---

## 1. At a glance

| Item | Project snapshot |
|---|---|
| Objective | Maximise eight unknown functions through sequential feedback |
| Search spaces | Two to eight dimensions, with inputs in \[0,1] |
| Data snapshot | 175 initial observations and 104 recorded queries |
| Method | Gaussian-process surrogates with UCB, EI and targeted PI |
| Candidate search | Cartesian grids and global/local Sobol sequences |
| Outcome | A higher best-observed value was found for all eight functions |
| Caveat | Unknown global optima prevent claims of global optimality |

---

## 2. Non-Technical Explanation

Imagine being asked to optimise an unfamiliar machine without access to its manual or internal mechanics. The machine has several adjustable settings, but each configuration is costly to evaluate and only one test can be conducted per week. With such a limited testing budget, relying on trial and error alone would be inefficient.

This is the optimisation problem explored in the project. Eight hidden black-box functions represented eight different machines, each accepting a combination of inputs and returning a performance score. Across 13 weekly rounds, one new configuration was submitted for each function, and the resulting observations were used to iteratively build and refine predictive models of their behaviour.

These models informed each subsequent decision by balancing two objectives: exploring uncertain regions that might contain better solutions and refining areas that had already produced promising results. By the end of the project, all 8 functions achieved scores above their initial baselines.

---

## 3. Research problem

For each hidden function:

```math
\mathbf{x}^{*}
=
\arg\max_{\mathbf{x}\in[0,1]^d} f(\mathbf{x}),
```

where:
- d ranges from 2 to 8 <br>
- f(x) can be evaluated only through the course query process.

The project combines exploratory analysis, Gaussian-process modelling, uncertainty-aware acquisitions, dimension-appropriate candidate generation and researcher judgement.

> **Research position:** this is a transparent researcher-in-the-loop study, not a fully automated optimiser or proof of global convergence.

---

## 4. Methodology

A separate Gaussian Process (GP) is fitted to each function. It produces a predicted value, \(\mu(\mathbf{x})\), and uncertainty, \(\sigma(\mathbf{x})\), which acquisition functions combine to select candidates.

- **Upper Confidence Bound:** balances predicted value and uncertainty.
- **Expected Improvement:** targets improvement over the incumbent.
- **Probability of Improvement:** supports selected local refinements.

Early searches use RBF-based global models. Later high-dimensional and local searches use Matérn kernels, output normalisation and multiple optimiser restarts.

| Search setting | Candidate method |
|---|---|
| Low-dimensional global search | Dense Cartesian grids |
| Higher-dimensional global search | Scrambled Sobol sequences |
| Local refinement | Sobol samples in clipped neighbourhoods |
| Ridge, edge or corner tests | Researcher-directed probes |

The highest acquisition score is not always accepted mechanically. Selection may also reflect distance from prior observations, local geometry, boundary behaviour, candidate resolution and the remaining budget.

The process moves from exploratory analysis and global modelling to acquisition comparison, improved candidate coverage, local refinement and explicit boundary tests as evidence accumulates.

---


## 5. Experimental data and Results

No external dataset is used. Initial observations are stored under:

```text
Initial Data/function_<n>/initial_inputs.npy
Initial Data/function_<n>/initial_outputs.npy
```

Sequential responses were appended inside the function-specific notebooks.

| Function | Dimension | Initial | New queries | Total | Initial best | Best observed | Absolute change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 10 | 13 | 23 | 7.71088E-16 | 8.190853E-16 | 4.799789E-17 |
| 2 | 2 | 10 | 13 | 23 | 0.611205 | 0.632404 | +0.021199 |
| 3 | 3 | 15 | 13 | 28 | -0.034835 | -0.006264 | +0.028571 |
| 4 | 4 | 30 | 13 | 43 | -4.025542 | 0.585186 | +4.610728 |
| 5 | 4 | 20 | 13 | 33 | 1,088.859618 | 8,662.4825 | +7,573.6229 |
| 6 | 5 | 20 | 13 | 33 | -0.714265 | -0.103406 | +0.61086 |
| 7 | 6 | 30 | 13 | 43 | 1.364968 | 2.854626 | +1.489658 |
| 8 | 8 | 40 | 13 | 53 | 9.598482 | 9.993086 | +0.394604 |
| **Total** | — | **175** | **104** | **279** |

Each function received 13 sequential queries, but differences in dimensionality, initial sample size and response scale prevent direct cross-function performance comparisons.

The datasets are small, repeated evaluations are unavailable, noise cannot be estimated directly, and new hidden-function responses cannot be reproduced locally. See `DATASHEET.md` for full provenance.

---

## 6. Function-level findings

- **Functions 1–2:** narrow local or boundary regions required targeted refinement.
- **Functions 3–4:** later exploitation improved materially on the initial incumbents.
- **Function 5:** the exact corner \([1,1,1,1]\) materially outperformed alternatives.
- **Functions 6–8:** Sobol candidates and local Matérn-GP refinement improved higher-dimensional search.
- **Function 7:** repeated refinement more than doubled the initial incumbent.

---

## 7. Research lessons

**Candidate design matters.** An acquisition function can only select among the points it is allowed to evaluate; Sobol candidates expanded effective coverage in higher dimensions.

**Exploration and exploitation are stages.** UCB, EI and PI were adapted to the maturity and geometry of each search.

**Boundaries must remain searchable.** The strongest Function 5 result occurred at the exact corner, while Function 2 benefited from boundary investigation.

**Negative queries remain informative.** Poor outcomes rejected regions and changed later decisions.

**Research judgement should be auditable.** Model proposals, manual probes and interpretation are distinguished rather than presented as one autonomous algorithm.

---

## 8. Reproducibility [TO REVIEW LATER]

```bash
git clone <repository-url>
cd <repository-directory>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

Run notebooks in numerical order from top to bottom. The repository can reproduce data loading, GP fitting, candidate generation, acquisitions, figures and recorded proposals, subject to package versions and random seeds.

It cannot evaluate new points against the hidden functions, recreate external responses, calculate true global regret or prove global optimality.

---

## 9. Repository structure

```text
.
├── README.md
├── DATASHEET.md
├── MODEL_CARD.md
├── requirements.txt
├── notebooks/
│   └── Function_1.ipynb ... Function_8.ipynb
├── reports/
│   └── Function_1.html ... Function_8.html
└── Initial Data/
    └── function_1/ ... function_8/
```

Version counters and export suffixes should be removed from public filenames.

---

## 10. Limitations and next steps

- Unknown objectives and optima prevent measurement of global regret.
- Only one sequential path exists per function.
- Noise cannot be estimated from repeated observations.
- Manual probes prevent attribution of gains solely to automation.
- Results may depend on modelling and candidate-set choices.
- No random-search benchmark or uncertainty-calibration study is recorded.

Useful extensions include reusable optimisation modules, benchmark policies, surrogate diagnostics, deterministic replay and structured query provenance.

---

## 11. Responsible interpretation

This repository shows how sparse observations, model uncertainty and researcher judgement can support an auditable sequential search.

It does not establish that global maxima were found, that GP uncertainty is calibrated or that the policy will generalise unchanged to unrelated objectives.

---

## 12. References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*.
2. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). *Efficient Global Optimization of Expensive Black-Box Functions.*
3. Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). *Gaussian Process Optimization in the Bandit Setting.*
4. Brochu, E., Cora, V. M., & de Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions.*
5. Sobol, I. M. (1967). *On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals.*

---

## 13. Attribution

This repository documents an individual capstone analysis completed for an Imperial College London machine-learning programme. The hidden functions and course materials remain subject to the programme's terms. Only materials the author is permitted to redistribute should be published.


