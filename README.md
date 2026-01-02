# Causal Machine Learning with Meta-Learners

## Project Overview

- Simulates observational data with heterogeneous potential outcomes
- Focuses on estimating the **Average Treatment Effect (ATE)**
- Applies causal machine learning **meta‑learning approaches**

## Methods
- Observational data simulation with heterogeneous treatment effects
- Causal inference via meta‑learning approaches
  - **S‑learner** : A single outcome model including treatment as a covariate
  - **T‑learner** : Separate outcome models for treated and control groups
- Base learners are **parametric regression models**
- Outcome modeling using:
  - Parametric regression models
  - Flexible machine learning models (Random Forests)
    
## Evaluation

- Models are compared using:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
- Estimated effects are evaluated against the **true simulated ATE**

## Objective

- Estimate treatment effects from observational data
- Demonstrate how causal inference methods can be implemented using meta‑learners
- Compare different learning strategies for ATE estimation

## Meta‑Learner Comparison (Parametric Models)

- Under nonlinear and heterogeneous treatment effects:
  - **S‑learner** provides more stable treatment effect estimation
  - **T‑learner** exhibits higher variance and extreme estimation errors

Under nonlinear and heterogeneous treatment effects, the **S‑learner** demonstrates more stable treatment effect estimation, while the **T‑learner** exhibits higher variance and more extreme errors.

**Performance Metrics**
- S‑learner RMSE: **1.023**
- T‑learner RMSE: **1.274**
- S‑learner MAE: **0.742**
- T‑learner MAE: **0.812**

<img width="425" height="281" alt="Parametric meta-learner comparison" src="https://github.com/user-attachments/assets/5c0cf717-1a6c-4682-9289-88909a475cac" />

## Results: Meta‑Learner Comparison (Machine Learning Models)

When Random Forests are used as base learners, the **S‑learner** again outperforms the **T‑learner**, achieving lower error metrics and more stable estimation under nonlinear and heterogeneous treatment effects.

**Performance Metrics**
- S‑learner (RF) RMSE: **1.294**
- T‑learner (RF) RMSE: **1.548**
- S‑learner (RF) MAE: **0.898**
- T‑learner (RF) MAE: **0.994**

<img width="425" height="281" alt="Random Forest meta-learner comparison" src="https://github.com/user-attachments/assets/785ae315-cf94-4456-91c4-52ef86f4eaa0" />

## Overall Evaluation

Across all evaluation metrics, including **RMSE** and **MAE**, the **parametric S‑learner** consistently outperforms both the parametric T‑learner and the Random Forest–based meta‑learners.  
Although machine‑learning models offer greater functional flexibility, their increased variance leads to inferior finite‑sample performance in this setting. These results highlight that more flexible base learners do not necessarily yield lower estimation error when the **bias–variance trade‑off** is unfavorable.



## Files
- `simulate_data.py`: Simulates high-dimensional causal data with heterogeneous treatment effects.
- `meta_learners.py`: Implements S-learner and T-learner approaches.
- `evaluate_parametric.py`: Evaluates meta-learners by comparing estimated and true treatment effects.
- `evaluate_ml.py`: Evaluates meta-learners by comparing estimated and true treatment effects.

