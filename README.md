# Causal Machine Learning with Meta-Learners

This project demonstrates causal effect estimation using machine learning
meta-learners in a simulated high-dimensional setting.

We simulate observational data with heterogeneous treatment effects and
estimate Conditional Average Treatment Effects (CATE) using popular
meta-learning approaches, including the S-learner and T-learner. Flexible
machine learning models (random forests) are used to model outcome functions.

The project is designed to illustrate how causal inference and modern machine
learning methods can be combined to estimate and compare treatment effects
beyond average effects.

## Files
- `simulate_data.py`: Simulates high-dimensional causal data with heterogeneous treatment effects.
- `meta_learners.py`: Implements S-learner and T-learner approaches.
- `evaluate.py`: Evaluates meta-learners by comparing estimated and true treatment effects.

## Methods
- Potential outcomes framework
- Heterogeneous treatment effects
- Meta-learning for causal inference
- Machine learning–based outcome modeling

This repository is intended as a concise, reproducible example of causal
machine learning methods for epidemiology, biostatistics, and applied ML
research.
