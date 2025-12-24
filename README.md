# Causal Machine Learning with Meta-Learners

This project demonstrates causal effect estimation using machine learning
meta-learners in a simulated high-dimensional setting.

We simulate observational data with heterogeneous treatment effects and
estimate Conditional Average Treatment Effects (CATE) using popular
meta-learning approaches, including the S-learner and T-learner. 
We implement S‑ and T‑learner frameworks using parametric regression models as base learners. Also, flexible
machine learning models (random forests) are used to model outcome functions.

The project is designed to illustrate how causal inference and modern machine
learning methods can be combined to estimate and compare treatment effects
beyond average effects.

## Files
- `simulate_data.py`: Simulates high-dimensional causal data with heterogeneous treatment effects.
- `meta_learners.py`: Implements S-learner and T-learner approaches.
- `evaluate_parametric.py`: Evaluates meta-learners by comparing estimated and true treatment effects.

## Methods
- Potential outcomes framework
- Heterogeneous treatment effects
- Meta-learning for causal inference
- Machine learning–based outcome modeling

Results on metalearners comparison with parametric models:

<img width="425" height="281" alt="image" src="https://github.com/user-attachments/assets/5c0cf717-1a6c-4682-9289-88909a475cac" />

Under nonlinear and heterogeneous treatment effects, the S‑learner achieves more stable individual treatment effect estimation, while the T‑learner suffers from increased variance and extreme errors.

S-learner RMSE: 1.023856192566453
T-learner RMSE: 1.2740642218320983

Results on metalearners comparison with ml models:

<img width="425" height="281" alt="image" src="https://github.com/user-attachments/assets/785ae315-cf94-4456-91c4-52ef86f4eaa0" />

Random Forest–based S‑learners demonstrate more stable CATE estimation than T‑learners under nonlinear and heterogeneous treatment effects, as evidenced by lower 

median errors and reduced tail behavior.

S-learner RF RMSE: 1.2944342252623569
T-learner RF RMSE: 1.5478131448350745

Overall on RMSE:
According to the RMSE criterion, the parametric S‑learner achieves the best performance among all evaluated models. While Random Forest–based meta‑learners offer greater flexibility, they incur higher variance, resulting in inferior RMSE in this finite‑sample setting.

