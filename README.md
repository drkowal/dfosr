# dfosr
Dynamic function-on-scalars regression

The package implements the dynamic function-on-scalars regression model (https://arxiv.org/abs/1806.01460), 
where a functional data response is regressed on scalar predictors. 
Here, both the functional response and the scalar predictors may be time-ordered. 
The functions are modeled nonparametrically using an unknown basis expansion, which is learned from the data. 
The regression coefficients themselves are functions, and may be dynamic as well. 
The model is represented using a state space construction, which allows for time-varying parameter regression and autocorrelated errors. 

Important special cases are also included: fdlm() implements a model for a time series of functional data (without predictors),
fosr() implements a function-on-scalars regression (without time-ordering), 
and fosr_ar() extends fosr() to include autocorrelated errors. The most general function is dfosr(), 
which allows for dynamic or non-dynamic regression coefficient functions, includes a variety of evolution equations, 
and may incorporate a stochastic volatility model for the observation error variance. In all cases, the models are 
implemented using a Gibbs sampler. Note that missing values (NAs) are allowed, and will be automatically imputed 
within the MCMC algorithm. 


# Example usage

```
# Load the package:
library(dfosr)

T = 200 # Number of time points
m = 50  # Number of observation points
p_0 = 2 # Number of true zero regression coefficients
p_1 = 2 # Number of true nonzero regression coefficients

# Simulate and store the output:
sim_data = simulate_dfosr(T = T, m = m, p_0 = p_0, p_1 = p_1)
Y = sim_data$Y; X = sim_data$X; tau = sim_data$tau

# Number of predictors, including the intercept:
p = ncol(X) 

# Run the MCMC with K = 6 factors
out = dfosr(Y = Y, tau = tau, X = X, 
            K = 6, # Number of factors
            factor_model = 'AR', # Model for the evolution
            use_dynamic_reg = TRUE, # Dynamic or non-dynamic regression coefficients
            mcmc_params = list("beta", "fk", "alpha", "Yhat", "Ypred") # Parameters to save
            )

# Plot a dynamic regression coefficient function
j = 3 # choose a predictor
post_alpha_tilde_j = get_post_alpha_tilde(out$fk, out$alpha[,,j,])

# Posterior mean:
alpha_tilde_j_pm = colMeans(post_alpha_tilde_j)

# Lower and Upper 95% credible intervals:
alpha_tilde_j_lower = apply(post_alpha_tilde_j, 2:3, quantile, c(0.05/2))
alpha_tilde_j_upper = apply(post_alpha_tilde_j, 2:3, quantile, c(1 - 0.05/2))

# Plot lower pointwise interval:
filled.contour(1:T, tau, alpha_tilde_j_lower,
               zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
               color = terrain.colors,
               xlab = 'Time', ylab = expression(tau),
               main = paste('Lower 95% Credible Intervals, j =',j))
               
# Plot posterior Mean:
filled.contour(1:T, tau, alpha_tilde_j_pm,
               zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
               color = terrain.colors,
               xlab = 'Time', ylab = expression(tau),
               main = paste('Posterior Mean, j =',j))
               
# Plot upper pointwise interval:
filled.contour(1:T, tau, sim_data$alpha_tilde_true[,j,],
               zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
               color = terrain.colors,
               xlab = 'Time', ylab = expression(tau),
               main = paste('Upper 95% Credible Intervals, j =',j))
               
# Truth:
filled.contour(1:T, tau, alpha_tilde_j_upper,
               zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
               color = terrain.colors,
               xlab = 'Time', ylab = expression(tau),
               main = paste('True regression coefficients, j =',j))

# Verify by plotting at two time slices:
t1 = ceiling(0.2*T); # Time t1
plot_curve(post_f = post_alpha_tilde_j[,t1,],
           tau = tau,
           main = paste('Predictor j =',j,'at time t =',t1))
           
# Add the true regression coefficient function:
lines(tau, sim_data$alpha_tilde_true[t1,j,], lwd=8, col='black', lty=6)

t2 = ceiling(0.8*T) # Time t2
plot_curve(post_f = post_alpha_tilde_j[,t2,],
           tau = tau,
           main = paste('Predictor j =',j,'at time t =',t2))
           
# Add the true regression coefficient function:
lines(tau, sim_data$alpha_tilde_true[t2,j,], lwd=8, col='black', lty=6)

# Plot the factors:
plot_factors(post_beta = out$beta)

# Plot the loading curves:
plot_flc(post_fk = out$fk, tau = tau)

# Plot a fitted value w/ posterior predictive credible intervals:
i = sample(1:T, 1); # Select a random time i
plot_fitted(y = Y[i,],
            mu = colMeans(out$Yhat)[i,],
            postY = out$Ypred[,i,],
            y_true = sim_data$Y_true[i,],
            t01 = tau)

```
