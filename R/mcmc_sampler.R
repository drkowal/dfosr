# Note: to update, use "git push -u origin master" (C******7)


#' MCMC Sampling Algorithm for the Functional Dynamic Linear Model
#'
#' Runs the MCMC for the functional dynamic linear model (with no predictor variables).
#' Models for the (dynamic) factors include independent factors, an AR(1) model,
#' and a random walk model.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param K the number of factors; if NULL, use SVD-based proportion of variability explained
#' @param factor_model model for the (dynamic) factors;
#' must be one of
#' \itemize{
#' \item "IND" (independent errors)
#' \item "AR" (stationary autoregression of order 1)
#' \item "RW" (random walk model)
#' }
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (dynamic factors)
#' \item "fk" (loading curves)
#' \item "mu_k" (intercept)
#' \item "sigma_et" (observation error SD; possibly dynamic)
#' \item "Yhat" (fitted values)
#' \item "Ypred" (posterior predictive values)
#' \item "Yfore" (one-step forecast; includes the estimate and the distribution)
#' }
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1}
#' @param use_obs_SV logical; when TRUE, include a stochastic volatility model
#' for the observation error variance
#' @param includeBasisInnovation logical; when TRUE, include an iid basis coefficient term for residual correlation
#' (i.e., the idiosyncratic error term for a factor model on the full basis matrix)
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#' @note If \code{Tm} is large, then storing all posterior samples for \code{Yhat} or \code{Ypred}, which are \code{nsave x T x m},  may be inefficient
#'
#' @examples
## Example 1: model-fitting and plotting
#' # Simulate some data (w/o predictors):
#' sim_data = simulate_dfosr(T = 100, m = 20,
#'                           p_0 = 0, p_1 = 0)
#' Y = sim_data$Y; tau = sim_data$tau
#' T = nrow(Y); m = ncol(Y); # Dimensions
#'
#' # Run the MCMC w/ K = 6:
#' out = fdlm(Y = Y, tau = tau, K = 6,
#'            factor_model = 'AR',
#'            mcmc_params = list("beta", "fk", "Yhat", "Ypred"))
#'
#' # Plot the factors:
#' plot_factors(post_beta = out$beta)
#'
#' # Plot the loading curves:
#' plot_flc(post_fk = out$fk, tau = tau)
#'
#' # Plot a fitted value w/ posterior predictive credible intervals:
#' i = sample(1:T, 1); # Select a random time i
#' plot_fitted(y = Y[i,],
#'             mu = colMeans(out$Yhat)[i,],
#'             postY = out$Ypred[,i,],
#'             y_true = sim_data$Y_true[i,],
#'             t01 = tau)
#'
# ## Example 2: forecasting
# # Now consider a forecasting the final time point:
# # Store the last curve separately, then delete
# Y_Tp1 = Y[T,]; Y_true_Tp1 = sim_data$Y_true[T,]
# Y = Y[-T,]; T = nrow(Y);
#
# # Run the MCMC w/ K = 6:
# out = fdlm(Y = Y, tau = tau, K = 6,
#            factor_model = 'AR',
#            mcmc_params = list("Yfore"))
#
# # Plot the results:
# plot_fitted(y = Y_Tp1, # (unobserved) curve
#             mu = colMeans(out$Yfore_hat),
#             postY = out$Yfore,
#             y_true = Y_true_Tp1,
#             t01 = tau)
# # Add the most recent observed curve:
# lines(tau, Y[T,], type='p', pch = 2)
#'
#' @import  KFAS truncdist
#' @export
fdlm = function(Y, tau, K = NULL,
                factor_model = "RW",
                nsave = 1000, nburn = 1000, nskip = 3,
                mcmc_params = list("beta", "fk"),
                use_obs_SV = FALSE,
                X_Tp1 = 1,
                includeBasisInnovation = FALSE,
                computeDIC = TRUE){

  # Checks will be done in dfosr():
  mcmc_output = dfosr(Y = Y, tau = tau, K = K,
                      X = NULL, # Key term!
                      factor_model = factor_model,
                      nsave = nsave, nburn = nburn, nskip = nskip,
                      mcmc_params = mcmc_params,
                      X_Tp1 = X_Tp1,
                      use_obs_SV = use_obs_SV,
                      includeBasisInnovation = includeBasisInnovation,
                      computeDIC = computeDIC)

  # Other DFOSR terms are not relevant
  return(mcmc_output)
}
#' MCMC Sampling Algorithm for the Function-on-Scalars Regression Model
#'
#' Runs the MCMC for the function-on-scalars regression model based on
#' a reduced-rank expansion. Here we assume the factor regression has independent errors,
#' as well as some additional default conditions.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param X the \code{T x p} matrix of predictors; if NULL, only include an intercept
#' @param K the number of factors; if NULL, use SVD-based proportion of variability explained
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (factors)
#' \item "fk" (loading curves)
#' \item "alpha" (regression coefficients)
#' \item "mu_k" (intercept term for factor k)
#' \item "sigma_e" (observation error SD)
#' \item "sigma_g" (random effects SD)
#' \item "Yhat" (fitted values)
#' \item "Ypred" (posterior predictive values)
#' }
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#' @note If \code{Tm} is large, then storing all posterior samples for \code{Yhat} or \code{Ypred}, which are \code{nsave x T x m},  may be inefficient
#'
#' @examples
#' \dontrun{
#' # Simulate some data (w/ NAs):
#' sim_data = simulate_dfosr(T = 100, m = 20,
#'                           p_0 = 2, p_1 = 2,
#'                           use_dynamic_reg = FALSE,
#'                           prop_missing = 0.5)
#' Y = sim_data$Y; X = sim_data$X; tau = sim_data$tau
#' T = nrow(Y); m = ncol(Y); p = ncol(X) # Dimensions
#'
#' # Run the MCMC w/ K = 6:
#' out = fosr(Y = Y, tau = tau, X = X, K = 6,
#'            mcmc_params = list("beta", "fk", "alpha", "Yhat", "Ypred"))
#'
#' # Plot a regression coefficient function (Note: these are non-dynamic)
#' j = 3 # choose a predictor
#' post_alpha_tilde_j = get_post_alpha_tilde(out$fk, out$alpha[,j,])
#' plot_curve(post_f = post_alpha_tilde_j,
#'            tau = tau,
#'            main = paste('Posterior Mean and Credible bands, j =',j))
#' # Add the true regression coefficient function:
#' lines(tau, sim_data$alpha_tilde_true[1,j,], lwd=8, col='black', lty=6)
#'
#' # Plot the factors:
#' plot_factors(post_beta = out$beta)
#'
#' # Plot the loading curves:
#' plot_flc(post_fk = out$fk, tau = tau)
#'
#' # Plot a fitted value w/ posterior predictive credible intervals:
#' i = sample(1:T, 1); # Select a random time i
#' plot_fitted(y = Y[i,],
#'             mu = colMeans(out$Yhat)[i,],
#'             postY = out$Ypred[,i,],
#'             y_true = sim_data$Y_true[i,],
#'             t01 = tau)
#'}
#' @import  KFAS truncdist
#' @export
fosr = function(Y, tau, X = NULL, K = NULL,
                nsave = 1000, nburn = 1000, nskip = 3,
                mcmc_params = list("beta", "fk", "alpha", "sigma_e", "sigma_g"),
                computeDIC = TRUE){

  # Some options (for now):
  sample_nu = TRUE # Sample DF parameter, or fix at nu=3?
  sample_a1a2 = TRUE # Sample a1, a2, or fix at a1=2, a2=3?

  #----------------------------------------------------------------------------
  # Assume that we've done checks elsewhere
  #----------------------------------------------------------------------------
  # Convert tau to matrix, if necessary:
  tau = as.matrix(tau)

  # Compute the dimensions:
  T = nrow(Y); m = ncol(Y); d = ncol(tau)

  # Rescale observation points to [0,1]
  tau01 = apply(tau, 2, function(x) (x - min(x))/(max(x) - min(x)))

  # Rescale by observation SD (and correct parameters later):
  sdY = sd(Y, na.rm=TRUE);
  Y = Y/sdY;
  #----------------------------------------------------------------------------
  # Initialize the main terms:

  # Initialize the FLC coefficients and factors:
  inits = fdlm_init(Y, tau, K); Beta = inits$Beta; Psi = inits$Psi; splineInfo = inits$splineInfo
  K = ncol(Beta) # to be sure we have the right value

  # Also use the imputed data values here for initialization:
  Yna = Y # The original data, including NAs
  any.missing = any(is.na(Yna)) # Any missing obs?
  if(any.missing){na.ind = which(is.na(Yna), arr.ind = TRUE); Y = inits$Y0}
  BtY = tcrossprod(t(splineInfo$Bmat), Y)

  # FLC matrix:
  Fmat = splineInfo$Bmat%*%Psi

  # Initialize the conditional expectation:
  Yhat = tcrossprod(Beta, Fmat)

  # Initialize the (time-dependent) observation error SD:
  sigma_e = sd(Y - Yhat, na.rm=TRUE); sigma_et = rep(sigma_e, T)

  # Initialize the FLC smoothing parameters (conditional MLE):
  tau_f_k = apply(Psi, 2, function(x) (ncol(splineInfo$Bmat) - (d+1))/crossprod(x, splineInfo$Omega)%*%x)
  #----------------------------------------------------------------------------
  # Predictors:
  if(!is.null(X)){
    # Assuming we have some predictors:
    X = as.matrix(X)

    # Remove any predictors which are constants/intercepts:
    const.pred = apply(X, 2, function(x) all(diff(x) == 0))
    if(any(const.pred)) X = as.matrix(X[,!const.pred])

    # Center and scale the (non-constant) predictors:
    # Note: may not be appropriate for intervention effects!
    #X = scale(X)
  }
  # Include an intercept:
  X = cbind(rep(1, T), X); #colnames(X)[1] = paste(intercept_model, "-Intercept", sep='')

  # Number of predictors (including the intercept)
  p = ncol(X)

  #----------------------------------------------------------------------------
  # Initialize the regression terms (and the mean term)
  alpha_pk = matrix(0, nrow = p, ncol = K) # Regression coefficients
  gamma_tk = matrix(0, nrow = T, ncol = K) # Residuals

  # Initialize the regression coefficients via sampling (p >= T) or OLS (p < T)
  for(k in 1:K) {
    if(p >= T){
      alpha_pk[,k] = sampleFastGaussian(Phi = X/sigma_et,
                                        Ddiag = rep(.01*sigma_e^2, p),
                                        alpha = tcrossprod(Y, t(Fmat[,k]))/sigma_e)
    } else alpha_pk[,k] = lm(Beta[,k] ~ X - 1)$coef

    # Residuals:
    gamma_tk[,k] = Beta[,k] - X%*%alpha_pk[,k]
  }

  # Intercept term:
  mu_k = as.matrix(alpha_pk[1,])

  # SD term for mu_k:
  a1_mu = 2; a2_mu = 3
  delta_mu_k = sampleMGP(matrix(mu_k, ncol = K), rep(1,K), a1 = a1_mu, a2 = a2_mu)
  sigma_mu_k = 1/sqrt(cumprod(delta_mu_k))
  #----------------------------------------------------------------------------
  # Initialize the corresponding SD term(s):
  xi_gamma_tk = 1/gamma_tk^2; # Precision scale
  nu = 3  # (initial) degrees of freedom

  # MGP term:
  a1_gamma = 2; a2_gamma = 3;
  delta_gamma_k = rep(1,K); sigma_delta_k = 1/sqrt(cumprod(delta_gamma_k))

  # Update the error SD for gamma:
  sigma_gamma_tk = rep(sigma_delta_k, each = T)/sqrt(xi_gamma_tk)
  #----------------------------------------------------------------------------
  if(p > 1){
    omega = matrix(alpha_pk[-1,], nrow = p-1) # Not the intercept

    # predictor p, factor k:
    sigma_omega_pk = abs(omega)
    xi_omega_pk = matrix(1, nrow = p-1, ncol = K) # PX term

    # predictor p:
    lambda_omega_p = rowMeans(sigma_omega_pk)
    xi_omega_p = rep(1, (p-1)) # PX term

    # global:
    lambda_omega_0 = mean(lambda_omega_p)
    xi_omega_0 = 1 # PX term
  }
  #----------------------------------------------------------------------------
  # Store the MCMC output in separate arrays (better computation times)
  mcmc_output = vector('list', length(mcmc_params)); names(mcmc_output) = mcmc_params
  if(!is.na(match('beta', mcmc_params))) post.beta = array(NA, c(nsave, T, K))
  if(!is.na(match('fk', mcmc_params))) post.fk = array(NA, c(nsave, m, K))
  if(!is.na(match('alpha', mcmc_params))) post.alpha = array(NA, c(nsave, p, K))
  if(!is.na(match('sigma_e', mcmc_params)) || computeDIC) post.sigma_e = array(NA, c(nsave, 1))
  if(!is.na(match('sigma_g', mcmc_params))) post.sigma_g = array(NA, c(nsave, T, K))
  if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat = array(NA, c(nsave, T, m))
  if(!is.na(match('Ypred', mcmc_params))) post.Ypred = array(NA, c(nsave, T, m))
  if(computeDIC) post_loglike = numeric(nsave)

  # Total number of MCMC simulations:
  nstot = nburn+(nskip+1)*(nsave)
  skipcount = 0; isave = 0 # For counting

  # Run the MCMC:
  timer0 = proc.time()[3] # For timing the sampler
  for(nsi in 1:nstot){

    #----------------------------------------------------------------------------
    # Step 1: Impute the data, Y:
    #----------------------------------------------------------------------------
    if(any.missing){
      Y[na.ind] = Yhat[na.ind] + sigma_et[na.ind[,1]]*rnorm(nrow(na.ind))
      BtY = tcrossprod(t(splineInfo$Bmat), Y)
    }
    #----------------------------------------------------------------------------
    # Step 2: Sample the FLCs
    #----------------------------------------------------------------------------
    # Sample the FLCs
    Psi = fdlm_flc(BtY = BtY,
                   Beta  = Beta,
                   Psi = Psi,
                   BtB = splineInfo$BtB, #diag(nrow(BtY)),
                   Omega = splineInfo$Omega,
                   lambda = tau_f_k,
                   sigmat2 = sigma_et^2)
    # And update the loading curves:
    Fmat = splineInfo$Bmat%*%Psi;

    # Sample the smoothing parameters:
    tau_f_k = sample_lambda(tau_f_k, Psi, Omega = splineInfo$Omega, d = d, uniformPrior = TRUE, orderLambdas = FALSE)
    #----------------------------------------------------------------------------
    # Step 3: Sample the regression coefficients (and therefore the factors)
    #----------------------------------------------------------------------------
    # Pseudo-response and pseudo-variance:
    Y_tilde = crossprod(BtY, Psi); sigma_tilde = sigma_et

    # Draw Separately for each k:
    for(k in 1:K){
      # Marginalize over gamma_{tk} to sample {alpha_pk}_p for fixed k:
      y_tilde_k = Y_tilde[,k]; sigma_tilde_k = sqrt(sigma_tilde^2 + sigma_gamma_tk[,k]^2)

      if(p >= T){
        # Fast sampler for p >= T (BHATTACHARYA et al., 2016)
        alpha_pk[,k] = sampleFastGaussian(Phi = X/sigma_tilde_k,
                                          Ddiag = as.numeric(c(sigma_mu_k[k],sigma_omega_pk[,k])^2),
                                          alpha = y_tilde_k/sigma_tilde_k)
      } else {
        # Fast sampler for p < T (Rue, 2001?)
        if(p > 1){
          chQ_k = chol(crossprod(X/sigma_tilde_k) + diag(as.numeric(1/c(sigma_mu_k[k],sigma_omega_pk[,k])^2)))
        } else chQ_k = chol(crossprod(X/sigma_tilde_k) + diag(as.numeric(1/c(sigma_mu_k[k])^2), p))
        ell_k = crossprod(X, y_tilde_k/sigma_tilde_k^2)
        alpha_pk[,k] = backsolve(chQ_k, forwardsolve(t(chQ_k), ell_k) + rnorm(p))
      }
    }

    # And sample the errors gamma_tk:
    postSD = 1/sqrt(rep(1/sigma_tilde^2, times = K) + matrix(1/sigma_gamma_tk^2))
    postMean = matrix((Y_tilde - X%*%alpha_pk)/rep(sigma_tilde^2, times = K))*postSD^2
    gamma_tk = matrix(rnorm(n = T*K, mean = postMean, sd = postSD), nrow = T)

    # Update the factors:
    Beta = X%*%alpha_pk + gamma_tk

    # And the fitted curves:
    Yhat = tcrossprod(Beta, Fmat)
    #----------------------------------------------------------------------------
    # Step 4: Sample the observation error variance
    #----------------------------------------------------------------------------
    # Or use uniform prior?
    sigma_e = 1/sqrt(rgamma(n = 1, shape = sum(!is.na(Y))/2, rate = sum((Y - Yhat)^2, na.rm=TRUE)/2))
    sigma_et = rep(sigma_e, T)

    #----------------------------------------------------------------------------
    # Step 5: Sample the intercept/gamma parameters (Note: could use ASIS)
    #----------------------------------------------------------------------------
    mu_k = alpha_pk[1,]

    # Prior variance: MGP
    # Mean Part
    delta_mu_k =  sampleMGP(theta.jh = matrix(mu_k, ncol = K),
                            delta.h = delta_mu_k,
                            a1 = a1_mu, a2 = a2_mu)
    sigma_mu_k = 1/sqrt(cumprod(delta_mu_k))
    # And hyperparameters:
    if(sample_a1a2){
      a1_mu = uni.slice(a1_mu, g = function(a){
        dgamma(delta_mu_k[1], shape = a, rate = 1, log = TRUE) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)}, lower = 0, upper = Inf)
      a2_mu = uni.slice(a2_mu,g = function(a){
        sum(dgamma(delta_mu_k[-1], shape = a, rate = 1, log = TRUE)) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)},lower = 0, upper = Inf)
    }

    # Variance part:
    # Standardize, then reconstruct as matrix of size T x K:
    delta_gamma_k = sampleMGP(theta.jh = matrix(gamma_tk*sqrt(xi_gamma_tk), ncol = K),
                              delta.h = delta_gamma_k,
                              a1 = a1_gamma, a2 = a2_gamma)
    sigma_delta_k = 1/sqrt(cumprod(delta_gamma_k))
    # And hyperparameters:
    if(sample_a1a2){
      a1_gamma = uni.slice(a1_gamma, g = function(a){
        dgamma(delta_gamma_k[1], shape = a, rate = 1, log = TRUE) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)}, lower = 0, upper = Inf)
      a2_gamma = uni.slice(a2_gamma, g = function(a){
        sum(dgamma(delta_gamma_k[-1], shape = a, rate = 1, log = TRUE)) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)},lower = 0, upper = Inf)
    }

    # Sample the corresponding prior variance term(s):
    xi_gamma_tk = matrix(rgamma(n = T*K,
                              shape = nu/2 + 1/2,
                              rate = nu/2 + (gamma_tk/rep(sigma_delta_k, each = T))^2/2), nrow = T)
    # Sample degrees of freedom?
    if(sample_nu){
      nu = uni.slice(nu, g = function(nu){
        sum(dgamma(xi_gamma_tk, shape = nu/2, rate = nu/2, log = TRUE)) +
          dunif(nu, min = 2, max = 128, log = TRUE)}, lower = 2, upper = 128)
    }

    # Update the error SD for gamma:
    sigma_gamma_tk = rep(sigma_delta_k, each = T)/sqrt(xi_gamma_tk)
    #----------------------------------------------------------------------------
    # Step 6: Sample the non-intercept parameters:
    #----------------------------------------------------------------------------
    # Non-intercept term:
    if(p > 1){

      omega = matrix(alpha_pk[-1,], nrow = p-1) # Not the intercept

      #----------------------------------------------------------------------------
      # predictor p, factor k:
      omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
      sigma_omega_pk = matrix(1/sqrt(rgamma(n = (p-1)*K,
                                            shape = 1/2 + 1/2,
                                            rate = xi_omega_pk + omega2/2)), nrow = p-1)
      xi_omega_pk = matrix(rgamma(n = (p-1)*K,
                                  shape = 1/2 + 1/2,
                                  rate = rep(1/lambda_omega_p^2, times = K) + 1/sigma_omega_pk^2), nrow = p-1)
      #----------------------------------------------------------------------------
      # predictor p:
      lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                     shape = 1/2 + K/2,
                                     rate = xi_omega_p + rowSums(xi_omega_pk)))
      xi_omega_p = rgamma(n = p-1,
                          shape = 1/2 + 1/2,
                          rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
      #----------------------------------------------------------------------------
      # global:
      lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                     shape = 1/2 + (p-1)/2,
                                     rate = xi_omega_0 + sum(xi_omega_p)))
      xi_omega_0 = rgamma(n = 1,
                          shape = 1/2 + 1/2,
                          rate = 1 + 1/lambda_omega_0^2)
    }
    #----------------------------------------------------------------------------
    # Step 7: Adjust the ordering
    #----------------------------------------------------------------------------
    #if(nsi == 10 && K > 1){adjOrder = order(tau_f_k, decreasing = TRUE); tau_f_k = tau_f_k[adjOrder]; Psi = Psi[,adjOrder]; Beta = as.matrix(Beta[,adjOrder])}

    # Store the MCMC output:
    if(nsi > nburn){
      # Increment the skip counter:
      skipcount = skipcount + 1

      # Save the iteration:
      if(skipcount > nskip){
        # Increment the save index
        isave = isave + 1

        # Save the MCMC samples:
        if(!is.na(match('beta', mcmc_params))) post.beta[isave,,] = Beta
        if(!is.na(match('fk', mcmc_params))) post.fk[isave,,] = Fmat
        if(!is.na(match('alpha', mcmc_params))) post.alpha[isave,,] = alpha_pk
        if(!is.na(match('sigma_e', mcmc_params)) || computeDIC) post.sigma_e[isave,] = sigma_e
        if(!is.na(match('sigma_g', mcmc_params))) post.sigma_g[isave,,] = sigma_gamma_tk
        if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat[isave,,] = Yhat
        if(!is.na(match('Ypred', mcmc_params))) post.Ypred[isave,,] = rnorm(n = T*m, mean = matrix(Yhat), sd = rep(sigma_et,m))
        if(computeDIC) post_loglike[isave] = sum(dnorm(matrix(Yna), mean = matrix(Yhat), sd = rep(sigma_et,m), log = TRUE), na.rm = TRUE)

        # And reset the skip counter:
        skipcount = 0
      }
    }
    computeTimeRemaining(nsi, timer0, nstot, nrep = 1000)
  }

  # Store the results (and correct for rescaling by sdY):
  if(!is.na(match('beta', mcmc_params))) mcmc_output$beta = post.beta*sdY
  if(!is.na(match('fk', mcmc_params))) mcmc_output$fk = post.fk
  if(!is.na(match('alpha', mcmc_params))) mcmc_output$alpha = post.alpha*sdY
  if(!is.na(match('sigma_e', mcmc_params))) mcmc_output$sigma_e = post.sigma_e*sdY
  if(!is.na(match('sigma_g', mcmc_params))) mcmc_output$sigma_g = post.sigma_g*sdY
  if(!is.na(match('Yhat', mcmc_params))) mcmc_output$Yhat = post.Yhat*sdY
  if(!is.na(match('Ypred', mcmc_params))) mcmc_output$Ypred = post.Ypred*sdY

  if(computeDIC){
    # Log-likelihood evaluated at posterior means:
    loglike_hat = sum(dnorm(matrix(Yna),
                            mean = matrix(colMeans(post.Yhat)),
                            sd = rep(colMeans(post.sigma_e), m*T),
                            log = TRUE), na.rm=TRUE)

    # Effective number of parameters (Note: two options)
    p_d = c(2*(loglike_hat - mean(post_loglike)),
            2*var(post_loglike))
    # DIC:
    DIC = -2*loglike_hat + 2*p_d

    # Store the DIC and the effective number of parameters (p_d)
    mcmc_output$DIC = DIC; mcmc_output$p_d = p_d
  }

  print(paste('Total time: ', round((proc.time()[3] - timer0)), 'seconds'))

  return (mcmc_output);
}
#' MCMC Sampling Algorithm for the Function-on-Scalars Regression Model with
#' Time Series Errors
#'
#' Runs the MCMC for the function-on-scalars regression model based on
#' an reduced-rank expansion. Here we assume the factor regression has AR(1) errors.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param X the \code{T x p} matrix of predictors; if NULL, only include an intercept
#' @param K the number of factors; if NULL, use SVD-based proportion of variability explained
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (dynamic factors)
#' \item "fk" (loading curves)
#' \item "alpha" (regression coefficients)
#' \item "mu_k" (intercept term for factor k)
#' \item "ar_phi" (AR coefficients for each k under AR(1) model)
#' \item "sigma_et" (observation error SD)
#' \item "Yhat" (fitted values)
#' \item "Ypred" (posterior predictive values)
#' \item "Yfore" (one-step forecast; includes the estimate and the distribution)
#' }
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1}
#' @param use_obs_SV logical; when TRUE, include a stochastic volatility model
#' for the observation error variance
#' @param includeBasisInnovation logical; when TRUE, include an iid basis coefficient term for residual correlation
#' (i.e., the idiosyncratic error term for a factor model on the full basis matrix)
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#' @note If \code{Tm} is large, then storing all posterior samples for \code{Yhat} or \code{Ypred}, which are \code{nsave x T x m},  may be inefficient
#'
#' @examples
#' \dontrun{
# ## Example 1: model-fitting and plotting
#' # Simulate some data:
#' sim_data = simulate_dfosr(T = 100, m = 20, p_0 = 2, p_1 = 2, use_dynamic_reg = FALSE)
#' Y = sim_data$Y; X = sim_data$X; tau = sim_data$tau
#' T = nrow(Y); m = ncol(Y); p = ncol(X) # Dimensions
#'
#' # Run the MCMC w/ K = 6:
#' out = fosr_ar(Y = Y, tau = tau, X = X, K = 6,
#'               mcmc_params = list("beta", "fk", "alpha", "ar_phi", "Yhat", "Ypred"))
#'
#' # Plot a regression coefficient function (Note: these are non-dynamic)
#' j = 3 # choose a predictor
#' post_alpha_tilde_j = get_post_alpha_tilde(out$fk, out$alpha[,1,j,])
#' plot_curve(post_f = post_alpha_tilde_j,
#'            tau = tau,
#'            main = paste('Posterior Mean and Credible bands, j =',j))
#' # Add the true regression coefficient function:
#' lines(tau, sim_data$alpha_tilde_true[1,j,], lwd=8, col='black', lty=6)
#'
#' # Evidence for autocorrelation via the AR(1) coefficients:
#' plot(as.ts(out$ar_phi))
#' apply(out$ar_phi, 2, quantile, c(0.05/2, 1 - 0.05/2)) # 95% credible intervals
#'
#' # Plot the factors:
#' plot_factors(post_beta = out$beta)
#'
#' # Plot the loading curves:
#' plot_flc(post_fk = out$fk, tau = tau)
#'
#' # Plot a fitted value w/ posterior predictive credible intervals:
#' i = sample(1:T, 1); # Select a random time i
#' plot_fitted(y = Y[i,],
#'             mu = colMeans(out$Yhat)[i,],
#'             postY = out$Ypred[,i,],
#'             y_true = sim_data$Y_true[i,],
#'             t01 = tau)
#'
# ## Example 2: forecasting
# # Now consider a forecasting the final time point:
# # Store the last curve separately, then delete
# Y_Tp1 = Y[T,]; X_Tp1 = X[T,]; Y_true_Tp1 = sim_data$Y_true[T,]
# Y = Y[-T,]; X = X[-T,]; T = nrow(Y);
#
# # Run the MCMC w/ K = 6:
# out = fosr_ar(Y = Y, tau = tau, X = X, K = 6,
#               X_Tp1 = X_Tp1,
#               mcmc_params = list("Yfore"))
#
# # Plot the results:
# plot_fitted(y = Y_Tp1, # (unobserved) curve
#             mu = colMeans(out$Yfore_hat),
#             postY = out$Yfore,
#             y_true = Y_true_Tp1,
#             t01 = tau)
# # Add the most recent observed curve:
# lines(tau, Y[T,], type='p', pch = 2)
#'
#'}
#'
#' @import  KFAS truncdist
#' @export
fosr_ar = function(Y, tau, X = NULL, K = NULL,
                   nsave = 1000, nburn = 1000, nskip = 3,
                   mcmc_params = list("beta", "fk", "alpha"),
                   X_Tp1 = 1,
                   use_obs_SV = FALSE,
                   includeBasisInnovation = FALSE,
                   computeDIC = TRUE){

  # Checks will be done in dfosr():
  mcmc_output = dfosr(Y = Y, tau = tau, X = X, K = K,
                      factor_model = "AR",  # Key term: AR(1) errors
                      use_dynamic_reg = FALSE, # Key term: non-dynamic regression coefficients
                      nsave = nsave, nburn = nburn, nskip = nskip,
                      mcmc_params = mcmc_params,
                      X_Tp1 = X_Tp1,
                      use_obs_SV = use_obs_SV,
                      includeBasisInnovation = includeBasisInnovation,
                      computeDIC = computeDIC)
  # Other DFOSR terms are not relevant
  return(mcmc_output)
}
#' MCMC Sampling Algorithm for the Dynamic Function-on-Scalars Regression Model
#'
#' Runs the MCMC for the dynamic function-on-scalars regression model based on
#' an reduced-rank expansion. We include several options for dynamics in the model,
#' including autoregressive, random walk, or independent errors, as well as
#' dynamic or non-dynamic regression coefficients. Default shrinkage priors are assumed.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param X the \code{T x p} matrix of predictors; if NULL, only include an intercept
#' @param K the number of factors; if NULL, use SVD-based proportion of variability explained
#' @param factor_model model for the factor-level (regression) errors;
#' must be one of
#' \itemize{
#' \item "IND" (independent errors)
#' \item "AR" (stationary autoregression of order 1)
#' \item "RW" (random walk model)
#' }
#' @param use_dynamic_reg logical; if TRUE, regression coefficients are dynamic
#' (with random walk models), otherwise independent
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (dynamic factors)
#' \item "fk" (loading curves)
#' \item "alpha" (regression coefficients; possibly dynamic)
#' \item "mu_k" (intercept term for factor k)
#' \item "ar_phi" (AR coefficients for each k under AR(1) model)
#' \item "sigma_et" (observation error SD; possibly dynamic)
#' \item "Yhat" (fitted values)
#' \item "Ypred" (posterior predictive values)
#' \item "Yfore" (one-step forecast; includes the estimate and the distribution)
#' }
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1}
#' @param use_obs_SV logical; when TRUE, include a stochastic volatility model
#' for the observation error variance
#' @param includeBasisInnovation logical; when TRUE, include an iid basis coefficient term for residual correlation
#' (i.e., the idiosyncratic error term for a factor model on the full basis matrix)
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#' @note  This sampler loops over the k=1,...,K factors,
#' so the sampler is O(T*K*p^3) instead of O(T*(K*p)^3).
#'
#'
#' @examples
#' \dontrun{
#' # Simulate some data:
#' sim_data = simulate_dfosr(T = 200, m = 50, p_0 = 2, p_1 = 2)
#' Y = sim_data$Y; X = sim_data$X; tau = sim_data$tau
#' T = nrow(Y); m = ncol(Y); p = ncol(X) # Dimensions
#'
#' # Run the MCMC w/ K = 6:
#' out = dfosr(Y = Y, tau = tau, X = X, K = 6,
#'            factor_model = 'AR',
#'            use_dynamic_reg = TRUE,
#'            mcmc_params = list("beta", "fk", "alpha", "Yhat", "Ypred"))
#'
#' # Plot a dynamic regression coefficient function
#' j = 3 # choose a predictor
#' post_alpha_tilde_j = get_post_alpha_tilde(out$fk, out$alpha[,,j,])
#'
#' # Posterior mean:
#' alpha_tilde_j_pm = colMeans(post_alpha_tilde_j)
#'
#' # Lower and Upper 95% credible intervals:
#' alpha_tilde_j_lower = apply(post_alpha_tilde_j, 2:3, quantile, c(0.05/2))
#' alpha_tilde_j_upper = apply(post_alpha_tilde_j, 2:3, quantile, c(1 - 0.05/2))
#'
#' # Plot lower pointwise interval:
#' filled.contour(1:T, tau, alpha_tilde_j_lower,
#'                zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
#'                color = terrain.colors,
#'                xlab = 'Time', ylab = expression(tau),
#'                main = paste('Lower 95% Credible Intervals, j =',j))
#' # Plot posterior Mean:
#' filled.contour(1:T, tau, alpha_tilde_j_pm,
#'                zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
#'                color = terrain.colors,
#'                xlab = 'Time', ylab = expression(tau),
#'                main = paste('Posterior Mean, j =',j))
#' # Plot upper pointwise interval:
#' filled.contour(1:T, tau, sim_data$alpha_tilde_true[,j,],
#'                zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
#'                color = terrain.colors,
#'                xlab = 'Time', ylab = expression(tau),
#'                main = paste('Upper 95% Credible Intervals, j =',j))
#' # Truth:
#' filled.contour(1:T, tau, alpha_tilde_j_upper,
#'                zlim = range(alpha_tilde_j_lower, alpha_tilde_j_upper),
#'                color = terrain.colors,
#'                xlab = 'Time', ylab = expression(tau),
#'                main = paste('True regression coefficients, j =',j))
#'
#' # Verify by plotting at two time slices:
#' t1 = ceiling(0.2*T); # Time t1
#' plot_curve(post_f = post_alpha_tilde_j[,t1,],
#'            tau = tau,
#'            main = paste('Predictor j =',j,'at time t =',t1))
#' # Add the true regression coefficient function:
#' lines(tau, sim_data$alpha_tilde_true[t1,j,], lwd=8, col='black', lty=6)
#'
#' t2 = ceiling(0.8*T) # Time t2
#' plot_curve(post_f = post_alpha_tilde_j[,t2,],
#'            tau = tau,
#'            main = paste('Predictor j =',j,'at time t =',t2))
#' # Add the true regression coefficient function:
#' lines(tau, sim_data$alpha_tilde_true[t2,j,], lwd=8, col='black', lty=6)
#'
#' # Plot the factors:
#' plot_factors(post_beta = out$beta)
#'
#' # Plot the loading curves:
#' plot_flc(post_fk = out$fk, tau = tau)
#'
#' # Plot a fitted value w/ posterior predictive credible intervals:
#' i = sample(1:T, 1); # Select a random time i
#' plot_fitted(y = Y[i,],
#'             mu = colMeans(out$Yhat)[i,],
#'             postY = out$Ypred[,i,],
#'             y_true = sim_data$Y_true[i,],
#'             t01 = tau)
#'}
#' @import  KFAS truncdist
#' @export
dfosr = function(Y, tau, X = NULL, K = NULL,
                 factor_model = 'AR', use_dynamic_reg = TRUE,
                 nsave = 1000, nburn = 1000, nskip = 3,
                 mcmc_params = list("beta", "fk", "alpha", "mu_k", "ar_phi"),
                 X_Tp1 = 1,
                 use_obs_SV = FALSE,
                 includeBasisInnovation = FALSE,
                 computeDIC = TRUE){
  #----------------------------------------------------------------------------
  # Run some checks:
  # Convert to upper case, then check for matches to existing models:
  factor_model = toupper(factor_model);
  if(is.na(match(factor_model, c("RW", "AR", "IND"))))
    stop("The factor model must be one of 'RW', 'AR', or 'IND'")
  #----------------------------------------------------------------------------
  # Call the relevant function:

  if(factor_model == "RW"){
    mcmc_output = dfosr_rw(Y = Y, tau = tau, X = X, K = K,
                           use_dynamic_reg = use_dynamic_reg,
                           nsave = nsave, nburn = nburn, nskip = nskip,
                           mcmc_params = mcmc_params,
                           X_Tp1 = X_Tp1,
                           use_obs_SV = use_obs_SV,
                           includeBasisInnovation = includeBasisInnovation,
                           computeDIC = computeDIC)
  }
  if(factor_model == "AR"){
    mcmc_output = dfosr_ar(Y = Y, tau = tau, X = X, K = K,
                           use_dynamic_reg = use_dynamic_reg,
                           nsave = nsave, nburn = nburn, nskip = nskip,
                           mcmc_params = mcmc_params,
                           X_Tp1 = X_Tp1,
                           use_obs_SV = use_obs_SV,
                           includeBasisInnovation = includeBasisInnovation,
                           computeDIC = computeDIC)
  }
  if(factor_model == "IND"){
    mcmc_output = dfosr_ind(Y = Y, tau = tau, X = X, K = K,
                            use_dynamic_reg = use_dynamic_reg,
                            nsave = nsave, nburn = nburn, nskip = nskip,
                            mcmc_params = mcmc_params,
                            X_Tp1 = X_Tp1,
                            use_obs_SV = use_obs_SV,
                            includeBasisInnovation = includeBasisInnovation,
                            computeDIC = computeDIC)
  }
  return(mcmc_output)
}
#' MCMC Sampling Algorithm for the Dynamic Function-on-Scalars Regression Model with Independent Errors
#'
#' Runs the MCMC for the dynamic function-on-scalars regression model based on
#' an reduced-rank expansion. Here, we assume the factor regression has independent errors.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param X the \code{T x p} matrix of predictors; if NULL, only include an intercept
#' @param K the number of factors; if NULL, use SVD-based proportion of variability explained
#' @param use_dynamic_reg logical; if TRUE, regression coefficients are dynamic
#' (with random walk models), otherwise independent
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (dynamic factors)
#' \item "fk" (loading curves)
#' \item "alpha" (regression coefficients; possibly dynamic)
#' \item "mu_k" (intercept term for factor k)
#' \item "sigma_et" (observation error SD; possibly dynamic)
#' \item "Yhat" (fitted values)
#' \item "Ypred" (posterior predictive values)
#' \item "Yfore" (one-step forecast; includes the estimate and the distribution)
#' }
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1}
#' @param use_obs_SV logical; when TRUE, include a stochastic volatility model
#' for the observation error variance
#' @param includeBasisInnovation logical; when TRUE, include an iid basis coefficient term for residual correlation
#' (i.e., the idiosyncratic error term for a factor model on the full basis matrix)
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#' @import  KFAS truncdist
dfosr_ind = function(Y, tau, X = NULL, K = NULL,
                     use_dynamic_reg = TRUE,
                     nsave = 1000, nburn = 1000, nskip = 3,
                     mcmc_params = list("beta", "fk", "alpha"),
                     X_Tp1 = 1,
                     use_obs_SV = FALSE,
                     includeBasisInnovation = FALSE,
                     computeDIC = TRUE){
  # Some options (for now):
  sample_nu = TRUE # Sample DF parameter, or fix at nu=3?
  sample_a1a2 = TRUE # Sample a1, a2, or fix at a1=2, a2=3?

  #----------------------------------------------------------------------------
  # Assume that we've done checks elsewhere
  #----------------------------------------------------------------------------
  # Convert tau to matrix, if necessary:
  tau = as.matrix(tau)

  # Compute the dimensions:
  T = nrow(Y); m = ncol(Y); d = ncol(tau)

  # Rescale observation points to [0,1]
  tau01 = apply(tau, 2, function(x) (x - min(x))/(max(x) - min(x)))

  # Rescale by observation SD (and correct parameters later):
  sdY = sd(Y, na.rm=TRUE);
  Y = Y/sdY;
  #----------------------------------------------------------------------------
  # Initialize the main terms:

  # Initialize the FLC coefficients and factors:
  inits = fdlm_init(Y, tau, K); Beta = inits$Beta; Psi = inits$Psi; splineInfo = inits$splineInfo
  K = ncol(Beta) # to be sure we have the right value

  # Also use the imputed data values here for initialization:
  Yna = Y # The original data, including NAs
  any.missing = any(is.na(Yna)) # Any missing obs?
  if(any.missing){na.ind = which(is.na(Yna), arr.ind = TRUE); Y = inits$Y0}
  BtY = tcrossprod(t(splineInfo$Bmat), Y)

  # FLC matrix:
  Fmat = splineInfo$Bmat%*%Psi

  # Initialize the conditional expectation:
  BetaPsit = tcrossprod(Beta, Psi); Btheta = tcrossprod(BetaPsit, splineInfo$Bmat)

  # Initialize the basis coefficient residuals and the corresponding standard deviation
  if(includeBasisInnovation){
    nu = t(BtY) - BetaPsit; sigma_nu = sd(nu)
    theta = BetaPsit + nu; Btheta = tcrossprod(theta,splineInfo$Bmat)
  } else sigma_nu = 0

  # Initialize the (time-dependent) observation error SD:
  if(use_obs_SV){
    svParams = initCommonSV(Y - Btheta)
    sigma_et = svParams$sigma_et
  } else {
    sigma_e = sd(Y - Btheta, na.rm=TRUE)
    sigma_et = rep(sigma_e, T)
  }

  # Initialize the FLC smoothing parameters (conditional MLE):
  tau_f_k = apply(Psi, 2, function(x) (ncol(splineInfo$Bmat) - (d+1))/crossprod(x, splineInfo$Omega)%*%x)
  #----------------------------------------------------------------------------
  # Predictors:
  if(!is.null(X)){
    # Assuming we have some predictors:
    X = as.matrix(X)

    # Remove any predictors which are constants/intercepts:
    const.pred = apply(X, 2, function(x) all(diff(x) == 0))
    if(any(const.pred)) X = as.matrix(X[,!const.pred])

    # Center and scale the (non-constant) predictors:
    # Note: may not be appropriate for intervention effects!
    #X = scale(X)
  }
  # Include an intercept:
  X = cbind(rep(1, T), X); #colnames(X)[1] = paste(intercept_model, "-Intercept", sep='')

  # Number of predictors:
  p = ncol(X)

  # Initialize the SSModel:
  X.arr = array(t(X), c(1, p, T))
  kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,1]), Zt = X.arr)

  # Identify the dynamic/non-dynamic components:
  if(p > 1 && !use_dynamic_reg) diag(kfas_model$R[,,1])[-1] = 0

  # Forecasting setup and checks:
  if(!is.na(match('Yfore', mcmc_params))){
    forecasting = TRUE # useful

    # Check the forecasting design points:
    if(length(X_Tp1) != p)
      stop("Dimension of predictor X_Tp1 for forecasting must align with alpha;
           try including/excluding an intercept or omit 'Yfore' from the mcmc_params list")
    X_Tp1 = matrix(X_Tp1, ncol = p)

    # Storage for the forecast estimate and distribution:
    alpha_fore_hat = alpha_fore = matrix(0, nrow = p, ncol = K)
  } else forecasting = FALSE
  #----------------------------------------------------------------------------
  # Overall mean term (and T x K case)
  mu_k = as.matrix(colMeans(Beta)); mu_tk = matrix(rep(mu_k, each =  T), nrow = T)

  # Variance term for mu_k:
  a1_mu = 2; a2_mu = 3
  delta_mu_k = sampleMGP(matrix(mu_k, ncol = K), rep(1,K), a1 = a1_mu, a2 = a2_mu)
  sigma_mu_k = 1/sqrt(cumprod(delta_mu_k))
  #----------------------------------------------------------------------------
  # Evolution matrix (must be identity for static components)
  G_alpha = diag(p); G_alpha[1,1] = 0
  #----------------------------------------------------------------------------
  # Initialize the regression terms:
  alpha.arr = array(0, c(T, p, K))
  for(k in 1:K){

    # Update the SSModel object given the new parameters
    kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,k] - mu_k[k]),
                                   Zt = X.arr,
                                   Gt = G_alpha,
                                   kfas_model = kfas_model)
    # Run the sampler
    alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

    # Conditional mean from regression equation:
    Beta[,k] = mu_k[k] + rowSums(X*alpha.arr[,,k])
  }

  #----------------------------------------------------------------------------
  # Evolution error variance:
  Wt = array(diag(p), c(p, p, T)); W0 = diag(10^-4, p);

  # Intercept (or error) components:
  # use eta_tk for consistency w/ other functions
  eta_tk = matrix(alpha.arr[,1,], nrow = T)

  # Initialize the corresponding prior variance term(s):
  xi_eta_tk = 1/eta_tk^2; # Precision scale
  nu = 3  # degrees of freedom

  # MGP term:
  a1_eta = 2; a2_eta = 3;
  delta_eta_k = rep(1,K); sigma_delta_k = 1/sqrt(cumprod(delta_eta_k))

  # Update the error SD for gamma:
  sigma_eta_tk = rep(sigma_delta_k, each = T)/sqrt(xi_eta_tk)
  #----------------------------------------------------------------------------
  # Non-intercept term:
  if(p > 1){

    if(use_dynamic_reg){

      # Dynamic setting:
      alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

      # Initial variance:
      sigma_alpha_0k = abs(alpha_reg[1,])

      # Innovation:
      omega = diff(alpha_reg)

      # Initialize the shrinkage terms:
      sigma_omega_tpk = abs(omega);
      xi_omega_tpk = matrix(1, nrow = T-1, ncol = K*(p-1)) # PX term

      # predictor p, factor k
      lambda_omega_pk = colMeans(sigma_omega_tpk)
      xi_omega_pk = rep(1, (p-1)*K) # PX term

      # predictor p:
      lambda_omega_p = rowMeans(matrix(lambda_omega_pk, nrow = p-1))
      xi_omega_p = rep(1, (p-1)) # PX term

      # global:
      lambda_omega_0 = mean(lambda_omega_p)
      xi_omega_0 = 1 # PX term

    } else{

      # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
      omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

      # factor k, predictor p:
      sigma_omega_kp = abs(omega)
      xi_omega_kp = matrix(1, nrow = K, ncol = p-1) # PX term

      # predictor p:
      lambda_omega_p = colMeans(sigma_omega_kp)
      xi_omega_p = rep(1, (p-1)) # PX term

      # global:
      lambda_omega_0 = mean(lambda_omega_p)
      xi_omega_0 = 1 # PX term
    }
  }
  #----------------------------------------------------------------------------
  # Store the MCMC output in separate arrays (better computation times)
  mcmc_output = vector('list', length(mcmc_params)); names(mcmc_output) = mcmc_params
  if(!is.na(match('beta', mcmc_params))) post.beta = array(NA, c(nsave, T, K))
  if(!is.na(match('fk', mcmc_params))) post.fk = array(NA, c(nsave, m, K))
  if(!is.na(match('alpha', mcmc_params))) post.alpha = array(NA, c(nsave, T, p, K))
  if(!is.na(match('mu_k', mcmc_params))) post.mu_k = array(NA, c(nsave, K))
  if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et = array(NA, c(nsave, T))
  if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat = array(NA, c(nsave, T, m))
  if(!is.na(match('Ypred', mcmc_params))) post.Ypred = array(NA, c(nsave, T, m))
  if(forecasting) {post.Yfore = post.Yfore_hat = array(NA, c(nsave, m))}
  if(computeDIC) post_loglike = numeric(nsave)

  # Total number of MCMC simulations:
  nstot = nburn+(nskip+1)*(nsave)
  skipcount = 0; isave = 0 # For counting

  # Run the MCMC:
  timer0 = proc.time()[3] # For timing the sampler
  for(nsi in 1:nstot){

    #----------------------------------------------------------------------------
    # Step 1: Impute the data, Y:
    #----------------------------------------------------------------------------
    if(any.missing){
      Y[na.ind] = Btheta[na.ind] + sigma_et[na.ind[,1]]*rnorm(nrow(na.ind))
      BtY = tcrossprod(t(splineInfo$Bmat), Y)
    }
    #----------------------------------------------------------------------------
    # Step 2: Sample the FLCs
    #----------------------------------------------------------------------------
    # Sample the FLCs
    Psi = fdlm_flc(BtY = BtY,
                   Beta  = Beta,
                   Psi = Psi,
                   BtB = splineInfo$BtB, #diag(nrow(BtY)),
                   Omega = splineInfo$Omega,
                   lambda = tau_f_k,
                   sigmat2 = sigma_et^2 + sigma_nu^2)
    # And update the loading curves:
    Fmat = splineInfo$Bmat%*%Psi;

    # Sample the smoothing parameters:
    tau_f_k = sample_lambda(tau_f_k, Psi, Omega = splineInfo$Omega, d = d, uniformPrior = TRUE, orderLambdas = FALSE)
    #----------------------------------------------------------------------------
    # Step 3: Sample the regression coefficients (and therefore the factors)
    #----------------------------------------------------------------------------
    # Pseudo-response and pseudo-variance depend on basis innovation:
    if(includeBasisInnovation){
      Y_tilde =  tcrossprod(theta, t(Psi)); sigma_tilde = sigma_nu
    } else {
      Y_tilde = crossprod(BtY, Psi); sigma_tilde = sigma_et
    }

    # Loop over each factor k = 1,...,K:
    for(k in 1:K){

      # Update the variances here:

      # Intercept/gamma:
      Wt[1,1,-T] = sigma_eta_tk[-1,k]^2; W0[1,1] = as.numeric(sigma_eta_tk[1,k]^2)

      # Regression:
      if(p > 1){
        if(use_dynamic_reg){
          if(p == 2){
            # Special case: one predictor (plus intercept)
            Wt[2,2,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,1,k]
            W0[2,2] = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          } else {
            # Usual case: more than one predictor (plus intercept)
            for(j in 1:(p-1)) Wt[-1, -1,][j,j,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,j,k]
            diag(W0[-1, -1]) = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          }
        } else  W0[-1, -1] = diag(as.numeric(sigma_omega_kp[k,]^2), p - 1)
      }

      # Sanity check for Wt: if variances too large, KFAS will stop running
      Wt[which(Wt > 10^6, arr.ind = TRUE)] = 10^6; W0[which(W0 > 10^6, arr.ind = TRUE)] = 10^6

      # Update the SSModel object given the new parameters
      kfas_model = update_kfas_model(Y.dlm = as.matrix(Y_tilde[,k] - mu_k[k]),
                                     Zt = X.arr,
                                     sigma_et = sigma_tilde,
                                     Gt = G_alpha,
                                     Wt = Wt,
                                     W0 = W0,
                                     kfas_model = kfas_model)
      # Run the sampler
      alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

      # Conditional mean from regression equation:
      Beta[,k] = mu_k[k] + rowSums(X*alpha.arr[,,k])

      # Sample from the forecasting distribution, if desired:
      if(forecasting && (nsi > nburn)){ # Only need to compute after burnin
        # Evolution matrix: assume diagonal
        evol_diag = diag(as.matrix(kfas_model$T[,,T-1]))

        # Evolution error SD: assume diagonal
        evol_sd_diag = sqrt(diag(as.matrix(kfas_model$R[,,1]))*diag(as.matrix(kfas_model$Q[,,T-1])))

        # Point forecasting estimate:
        alpha_fore_hat[,k] = evol_diag*alpha.arr[T,,k]

        # Sample from forecasting distribution:
        alpha_fore[,k] = alpha_fore_hat[,k] + rnorm(n = p, mean = 0, sd = evol_sd_diag)
      }
    }

    # Update the forecasting terms:
    if(forecasting){
      # Factors:
      Beta_fore_hat = mu_k + matrix(X_Tp1%*%alpha_fore_hat)
      Beta_fore = mu_k + matrix(X_Tp1%*%alpha_fore)

      # Curves:
      Yfore_hat = Fmat%*%Beta_fore_hat
      Yfore = Fmat%*%Beta_fore + sigma_et[T]*rnorm(n = m)
    }

    # Store this term:
    BetaPsit = tcrossprod(Beta,Psi)
    #----------------------------------------------------------------------------
    # Step 4: Sample the basis terms (if desired)
    #----------------------------------------------------------------------------
    if(includeBasisInnovation){

      # Quad/linear construction a little faster w/o observation SV, but both work:
      if(use_obs_SV){
        Sigma_prec = matrix(rep(sigma_et^-2, ncol(theta)), nrow = T)
        chQtheta = sqrt(Sigma_prec + sigma_nu^-2) # Chol of diag (quadratic term) is just sqrt
        linTheta = t(BtY)*Sigma_prec + BetaPsit/sigma_nu^2 # Linear term from the posterior
      } else {
        chQtheta = sqrt(sigma_e^-2 + sigma_nu^-2) # Chol of diag (quadratic term) is just sqrt
        linTheta = t(BtY)/sigma_e^2 + BetaPsit/sigma_nu^2 # Linear term from the posterior
      }
      theta = linTheta/chQtheta^2 + 1/chQtheta*rnorm(length(theta))
      Btheta = tcrossprod(theta,splineInfo$Bmat)

      sigma_nu = 1/sqrt(truncdist::rtrunc(1, "gamma",
                                          a = 10^-8, b = Inf,
                                          shape = (length(theta)+1)/2,
                                          rate = 1/2*sum((theta - BetaPsit)^2)))
    } else {theta = BetaPsit; sigma_nu = 0; Btheta = tcrossprod(theta,splineInfo$Bmat)}
    #----------------------------------------------------------------------------
    # Step 5: Sample the observation error variance
    #----------------------------------------------------------------------------
    if(use_obs_SV){
      svParams = sampleCommonSV(Y - Btheta, svParams)
      sigma_et = svParams$sigma_et
    } else {
      # Or use uniform prior?
      sigma_e = 1/sqrt(rgamma(n = 1, shape = sum(!is.na(Y))/2, rate = sum((Y - Btheta)^2, na.rm=TRUE)/2))
      sigma_et = rep(sigma_e, T)
    }
    #----------------------------------------------------------------------------
    # Step 6: Sample the intercept/gamma parameters (Note: could use ASIS)
    #----------------------------------------------------------------------------
    # Centerend and non-centered:
    gamma_tk =  matrix(alpha.arr[,1,], nrow = T)

    gamma_tk_c = gamma_tk + mu_tk

    # Sample the unconditional mean term:
    postSD = 1/sqrt(colSums(1/sigma_eta_tk^2) + 1/sigma_mu_k^2)
    postMean = (colSums(gamma_tk_c/sigma_eta_tk^2))*postSD^2
    mu_k = rnorm(n = K, mean = postMean, sd = postSD)
    mu_tk = matrix(rep(mu_k, each =  T), nrow = T)

    # And update the non-centered parameter:
    gamma_tk = gamma_tk_c - mu_tk

    # This is the parameter we use (for consistency)
    eta_tk = gamma_tk

    # Prior variance: MGP
    # Mean Part
    delta_mu_k =  sampleMGP(theta.jh = matrix(mu_k, ncol = K),
                            delta.h = delta_mu_k,
                            a1 = a1_mu, a2 = a2_mu)
    sigma_mu_k = 1/sqrt(cumprod(delta_mu_k))
    # And hyperparameters:
    if(sample_a1a2){
      a1_mu = uni.slice(a1_mu, g = function(a){
        dgamma(delta_mu_k[1], shape = a, rate = 1, log = TRUE) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)}, lower = 0, upper = Inf)
      a2_mu = uni.slice(a2_mu,g = function(a){
        sum(dgamma(delta_mu_k[-1], shape = a, rate = 1, log = TRUE)) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)},lower = 0, upper = Inf)
    }
    # Variance part:
    # Standardize, then reconstruct as matrix of size T x K:
    delta_eta_k = sampleMGP(theta.jh = matrix(eta_tk*sqrt(xi_eta_tk), ncol = K),
                            delta.h = delta_eta_k,
                            a1 = a1_eta, a2 = a2_eta)
    sigma_delta_k = 1/sqrt(cumprod(delta_eta_k))
    # And hyperparameters:
    if(sample_a1a2){
      a1_eta = uni.slice(a1_eta, g = function(a){
        dgamma(delta_eta_k[1], shape = a, rate = 1, log = TRUE) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)}, lower = 0, upper = Inf)
      a2_eta = uni.slice(a2_eta, g = function(a){
        sum(dgamma(delta_eta_k[-1], shape = a, rate = 1, log = TRUE)) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)},lower = 0, upper = Inf)
    }
    # Sample the corresponding prior variance term(s):
    xi_eta_tk = matrix(rgamma(n = T*K,
                              shape = nu/2 + 1/2,
                              rate = nu/2 + (eta_tk/rep(sigma_delta_k, each = T))^2/2), nrow = T)
    # Sample degrees of freedom?
    if(sample_nu){
      nu = uni.slice(nu, g = function(nu){
        sum(dgamma(xi_eta_tk, shape = nu/2, rate = nu/2, log = TRUE)) +
          dunif(nu, min = 2, max = 128, log = TRUE)}, lower = 2, upper = 128)
    }

    # Update the error SD for gamma:
    sigma_eta_tk = rep(sigma_delta_k, each = T)/sqrt(xi_eta_tk)

    # Cap at machine epsilon:
    sigma_eta_tk[which(sigma_eta_tk < sqrt(.Machine$double.eps), arr.ind = TRUE)] = sqrt(.Machine$double.eps)
    #----------------------------------------------------------------------------
    # Step 7: Sample the non-intercept parameters:
    #----------------------------------------------------------------------------
    # Non-intercept term:
    if(p > 1){

      if(use_dynamic_reg){

        # Dynamic setting

        # Regression (non-intercept) coefficients
        alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

        # Initial variance:
        sigma_alpha_0k = 1/sqrt(rgamma(n = K*(p-1),
                                       shape = 3/2 + 1/2,
                                       rate = 3/2 + alpha_reg[1,]^2/2))

        # Random walk, so compute difference for innovations:
        omega = diff(alpha_reg)

        #----------------------------------------------------------------------------
        # tpk-specicif terms:
        omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
        sigma_omega_tpk = matrix(1/sqrt(rgamma(n = (T-1)*K*(p-1),
                                               shape = 1/2 + 1/2,
                                               rate = xi_omega_tpk + omega2/2)), nrow = T-1)
        xi_omega_tpk = matrix(rgamma(n = (T-1)*K*(p-1),
                                     shape = 1/2 + 1/2,
                                     rate = rep(1/lambda_omega_pk^2, each = T-1) + 1/sigma_omega_tpk^2), nrow = T-1)
        #----------------------------------------------------------------------------
        # predictor p, factor k
        lambda_omega_pk = 1/sqrt(rgamma(n = (p-1)*K,
                                        shape = 1/2 + (T-1)/2,
                                        rate = xi_omega_pk + colSums(xi_omega_tpk)))
        xi_omega_pk = rgamma(n = (p-1)*K,
                             shape = 1/2 + 1/2,
                             rate = rep(1/lambda_omega_p^2, times = K) + 1/lambda_omega_pk^2)
        #----------------------------------------------------------------------------
        # predictor p:
        lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                       shape = 1/2 + K/2,
                                       rate = xi_omega_p + rowSums(matrix(xi_omega_pk, nrow = p-1))))
        xi_omega_p = rgamma(n = p-1,
                            shape = 1/2 + 1/2,
                            rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
        #----------------------------------------------------------------------------
        # global:
        lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                       shape = 1/2 + (p-1)/2,
                                       rate = xi_omega_0 + sum(xi_omega_p)))
        xi_omega_0 = rgamma(n = 1,
                            shape = 1/2 + 1/2,
                            rate = (T-1) + 1/lambda_omega_0^2)
        #rate = 1 + 1/lambda_omega_0^2)
      } else{

        # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
        omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

        #----------------------------------------------------------------------------
        # factor k, predictor p:
        omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
        sigma_omega_kp = matrix(1/sqrt(rgamma(n = K*(p-1),
                                              shape = 1/2 + 1/2,
                                              rate = xi_omega_kp + omega2/2)), nrow = K)
        xi_omega_kp = matrix(rgamma(n = K*(p-1),
                                    shape = 1/2 + 1/2,
                                    rate = rep(1/lambda_omega_p^2, each = K) + 1/sigma_omega_kp^2), nrow = K)
        #----------------------------------------------------------------------------
        # predictor p:
        lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                       shape = 1/2 + K/2,
                                       rate = xi_omega_p + colSums(xi_omega_kp)))
        xi_omega_p = rgamma(n = p-1,
                            shape = 1/2 + 1/2,
                            rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
        #----------------------------------------------------------------------------
        # global:
        lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                       shape = 1/2 + (p-1)/2,
                                       rate = xi_omega_0 + sum(xi_omega_p)))
        xi_omega_0 = rgamma(n = 1,
                            shape = 1/2 + 1/2,
                            rate = 1 + 1/lambda_omega_0^2)
      }
    }
    #----------------------------------------------------------------------------
    # Step 10: Adjust the ordering
    #----------------------------------------------------------------------------
    #if(nsi == 10 && K > 1){adjOrder = order(tau_f_k, decreasing = TRUE); tau_f_k = tau_f_k[adjOrder]; Psi = Psi[,adjOrder]; Beta = as.matrix(Beta[,adjOrder])}

    # Store the MCMC output:
    if(nsi > nburn){
      # Increment the skip counter:
      skipcount = skipcount + 1

      # Save the iteration:
      if(skipcount > nskip){
        # Increment the save index
        isave = isave + 1

        # Save the MCMC samples:
        if(!is.na(match('beta', mcmc_params))) post.beta[isave,,] = Beta
        if(!is.na(match('fk', mcmc_params))) post.fk[isave,,] = Fmat
        if(!is.na(match('alpha', mcmc_params))) post.alpha[isave,,,] = alpha.arr
        if(!is.na(match('mu_k', mcmc_params))) post.mu_k[isave,] = mu_k
        if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et[isave,] = sigma_et
        if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat[isave,,] = Btheta
        if(!is.na(match('Ypred', mcmc_params))) post.Ypred[isave,,] = rnorm(n = T*m, mean = matrix(Btheta), sd = rep(sigma_et,m))
        if(forecasting) {post.Yfore[isave,] = Yfore; post.Yfore_hat[isave,] = Yfore_hat}
        if(computeDIC) post_loglike[isave] = sum(dnorm(matrix(Yna), mean = matrix(Btheta), sd = rep(sigma_et,m), log = TRUE), na.rm = TRUE)

        # And reset the skip counter:
        skipcount = 0
      }
    }
    computeTimeRemaining(nsi, timer0, nstot, nrep = 500)
  }

  # Store the results (and correct for rescaling by sdY):
  if(!is.na(match('beta', mcmc_params))) mcmc_output$beta = post.beta*sdY
  if(!is.na(match('fk', mcmc_params))) mcmc_output$fk = post.fk
  if(!is.na(match('alpha', mcmc_params))) mcmc_output$alpha = post.alpha*sdY
  if(!is.na(match('mu_k', mcmc_params))) mcmc_output$mu_k = post.mu_k*sdY
  if(!is.na(match('sigma_et', mcmc_params))) mcmc_output$sigma_et = post.sigma_et*sdY
  if(!is.na(match('Yhat', mcmc_params))) mcmc_output$Yhat = post.Yhat*sdY
  if(!is.na(match('Ypred', mcmc_params))) mcmc_output$Ypred = post.Ypred*sdY
  if(forecasting) {mcmc_output$Yfore = post.Yfore*sdY; mcmc_output$Yfore_hat = post.Yfore_hat*sdY}

  if(computeDIC){
    # Log-likelihood evaluated at posterior means:
    loglike_hat = sum(dnorm(matrix(Yna),
                            mean = matrix(colMeans(post.Yhat)),
                            sd = rep(colMeans(post.sigma_et), m),
                            log = TRUE), na.rm=TRUE)

    # Effective number of parameters (Note: two options)
    p_d = c(2*(loglike_hat - mean(post_loglike)),
            2*var(post_loglike))
    # DIC:
    DIC = -2*loglike_hat + 2*p_d

    # Store the DIC and the effective number of parameters (p_d)
    mcmc_output$DIC = DIC; mcmc_output$p_d = p_d
  }

  print(paste('Total time: ', round((proc.time()[3] - timer0)/60), 'minutes'))

  return (mcmc_output);
}
#' MCMC Sampling Algorithm for the Dynamic Function-on-Scalars Regression Model with Random Walk Errors
#'
#' Runs the MCMC for the dynamic function-on-scalars regression model based on
#' an reduced-rank expansion. Here, we assume the factor regression has RW errors.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param X the \code{T x p} matrix of predictors; if NULL, only include an intercept
#' @param K the number of factors; if NULL, use SVD-based proportion of variability explained
#' @param use_dynamic_reg logical; if TRUE, regression coefficients are dynamic
#' (with random walk models), otherwise independent
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (dynamic factors)
#' \item "fk" (loading curves)
#' \item "alpha" (regression coefficients; possibly dynamic)
#' \item "mu_k" (intercept term for factor k)
#' \item "sigma_et" (observation error SD; possibly dynamic)
#' \item "Yhat" (fitted values)
#' \item "Ypred" (posterior predictive values)
#' \item "Yfore" (one-step forecast; includes the estimate and the distribution)
#' }
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1}
#' @param use_obs_SV logical; when TRUE, include a stochastic volatility model
#' for the observation error variance
#' @param includeBasisInnovation logical; when TRUE, include an iid basis coefficient term for residual correlation
#' (i.e., the idiosyncratic error term for a factor model on the full basis matrix)
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#'
#' @import  KFAS truncdist
dfosr_rw = function(Y, tau, X = NULL, K = NULL,
                    use_dynamic_reg = TRUE,
                    nsave = 1000, nburn = 1000, nskip = 3,
                    mcmc_params = list("beta", "fk", "alpha"),
                    X_Tp1 = 1,
                    use_obs_SV = FALSE,
                    includeBasisInnovation = FALSE,
                    computeDIC = TRUE){
  # Some options (for now):
  sample_nu = TRUE # Sample DF parameter, or fix at nu=3?
  sample_a1a2 = TRUE # Sample a1, a2, or fix at a1=2, a2=3?

  #----------------------------------------------------------------------------
  # Assume that we've done checks elsewhere
  #----------------------------------------------------------------------------
  # Convert tau to matrix, if necessary:
  tau = as.matrix(tau)

  # Compute the dimensions:
  T = nrow(Y); m = ncol(Y); d = ncol(tau)

  # Rescale observation points to [0,1]
  tau01 = apply(tau, 2, function(x) (x - min(x))/(max(x) - min(x)))

  # Rescale by observation SD (and correct parameters later):
  sdY = sd(Y, na.rm=TRUE);
  Y = Y/sdY;
  #----------------------------------------------------------------------------
  # Initialize the main terms:

  # Initialize the FLC coefficients and factors:
  inits = fdlm_init(Y, tau, K); Beta = inits$Beta; Psi = inits$Psi; splineInfo = inits$splineInfo
  K = ncol(Beta) # to be sure we have the right value

  # Also use the imputed data values here for initialization:
  Yna = Y # The original data, including NAs
  any.missing = any(is.na(Yna)) # Any missing obs?
  if(any.missing){na.ind = which(is.na(Yna), arr.ind = TRUE); Y = inits$Y0}
  BtY = tcrossprod(t(splineInfo$Bmat), Y)

  # FLC matrix:
  Fmat = splineInfo$Bmat%*%Psi

  # Initialize the conditional expectation:
  BetaPsit = tcrossprod(Beta, Psi); Btheta = tcrossprod(BetaPsit, splineInfo$Bmat)

  # Initialize the basis coefficient residuals and the corresponding standard deviation
  if(includeBasisInnovation){
    nu = t(BtY) - BetaPsit; sigma_nu = sd(nu)
    theta = BetaPsit + nu; Btheta = tcrossprod(theta,splineInfo$Bmat)
  } else sigma_nu = 0

  # Initialize the (time-dependent) observation error SD:
  if(use_obs_SV){
    svParams = initCommonSV(Y - Btheta)
    sigma_et = svParams$sigma_et
  } else {
    sigma_e = sd(Y - Btheta, na.rm=TRUE)
    sigma_et = rep(sigma_e, T)
  }

  # Initialize the FLC smoothing parameters (conditional MLE):
  tau_f_k = apply(Psi, 2, function(x) (ncol(splineInfo$Bmat) - (d+1))/crossprod(x, splineInfo$Omega)%*%x)
  #----------------------------------------------------------------------------
  # Predictors:
  if(!is.null(X)){
    # Assuming we have some predictors:
    X = as.matrix(X)

    # Remove any predictors which are constants/intercepts:
    const.pred = apply(X, 2, function(x) all(diff(x) == 0))
    if(any(const.pred)) X = as.matrix(X[,!const.pred])

    # Center and scale the (non-constant) predictors:
    # Note: may not be appropriate for intervention effects!
    #X = scale(X)
  }
  # Include an intercept:
  X = cbind(rep(1, T), X); #colnames(X)[1] = paste(intercept_model, "-Intercept", sep='')

  # Number of predictors:
  p = ncol(X)

  # Initialize the SSModel:
  X.arr = array(t(X), c(1, p, T))
  kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,1]), Zt = X.arr)

  # Identify the dynamic/non-dynamic components:
  if(p > 1 && !use_dynamic_reg) diag(kfas_model$R[,,1])[-1] = 0

  # Forecasting setup and checks:
  if(!is.na(match('Yfore', mcmc_params))){
    forecasting = TRUE # useful

    # Check the forecasting design points:
    if(length(X_Tp1) != p)
      stop("Dimension of predictor X_Tp1 for forecasting must align with alpha;
           try including/excluding an intercept or omit 'Yfore' from the mcmc_params list")
    X_Tp1 = matrix(X_Tp1, ncol = p)

    # Storage for the forecast estimate and distribution:
    alpha_fore_hat = alpha_fore = matrix(0, nrow = p, ncol = K)
  } else forecasting = FALSE
  #----------------------------------------------------------------------------
  # Evolution matrix (must be identity for static components)
  G_alpha = diag(p) # Replace the intercept terms as needed
  #----------------------------------------------------------------------------
  # Initialize the regression terms:
  alpha.arr = array(0, c(T, p, K))
  for(k in 1:K){

    # Update the SSModel object given the new parameters
    kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,k]),
                                   Zt = X.arr,
                                   Gt = G_alpha,
                                   kfas_model = kfas_model)
    # Run the sampler
    alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

    # Conditional mean from regression equation:
    Beta[,k] = rowSums(X*alpha.arr[,,k])
  }

  #----------------------------------------------------------------------------
  # Evolution error variance:
  Wt = array(diag(p), c(p, p, T)); W0 = diag(10^-4, p);

  # Intercept (or gamma) components:
  gamma_tk =  matrix(alpha.arr[,1,], nrow = T)

  # Difference:
  eta_tk = diff(gamma_tk)

  # Initialize the corresponding prior variance term(s):
  xi_eta_tk = 1/eta_tk^2; # Precision scale
  nu = 3  # degrees of freedom

  # Initial variance term:
  sigma_eta_0k = abs(gamma_tk[1,])

  # MGP term:
  a1_eta = 2; a2_eta = 3;
  delta_eta_k = rep(1,K); sigma_delta_k = 1/sqrt(cumprod(delta_eta_k))

  # Update the error SD for gamma:
  sigma_eta_tk = rep(sigma_delta_k, each = T-1)/sqrt(xi_eta_tk)
  #----------------------------------------------------------------------------
  # Non-intercept term:
  if(p > 1){

    if(use_dynamic_reg){

      # Dynamic setting:
      alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

      # Initial variance:
      sigma_alpha_0k = abs(alpha_reg[1,])

      # Innovation:
      omega = diff(alpha_reg)

      # Initialize the shrinkage terms:
      sigma_omega_tpk = abs(omega);
      xi_omega_tpk = matrix(1, nrow = T-1, ncol = K*(p-1)) # PX term

      # predictor p, factor k
      lambda_omega_pk = colMeans(sigma_omega_tpk)
      xi_omega_pk = rep(1, (p-1)*K) # PX term

      # predictor p:
      lambda_omega_p = rowMeans(matrix(lambda_omega_pk, nrow = p-1))
      xi_omega_p = rep(1, (p-1)) # PX term

      # global:
      lambda_omega_0 = mean(lambda_omega_p)
      xi_omega_0 = 1 # PX term

    } else{

      # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
      omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

      # factor k, predictor p:
      sigma_omega_kp = abs(omega)
      xi_omega_kp = matrix(1, nrow = K, ncol = p-1) # PX term

      # predictor p:
      lambda_omega_p = colMeans(sigma_omega_kp)
      xi_omega_p = rep(1, (p-1)) # PX term

      # global:
      lambda_omega_0 = mean(lambda_omega_p)
      xi_omega_0 = 1 # PX term
    }
  }
  #----------------------------------------------------------------------------
  # Store the MCMC output in separate arrays (better computation times)
  mcmc_output = vector('list', length(mcmc_params)); names(mcmc_output) = mcmc_params
  if(!is.na(match('beta', mcmc_params))) post.beta = array(NA, c(nsave, T, K))
  if(!is.na(match('fk', mcmc_params))) post.fk = array(NA, c(nsave, m, K))
  if(!is.na(match('alpha', mcmc_params))) post.alpha = array(NA, c(nsave, T, p, K))
  if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et = array(NA, c(nsave, T))
  if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat = array(NA, c(nsave, T, m))
  if(!is.na(match('Ypred', mcmc_params))) post.Ypred = array(NA, c(nsave, T, m))
  if(forecasting) {post.Yfore = post.Yfore_hat = array(NA, c(nsave, m))}
  if(computeDIC) post_loglike = numeric(nsave)

  # Total number of MCMC simulations:
  nstot = nburn+(nskip+1)*(nsave)
  skipcount = 0; isave = 0 # For counting

  # Run the MCMC:
  timer0 = proc.time()[3] # For timing the sampler
  for(nsi in 1:nstot){

    #----------------------------------------------------------------------------
    # Step 1: Impute the data, Y:
    #----------------------------------------------------------------------------
    if(any.missing){
      Y[na.ind] = Btheta[na.ind] + sigma_et[na.ind[,1]]*rnorm(nrow(na.ind))
      BtY = tcrossprod(t(splineInfo$Bmat), Y)
    }
    #----------------------------------------------------------------------------
    # Step 2: Sample the FLCs
    #----------------------------------------------------------------------------
    # Sample the FLCs
    Psi = fdlm_flc(BtY = BtY,
                   Beta  = Beta,
                   Psi = Psi,
                   BtB = splineInfo$BtB, #diag(nrow(BtY)),
                   Omega = splineInfo$Omega,
                   lambda = tau_f_k,
                   sigmat2 = sigma_et^2 + sigma_nu^2)
    # And update the loading curves:
    Fmat = splineInfo$Bmat%*%Psi;

    # Sample the smoothing parameters:
    tau_f_k = sample_lambda(tau_f_k, Psi, Omega = splineInfo$Omega, d = d, uniformPrior = TRUE, orderLambdas = FALSE)
    #----------------------------------------------------------------------------
    # Step 3: Sample the regression coefficients (and therefore the factors)
    #----------------------------------------------------------------------------
    # Pseudo-response and pseudo-variance depend on basis innovation:
    if(includeBasisInnovation){
      Y_tilde =  tcrossprod(theta, t(Psi)); sigma_tilde = sigma_nu
    } else {
      Y_tilde = crossprod(BtY, Psi); sigma_tilde = sigma_et
    }

    # Loop over each factor k = 1,...,K:
    for(k in 1:K){

      # Update the variances here:

      # Intercept/gamma:
      Wt[1,1,-T] = sigma_eta_tk[,k]^2; W0[1,1] = sigma_eta_0k[k]^2

      # Regression:
      if(p > 1){
        if(use_dynamic_reg){
          if(p == 2){
            # Special case: one predictor (plus intercept)
            Wt[2,2,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,1,k]
            W0[2,2] = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          } else {
            # Usual case: more than one predictor (plus intercept)
            for(j in 1:(p-1)) Wt[-1, -1,][j,j,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,j,k]
            diag(W0[-1, -1]) = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          }
        } else  W0[-1, -1] = diag(as.numeric(sigma_omega_kp[k,]^2), p - 1)
      }

      # Sanity check for Wt: if variances too large, KFAS will stop running
      Wt[which(Wt > 10^6, arr.ind = TRUE)] = 10^6; W0[which(W0 > 10^6, arr.ind = TRUE)] = 10^6

      # Update the SSModel object given the new parameters
      kfas_model = update_kfas_model(Y.dlm = as.matrix(Y_tilde[,k]),
                                     Zt = X.arr,
                                     sigma_et = sigma_tilde,
                                     Gt = G_alpha,
                                     Wt = Wt,
                                     W0 = W0,
                                     kfas_model = kfas_model)
      # Run the sampler
      alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

      # Conditional mean from regression equation:
      Beta[,k] = rowSums(X*alpha.arr[,,k])

      # Sample from the forecasting distribution, if desired:
      if(forecasting && (nsi > nburn)){ # Only need to compute after burnin
        # Evolution matrix: assume diagonal
        evol_diag = diag(as.matrix(kfas_model$T[,,T-1]))

        # Evolution error SD: assume diagonal
        evol_sd_diag = sqrt(diag(as.matrix(kfas_model$R[,,1]))*diag(as.matrix(kfas_model$Q[,,T-1])))

        # Point forecasting estimate:
        alpha_fore_hat[,k] = evol_diag*alpha.arr[T,,k]

        # Sample from forecasting distribution:
        alpha_fore[,k] = alpha_fore_hat[,k] + rnorm(n = p, mean = 0, sd = evol_sd_diag)
      }
    }

    # Update the forecasting terms:
    if(forecasting){
      # Factors:
      Beta_fore_hat = matrix(X_Tp1%*%alpha_fore_hat)
      Beta_fore = matrix(X_Tp1%*%alpha_fore)

      # Curves:
      Yfore_hat = Fmat%*%Beta_fore_hat
      Yfore = Fmat%*%Beta_fore + sigma_et[T]*rnorm(n = m)
    }

    # Store this term:
    BetaPsit = tcrossprod(Beta,Psi)
    #----------------------------------------------------------------------------
    # Step 4: Sample the basis terms (if desired)
    #----------------------------------------------------------------------------
    if(includeBasisInnovation){

      # Quad/linear construction a little faster w/o observation SV, but both work:
      if(use_obs_SV){
        Sigma_prec = matrix(rep(sigma_et^-2, ncol(theta)), nrow = T)
        chQtheta = sqrt(Sigma_prec + sigma_nu^-2) # Chol of diag (quadratic term) is just sqrt
        linTheta = t(BtY)*Sigma_prec + BetaPsit/sigma_nu^2 # Linear term from the posterior
      } else {
        chQtheta = sqrt(sigma_e^-2 + sigma_nu^-2) # Chol of diag (quadratic term) is just sqrt
        linTheta = t(BtY)/sigma_e^2 + BetaPsit/sigma_nu^2 # Linear term from the posterior
      }
      theta = linTheta/chQtheta^2 + 1/chQtheta*rnorm(length(theta))
      Btheta = tcrossprod(theta,splineInfo$Bmat)

      sigma_nu = 1/sqrt(truncdist::rtrunc(1, "gamma",
                                          a = 10^-8, b = Inf,
                                          shape = (length(theta)+1)/2,
                                          rate = 1/2*sum((theta - BetaPsit)^2)))
    } else {theta = BetaPsit; sigma_nu = 0; Btheta = tcrossprod(theta,splineInfo$Bmat)}
    #----------------------------------------------------------------------------
    # Step 5: Sample the observation error variance
    #----------------------------------------------------------------------------
    if(use_obs_SV){
      svParams = sampleCommonSV(Y - Btheta, svParams)
      sigma_et = svParams$sigma_et
    } else {
      # Or use uniform prior?
      sigma_e = 1/sqrt(rgamma(n = 1, shape = sum(!is.na(Y))/2, rate = sum((Y - Btheta)^2, na.rm=TRUE)/2))
      sigma_et = rep(sigma_e, T)
    }
    #----------------------------------------------------------------------------
    # Step 6: Sample the intercept/gamma parameters (Note: could use ASIS)
    #----------------------------------------------------------------------------
    # Centerend and non-centered:
    gamma_tk =  matrix(alpha.arr[,1,], nrow = T)

    # Difference:
    eta_tk = diff(gamma_tk)

    # Variance part:
    # Standardize, then reconstruct as matrix of size T x K:
    delta_eta_k = sampleMGP(theta.jh = matrix(eta_tk*sqrt(xi_eta_tk), ncol = K),
                            delta.h = delta_eta_k,
                            a1 = a1_eta, a2 = a2_eta)
    sigma_delta_k = 1/sqrt(cumprod(delta_eta_k))
    # And hyperparameters:
    if(sample_a1a2){
      a1_eta = uni.slice(a1_eta, g = function(a){
        dgamma(delta_eta_k[1], shape = a, rate = 1, log = TRUE) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)}, lower = 0, upper = Inf)
      a2_eta = uni.slice(a2_eta, g = function(a){
        sum(dgamma(delta_eta_k[-1], shape = a, rate = 1, log = TRUE)) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)},lower = 0, upper = Inf)
    }

    # Sample the corresponding prior variance term(s):
    xi_eta_tk = matrix(rgamma(n = (T-1)*K,
                              shape = nu/2 + 1/2,
                              rate = nu/2 + (eta_tk/rep(sigma_delta_k, each = T-1))^2/2), nrow = T-1)
    # Sample degrees of freedom?
    if(sample_nu){
      nu = uni.slice(nu, g = function(nu){
        sum(dgamma(xi_eta_tk, shape = nu/2, rate = nu/2, log = TRUE)) +
          dunif(nu, min = 2, max = 128, log = TRUE)}, lower = 2, upper = 128)
    }

    # Initial sd:
    sigma_eta_0k = 1/sqrt(rgamma(n = K,
                                 shape = 3/2 + 1/2,
                                 rate = 3/2 + gamma_tk[1,]^2/2))

    # Update the error SD for gamma:
    sigma_eta_tk = rep(sigma_delta_k, each = T-1)/sqrt(xi_eta_tk)

    # Cap at machine epsilon:
    sigma_eta_tk[which(sigma_eta_tk < sqrt(.Machine$double.eps), arr.ind = TRUE)] = sqrt(.Machine$double.eps)
    #----------------------------------------------------------------------------
    # Step 7: Sample the non-intercept parameters:
    #----------------------------------------------------------------------------
    # Non-intercept term:
    if(p > 1){

      if(use_dynamic_reg){

        # Dynamic setting

        # Regression (non-intercept) coefficients
        alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

        # Initial variance:
        sigma_alpha_0k = 1/sqrt(rgamma(n = K*(p-1),
                                       shape = 3/2 + 1/2,
                                       rate = 3/2 + alpha_reg[1,]^2/2))

        # Random walk, so compute difference for innovations:
        omega = diff(alpha_reg)

        #----------------------------------------------------------------------------
        # tpk-specicif terms:
        omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
        sigma_omega_tpk = matrix(1/sqrt(rgamma(n = (T-1)*K*(p-1),
                                               shape = 1/2 + 1/2,
                                               rate = xi_omega_tpk + omega2/2)), nrow = T-1)
        xi_omega_tpk = matrix(rgamma(n = (T-1)*K*(p-1),
                                     shape = 1/2 + 1/2,
                                     rate = rep(1/lambda_omega_pk^2, each = T-1) + 1/sigma_omega_tpk^2), nrow = T-1)
        #----------------------------------------------------------------------------
        # predictor p, factor k
        lambda_omega_pk = 1/sqrt(rgamma(n = (p-1)*K,
                                        shape = 1/2 + (T-1)/2,
                                        rate = xi_omega_pk + colSums(xi_omega_tpk)))
        xi_omega_pk = rgamma(n = (p-1)*K,
                             shape = 1/2 + 1/2,
                             rate = rep(1/lambda_omega_p^2, times = K) + 1/lambda_omega_pk^2)
        #----------------------------------------------------------------------------
        # predictor p:
        lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                       shape = 1/2 + K/2,
                                       rate = xi_omega_p + rowSums(matrix(xi_omega_pk, nrow = p-1))))
        xi_omega_p = rgamma(n = p-1,
                            shape = 1/2 + 1/2,
                            rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
        #----------------------------------------------------------------------------
        # global:
        lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                       shape = 1/2 + (p-1)/2,
                                       rate = xi_omega_0 + sum(xi_omega_p)))
        xi_omega_0 = rgamma(n = 1,
                            shape = 1/2 + 1/2,
                            rate = (T-1) + 1/lambda_omega_0^2)
        #rate = 1 + 1/lambda_omega_0^2)
      } else{

        # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
        omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

        #----------------------------------------------------------------------------
        # factor k, predictor p:
        omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
        sigma_omega_kp = matrix(1/sqrt(rgamma(n = K*(p-1),
                                              shape = 1/2 + 1/2,
                                              rate = xi_omega_kp + omega2/2)), nrow = K)
        xi_omega_kp = matrix(rgamma(n = K*(p-1),
                                    shape = 1/2 + 1/2,
                                    rate = rep(1/lambda_omega_p^2, each = K) + 1/sigma_omega_kp^2), nrow = K)
        #----------------------------------------------------------------------------
        # predictor p:
        lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                       shape = 1/2 + K/2,
                                       rate = xi_omega_p + colSums(xi_omega_kp)))
        xi_omega_p = rgamma(n = p-1,
                            shape = 1/2 + 1/2,
                            rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
        #----------------------------------------------------------------------------
        # global:
        lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                       shape = 1/2 + (p-1)/2,
                                       rate = xi_omega_0 + sum(xi_omega_p)))
        xi_omega_0 = rgamma(n = 1,
                            shape = 1/2 + 1/2,
                            rate = 1 + 1/lambda_omega_0^2)
      }
    }
    #----------------------------------------------------------------------------
    # Step 10: Adjust the ordering
    #----------------------------------------------------------------------------
    #if(nsi == 10 && K > 1){adjOrder = order(tau_f_k, decreasing = TRUE); tau_f_k = tau_f_k[adjOrder]; Psi = Psi[,adjOrder]; Beta = as.matrix(Beta[,adjOrder])}

    # Store the MCMC output:
    if(nsi > nburn){
      # Increment the skip counter:
      skipcount = skipcount + 1

      # Save the iteration:
      if(skipcount > nskip){
        # Increment the save index
        isave = isave + 1

        # Save the MCMC samples:
        if(!is.na(match('beta', mcmc_params))) post.beta[isave,,] = Beta
        if(!is.na(match('fk', mcmc_params))) post.fk[isave,,] = Fmat
        if(!is.na(match('alpha', mcmc_params))) post.alpha[isave,,,] = alpha.arr
        if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et[isave,] = sigma_et
        if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat[isave,,] = Btheta # + sigma_e*rnorm(length(Y))
        if(!is.na(match('Ypred', mcmc_params))) post.Ypred[isave,,] = rnorm(n = T*m, mean = matrix(Btheta), sd = rep(sigma_et,m))
        if(forecasting) {post.Yfore[isave,] = Yfore; post.Yfore_hat[isave,] = Yfore_hat}
        if(computeDIC) post_loglike[isave] = sum(dnorm(matrix(Yna), mean = matrix(Btheta), sd = rep(sigma_et,m), log = TRUE), na.rm = TRUE)

        # And reset the skip counter:
        skipcount = 0
      }
    }
    computeTimeRemaining(nsi, timer0, nstot, nrep = 500)
  }

  # Store the results (and correct for rescaling by sdY):
  if(!is.na(match('beta', mcmc_params))) mcmc_output$beta = post.beta*sdY
  if(!is.na(match('fk', mcmc_params))) mcmc_output$fk = post.fk
  if(!is.na(match('alpha', mcmc_params))) mcmc_output$alpha = post.alpha*sdY
  if(!is.na(match('sigma_et', mcmc_params))) mcmc_output$sigma_et = post.sigma_et*sdY
  if(!is.na(match('Yhat', mcmc_params))) mcmc_output$Yhat = post.Yhat*sdY
  if(!is.na(match('Ypred', mcmc_params))) mcmc_output$Ypred = post.Ypred*sdY
  if(forecasting) {mcmc_output$Yfore = post.Yfore*sdY; mcmc_output$Yfore_hat = post.Yfore_hat*sdY}

  if(computeDIC){
    # Log-likelihood evaluated at posterior means:
    loglike_hat = sum(dnorm(matrix(Yna),
                            mean = matrix(colMeans(post.Yhat)),
                            sd = rep(colMeans(post.sigma_et), m),
                            log = TRUE), na.rm=TRUE)

    # Effective number of parameters (Note: two options)
    p_d = c(2*(loglike_hat - mean(post_loglike)),
            2*var(post_loglike))
    # DIC:
    DIC = -2*loglike_hat + 2*p_d

    # Store the DIC and the effective number of parameters (p_d)
    mcmc_output$DIC = DIC; mcmc_output$p_d = p_d
  }

  print(paste('Total time: ', round((proc.time()[3] - timer0)/60), 'minutes'))

  return (mcmc_output);
}
#' MCMC Sampling Algorithm for the Dynamic Function-on-Scalars Regression Model with AR(1) Errors
#'
#' Runs the MCMC for the dynamic function-on-scalars regression model based on
#' an reduced-rank expansion. Here, we assume the factor regression has AR(1) errors.
#' This particular sampler loops over the k=1,...,K factors, so the sampler is
#' O(T*K*p^3) instead of O(T*(K*p)^3). We assume some default settings in this case.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param X the \code{T x p} matrix of predictors; if NULL, only include an intercept
#' @param K the number of factors; if NULL, use SVD-based proportion of variability explained
#' @param use_dynamic_reg logical; if TRUE, regression coefficients are dynamic
#' (with random walk models), otherwise independent
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (dynamic factors)
#' \item "fk" (loading curves)
#' \item "alpha" (regression coefficients; possibly dynamic)
#' \item "mu_k" (intercept term for factor k)
#' \item "ar_phi" (AR coefficients for each k under AR(1) model)
#' \item "sigma_et" (observation error SD; possibly dynamic)
#' \item "Yhat" (fitted values)
#' \item "Ypred" (posterior predictive values)
#' \item "Yfore" (one-step forecast; includes the estimate and the distribution)
#' }
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1}
#' @param use_obs_SV logical; when TRUE, include a stochastic volatility model
#' for the observation error variance
#' @param includeBasisInnovation logical; when TRUE, include an iid basis coefficient term for residual correlation
#' (i.e., the idiosyncratic error term for a factor model on the full basis matrix)
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#'
#' @import  KFAS truncdist
dfosr_ar = function(Y, tau, X = NULL, K = NULL,
                    use_dynamic_reg = TRUE,
                    nsave = 1000, nburn = 1000, nskip = 2,
                    mcmc_params = list("beta", "fk", "alpha", "mu_k", "ar_phi"),
                    X_Tp1 = 1,
                    use_obs_SV = FALSE,
                    includeBasisInnovation = FALSE,
                    computeDIC = TRUE){
  # Some options (for now):
  sample_nu = TRUE # Sample DF parameter, or fix at nu=3?
  sample_a1a2 = TRUE # Sample a1, a2, or fix at a1=2, a2=3?

  #----------------------------------------------------------------------------
  # Assume that we've done checks elsewhere
  #----------------------------------------------------------------------------
  # Convert tau to matrix, if necessary:
  tau = as.matrix(tau)

  # Compute the dimensions:
  T = nrow(Y); m = ncol(Y); d = ncol(tau)

  # Rescale observation points to [0,1]
  tau01 = apply(tau, 2, function(x) (x - min(x))/(max(x) - min(x)))

  # Rescale by observation SD (and correct parameters later):
  sdY = sd(Y, na.rm=TRUE);
  Y = Y/sdY;
  #----------------------------------------------------------------------------
  # Initialize the main terms:

  # Initialize the FLC coefficients and factors:
  inits = fdlm_init(Y, tau, K); Beta = inits$Beta; Psi = inits$Psi; splineInfo = inits$splineInfo
  K = ncol(Beta) # to be sure we have the right value

  # Also use the imputed data values here for initialization:
  Yna = Y # The original data, including NAs
  any.missing = any(is.na(Yna)) # Any missing obs?
  if(any.missing){na.ind = which(is.na(Yna), arr.ind = TRUE); Y = inits$Y0}
  BtY = tcrossprod(t(splineInfo$Bmat), Y)

  # FLC matrix:
  Fmat = splineInfo$Bmat%*%Psi

  # Initialize the conditional expectation:
  BetaPsit = tcrossprod(Beta, Psi); Btheta = tcrossprod(BetaPsit, splineInfo$Bmat)

  # Initialize the basis coefficient residuals and the corresponding standard deviation
  if(includeBasisInnovation){
    nu = t(BtY) - BetaPsit; sigma_nu = sd(nu)
    theta = BetaPsit + nu; Btheta = tcrossprod(theta,splineInfo$Bmat)
  } else sigma_nu = 0

  # Initialize the (time-dependent) observation error SD:
  if(use_obs_SV){
    svParams = initCommonSV(Y - Btheta)
    sigma_et = svParams$sigma_et
  } else {
    sigma_e = sd(Y - Btheta, na.rm=TRUE)
    sigma_et = rep(sigma_e, T)
  }

  # Initialize the FLC smoothing parameters (conditional MLE):
  tau_f_k = apply(Psi, 2, function(x) (ncol(splineInfo$Bmat) - (d+1))/crossprod(x, splineInfo$Omega)%*%x)
  #----------------------------------------------------------------------------
  # Predictors:
  if(!is.null(X)){
    # Assuming we have some predictors:
    X = as.matrix(X)

    # Remove any predictors which are constants/intercepts:
    const.pred = apply(X, 2, function(x) all(diff(x) == 0))
    if(any(const.pred)) X = as.matrix(X[,!const.pred])

    # Center and scale the (non-constant) predictors:
    # Note: may not be appropriate for intervention effects!
    #X = scale(X)
  }
  # Include an intercept:
  X = cbind(rep(1, T), X); #colnames(X)[1] = paste(intercept_model, "-Intercept", sep='')

  # Number of predictors:
  p = ncol(X)

  # Initialize the SSModel:
  X.arr = array(t(X), c(1, p, T))
  kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,1]), Zt = X.arr)

  # Identify the dynamic/non-dynamic components:
  if(p > 1 && !use_dynamic_reg) diag(kfas_model$R[,,1])[-1] = 0

  # Forecasting setup and checks:
  if(!is.na(match('Yfore', mcmc_params))){
    forecasting = TRUE # useful

    # Check the forecasting design points:
    if(length(X_Tp1) != p)
      stop("Dimension of predictor X_Tp1 for forecasting must align with alpha;
           try including/excluding an intercept or omit 'Yfore' from the mcmc_params list")
    X_Tp1 = matrix(X_Tp1, ncol = p)

    # Storage for the forecast estimate and distribution:
    alpha_fore_hat = alpha_fore = matrix(0, nrow = p, ncol = K)
  } else forecasting = FALSE
  #----------------------------------------------------------------------------
  # Overall mean term (and T x K case)
  mu_k = as.matrix(colMeans(Beta)); mu_tk = matrix(rep(mu_k, each =  T), nrow = T)

  # Variance term for mu_k:
  a1_mu = 2; a2_mu = 3
  delta_mu_k = sampleMGP(matrix(mu_k, ncol = K), rep(1,K), a1 = a1_mu, a2 = a2_mu)
  sigma_mu_k = 1/sqrt(cumprod(delta_mu_k))
  #----------------------------------------------------------------------------
  # AR(1) Evolution Matrix
  G_alpha = diag(p) # Replace the intercept terms as needed

  # AR(1) coefficients:
  ar_int = apply(Beta - mu_tk, 2, function(x) lm(x[-1] ~ - 1 +  x[-length(x)])$coef)

  # Stationarity fix:
  ar_int[which(abs(ar_int) > 0.95)] = 0.8*sign(ar_int[which(abs(ar_int) > 0.95)])

  #----------------------------------------------------------------------------
  # Initialize the regression terms:
  alpha.arr = array(0, c(T, p, K))
  for(k in 1:K){
    # Update the evoluation matrix
    G_alpha[1,1] = ar_int[k]

    # Update the SSModel object given the new parameters
    kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,k] - mu_k[k]),
                                   Zt = X.arr,
                                   Gt = G_alpha,
                                   kfas_model = kfas_model)
    # Run the sampler
    alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

    # Conditional mean from regression equation:
    Beta[,k] = mu_k[k] + rowSums(X*alpha.arr[,,k])
  }

  #----------------------------------------------------------------------------
  # Evolution error variance:
  Wt = array(diag(p), c(p, p, T)); W0 = diag(10^-4, p);

  # Intercept (or gamma) components:
  gamma_tk =  matrix(alpha.arr[,1,], nrow = T)

  # Then subtract the AR(1) part:
  eta_tk = gamma_tk[-1,] -  t(ar_int*t(gamma_tk[-T,]))

  # Initialize the corresponding prior variance term(s):
  xi_eta_tk = 1/eta_tk^2; # Precision scale
  nu = 3  # degrees of freedom

  # Initial variance term:
  sigma_eta_0k = abs(gamma_tk[1,])

  # MGP term:
  a1_eta = 2; a2_eta = 3;
  delta_eta_k = rep(1,K); sigma_delta_k = 1/sqrt(cumprod(delta_eta_k))

  # Update the error SD for gamma:
  sigma_eta_tk = rep(sigma_delta_k, each = T-1)/sqrt(xi_eta_tk)
  #----------------------------------------------------------------------------
  # Non-intercept term:
  if(p > 1){

    if(use_dynamic_reg){

      # Dynamic setting:
      alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

      # Initial variance:
      sigma_alpha_0k = abs(alpha_reg[1,])

      # Innovation:
      omega = diff(alpha_reg)

      # Initialize the shrinkage terms:
      sigma_omega_tpk = abs(omega);
      xi_omega_tpk = matrix(1, nrow = T-1, ncol = K*(p-1)) # PX term

      # predictor p, factor k
      lambda_omega_pk = colMeans(sigma_omega_tpk)
      xi_omega_pk = rep(1, (p-1)*K) # PX term

      # predictor p:
      lambda_omega_p = rowMeans(matrix(lambda_omega_pk, nrow = p-1))
      xi_omega_p = rep(1, (p-1)) # PX term

      # global:
      lambda_omega_0 = mean(lambda_omega_p)
      xi_omega_0 = 1 # PX term

    } else{

      # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
      omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

      # factor k, predictor p:
      sigma_omega_kp = abs(omega)
      xi_omega_kp = matrix(1, nrow = K, ncol = p-1) # PX term

      # predictor p:
      lambda_omega_p = colMeans(sigma_omega_kp)
      xi_omega_p = rep(1, (p-1)) # PX term

      # global:
      lambda_omega_0 = mean(lambda_omega_p)
      xi_omega_0 = 1 # PX term
    }
  }
  #----------------------------------------------------------------------------
  # Store the MCMC output in separate arrays (better computation times)
  mcmc_output = vector('list', length(mcmc_params)); names(mcmc_output) = mcmc_params
  if(!is.na(match('beta', mcmc_params))) post.beta = array(NA, c(nsave, T, K))
  if(!is.na(match('fk', mcmc_params))) post.fk = array(NA, c(nsave, m, K))
  if(!is.na(match('alpha', mcmc_params))) post.alpha = array(NA, c(nsave, T, p, K))
  if(!is.na(match('mu_k', mcmc_params))) post.mu_k = array(NA, c(nsave, K))
  if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et = array(NA, c(nsave, T))
  if(!is.na(match('ar_phi', mcmc_params))) post.ar_phi = array(NA, c(nsave, K))
  if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat = array(NA, c(nsave, T, m))
  if(!is.na(match('Ypred', mcmc_params))) post.Ypred = array(NA, c(nsave, T, m))
  if(forecasting) {post.Yfore = post.Yfore_hat = array(NA, c(nsave, m))}
  if(computeDIC) post_loglike = numeric(nsave)

  # Total number of MCMC simulations:
  nstot = nburn+(nskip+1)*(nsave)
  skipcount = 0; isave = 0 # For counting

  # Run the MCMC:
  timer0 = proc.time()[3] # For timing the sampler
  for(nsi in 1:nstot){

    #----------------------------------------------------------------------------
    # Step 1: Impute the data, Y:
    #----------------------------------------------------------------------------
    if(any.missing){
      Y[na.ind] = Btheta[na.ind] + sigma_et[na.ind[,1]]*rnorm(nrow(na.ind))
      BtY = tcrossprod(t(splineInfo$Bmat), Y)
    }
    #----------------------------------------------------------------------------
    # Step 2: Sample the FLCs
    #----------------------------------------------------------------------------
    # Sample the FLCs
    Psi = fdlm_flc(BtY = BtY,
                   Beta  = Beta,
                   Psi = Psi,
                   BtB = splineInfo$BtB, #diag(nrow(BtY)),
                   Omega = splineInfo$Omega,
                   lambda = tau_f_k,
                   sigmat2 = sigma_et^2 + sigma_nu^2)
    # And update the loading curves:
    Fmat = splineInfo$Bmat%*%Psi;

    # Sample the smoothing parameters:
    tau_f_k = sample_lambda(tau_f_k, Psi, Omega = splineInfo$Omega, d = d, uniformPrior = TRUE, orderLambdas = FALSE)
    #----------------------------------------------------------------------------
    # Step 3: Sample the regression coefficients (and therefore the factors)
    #----------------------------------------------------------------------------
    # Pseudo-response and pseudo-variance depend on basis innovation:
    if(includeBasisInnovation){
      Y_tilde =  tcrossprod(theta, t(Psi)); sigma_tilde = sigma_nu
    } else {
      Y_tilde = crossprod(BtY, Psi); sigma_tilde = sigma_et
    }

    # Loop over each factor k = 1,...,K:
    for(k in 1:K){

      # Update the evoluation matrix:
      G_alpha[1,1] = ar_int[k]

      # Update the variances here:

      # Intercept/gamma:
      Wt[1,1,-T] = sigma_eta_tk[,k]^2; W0[1,1] = sigma_eta_0k[k]^2

      # Regression:
      if(p > 1){
        if(use_dynamic_reg){
          if(p == 2){
            # Special case: one predictor (plus intercept)
            Wt[2,2,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,1,k]
            W0[2,2] = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          } else {
            # Usual case: more than one predictor (plus intercept)
            for(j in 1:(p-1)) Wt[-1, -1,][j,j,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,j,k]
            diag(W0[-1, -1]) = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          }
        } else  W0[-1, -1] = diag(as.numeric(sigma_omega_kp[k,]^2), p - 1)
      }

      # Sanity check for Wt: if variances too large, KFAS will stop running
      Wt[which(Wt > 10^6, arr.ind = TRUE)] = 10^6; W0[which(W0 > 10^6, arr.ind = TRUE)] = 10^6

      # Update the SSModel object given the new parameters
      kfas_model = update_kfas_model(Y.dlm = as.matrix(Y_tilde[,k] - mu_k[k]),
                                     Zt = X.arr,
                                     sigma_et = sigma_tilde,
                                     Gt = G_alpha,
                                     Wt = Wt,
                                     W0 = W0,
                                     kfas_model = kfas_model)
      # Run the sampler
      alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

      # Conditional mean from regression equation:
      Beta[,k] = mu_k[k] + rowSums(X*alpha.arr[,,k])

      # Sample from the forecasting distribution, if desired:
      if(forecasting && (nsi > nburn)){ # Only need to compute after burnin
        # Evolution matrix: assume diagonal
        evol_diag = diag(as.matrix(kfas_model$T[,,T-1]))

        # Evolution error SD: assume diagonal
        evol_sd_diag = sqrt(diag(as.matrix(kfas_model$R[,,1]))*diag(as.matrix(kfas_model$Q[,,T-1])))

        # Point forecasting estimate:
        alpha_fore_hat[,k] = evol_diag*alpha.arr[T,,k]

        # Sample from forecasting distribution:
        alpha_fore[,k] = alpha_fore_hat[,k] + rnorm(n = p, mean = 0, sd = evol_sd_diag)
      }
    }

    # Update the forecasting terms:
    if(forecasting){
      # Factors:
      Beta_fore_hat = mu_k + matrix(X_Tp1%*%alpha_fore_hat)
      Beta_fore = mu_k + matrix(X_Tp1%*%alpha_fore)

      # Curves:
      Yfore_hat = Fmat%*%Beta_fore_hat
      Yfore = Fmat%*%Beta_fore + sigma_et[T]*rnorm(n = m)
    }

    # Store this term:
    BetaPsit = tcrossprod(Beta,Psi)
    #----------------------------------------------------------------------------
    # Step 4: Sample the basis terms (if desired)
    #----------------------------------------------------------------------------
    if(includeBasisInnovation){

      # Quad/linear construction a little faster w/o observation SV, but both work:
      if(use_obs_SV){
        Sigma_prec = matrix(rep(sigma_et^-2, ncol(theta)), nrow = T)
        chQtheta = sqrt(Sigma_prec + sigma_nu^-2) # Chol of diag (quadratic term) is just sqrt
        linTheta = t(BtY)*Sigma_prec + BetaPsit/sigma_nu^2 # Linear term from the posterior
      } else {
        chQtheta = sqrt(sigma_e^-2 + sigma_nu^-2) # Chol of diag (quadratic term) is just sqrt
        linTheta = t(BtY)/sigma_e^2 + BetaPsit/sigma_nu^2 # Linear term from the posterior
      }
      theta = linTheta/chQtheta^2 + 1/chQtheta*rnorm(length(theta))
      Btheta = tcrossprod(theta,splineInfo$Bmat)

      sigma_nu = 1/sqrt(truncdist::rtrunc(1, "gamma",
                                          a = 10^-8, b = Inf,
                                          shape = (length(theta)+1)/2,
                                          rate = 1/2*sum((theta - BetaPsit)^2)))
    } else {theta = BetaPsit; sigma_nu = 0; Btheta = tcrossprod(theta,splineInfo$Bmat)}
    #----------------------------------------------------------------------------
    # Step 5: Sample the observation error variance
    #----------------------------------------------------------------------------
    if(use_obs_SV){
      svParams = sampleCommonSV(Y - Btheta, svParams)
      sigma_et = svParams$sigma_et
    } else {
      # Or use uniform prior?
      sigma_e = 1/sqrt(rgamma(n = 1, shape = sum(!is.na(Y))/2, rate = sum((Y - Btheta)^2, na.rm=TRUE)/2))
      sigma_et = rep(sigma_e, T)
    }
    #----------------------------------------------------------------------------
    # Step 6: Sample the intercept/gamma parameters (Note: could use ASIS)
    #----------------------------------------------------------------------------
    # Centerend and non-centered:
    gamma_tk =  matrix(alpha.arr[,1,], nrow = T)

    gamma_tk_c = gamma_tk + mu_tk

    # Sample the unconditional mean term:
    mu_k = sampleARmu(yt = gamma_tk_c,
                      phi_j = ar_int,
                      sigma_tj = sigma_eta_tk,
                      priorPrec = 1/sigma_mu_k^2)
    mu_tk = matrix(rep(mu_k, each =  T), nrow = T)

    # And update the non-centered parameter:
    gamma_tk = gamma_tk_c - mu_tk

    # AR(1) coefficients:
    ar_int = sampleARphi(yt = gamma_tk,
                         phi_j = ar_int,
                         sigma_tj = sigma_eta_tk,
                         prior_phi = c(5,2))
                         #prior_phi = NULL)

    # Then subtract the AR(1) part:
    eta_tk = gamma_tk[-1,] -  t(ar_int*t(gamma_tk[-T,]))

    # Prior variance: MGP
    # Mean Part
    delta_mu_k =  sampleMGP(theta.jh = matrix(mu_k, ncol = K),
                            delta.h = delta_mu_k,
                            a1 = a1_mu, a2 = a2_mu)
    sigma_mu_k = 1/sqrt(cumprod(delta_mu_k))
    # And hyperparameters:
    if(sample_a1a2){
      a1_mu = uni.slice(a1_mu, g = function(a){
        dgamma(delta_mu_k[1], shape = a, rate = 1, log = TRUE) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)}, lower = 0, upper = Inf)
      a2_mu = uni.slice(a2_mu,g = function(a){
        sum(dgamma(delta_mu_k[-1], shape = a, rate = 1, log = TRUE)) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)},lower = 0, upper = Inf)
    }

    # Variance part:
    # Standardize, then reconstruct as matrix of size T x K:
    delta_eta_k = sampleMGP(theta.jh = matrix(eta_tk*sqrt(xi_eta_tk), ncol = K),
                            delta.h = delta_eta_k,
                            a1 = a1_eta, a2 = a2_eta)
    sigma_delta_k = 1/sqrt(cumprod(delta_eta_k))
    # And hyperparameters:
    if(sample_a1a2){
      a1_eta = uni.slice(a1_eta, g = function(a){
        dgamma(delta_eta_k[1], shape = a, rate = 1, log = TRUE) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)}, lower = 0, upper = Inf)
      a2_eta = uni.slice(a2_eta, g = function(a){
        sum(dgamma(delta_eta_k[-1], shape = a, rate = 1, log = TRUE)) +
          dgamma(a, shape = 2, rate = 1, log = TRUE)},lower = 0, upper = Inf)
    }

    # Sample the corresponding prior variance term(s):
    xi_eta_tk = matrix(rgamma(n = (T-1)*K,
                              shape = nu/2 + 1/2,
                              rate = nu/2 + (eta_tk/rep(sigma_delta_k, each = T-1))^2/2), nrow = T-1)
    # Sample degrees of freedom?
    if(sample_nu){
      nu = uni.slice(nu, g = function(nu){
        sum(dgamma(xi_eta_tk, shape = nu/2, rate = nu/2, log = TRUE)) +
          dunif(nu, min = 2, max = 128, log = TRUE)}, lower = 2, upper = 128)
    }
    # Or, inverse gamma prior on each k:
    #xi_eta_tk = matrix(rep(apply(eta_tk/rep(sigma_delta_k, each = T-1), 2, function(x)
    #  rgamma(n = 1, shape = (T-1)/2 + 0.01, rate = sum(x^2)/2 + 0.01)),
    #  each = T-1), nrow = T-1)

    # Or, fix at 1:
    #xi_eta_tk = matrix(1, nrow = T-1, ncol = K)

    # Initial sd:
    sigma_eta_0k = 1/sqrt(rgamma(n = K,
                                 shape = 3/2 + 1/2,
                                 rate = 3/2 + gamma_tk[1,]^2/2))

    # Update the error SD for gamma:
    sigma_eta_tk = rep(sigma_delta_k, each = T-1)/sqrt(xi_eta_tk)

    # Cap at machine epsilon:
    sigma_eta_tk[which(sigma_eta_tk < sqrt(.Machine$double.eps), arr.ind = TRUE)] = sqrt(.Machine$double.eps)
    #----------------------------------------------------------------------------
    # Step 7: Sample the non-intercept parameters:
    #----------------------------------------------------------------------------
    # Non-intercept term:
    if(p > 1){

      if(use_dynamic_reg){

        # Dynamic setting

        # Regression (non-intercept) coefficients
        alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

        # Initial variance:
        sigma_alpha_0k = 1/sqrt(rgamma(n = K*(p-1),
                                       shape = 3/2 + 1/2,
                                       rate = 3/2 + alpha_reg[1,]^2/2))

        # Random walk, so compute difference for innovations:
        omega = diff(alpha_reg)

        #----------------------------------------------------------------------------
        # tpk-specicif terms:
        omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
        sigma_omega_tpk = matrix(1/sqrt(rgamma(n = (T-1)*K*(p-1),
                                               shape = 1/2 + 1/2,
                                               rate = xi_omega_tpk + omega2/2)), nrow = T-1)
        xi_omega_tpk = matrix(rgamma(n = (T-1)*K*(p-1),
                                     shape = 1/2 + 1/2,
                                     rate = rep(1/lambda_omega_pk^2, each = T-1) + 1/sigma_omega_tpk^2), nrow = T-1)
        #----------------------------------------------------------------------------
        # predictor p, factor k
        lambda_omega_pk = 1/sqrt(rgamma(n = (p-1)*K,
                                        shape = 1/2 + (T-1)/2,
                                        rate = xi_omega_pk + colSums(xi_omega_tpk)))
        xi_omega_pk = rgamma(n = (p-1)*K,
                             shape = 1/2 + 1/2,
                             rate = rep(1/lambda_omega_p^2, times = K) + 1/lambda_omega_pk^2)
        #----------------------------------------------------------------------------
        # predictor p:
        lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                        shape = 1/2 + K/2,
                                        rate = xi_omega_p + rowSums(matrix(xi_omega_pk, nrow = p-1))))
        xi_omega_p = rgamma(n = p-1,
                            shape = 1/2 + 1/2,
                            rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
        #----------------------------------------------------------------------------
        # global:
        lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                       shape = 1/2 + (p-1)/2,
                                       rate = xi_omega_0 + sum(xi_omega_p)))
        xi_omega_0 = rgamma(n = 1,
                            shape = 1/2 + 1/2,
                            rate = (T-1) + 1/lambda_omega_0^2)
                            #rate = 1 + 1/lambda_omega_0^2)
      } else{

        # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
        omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

        #----------------------------------------------------------------------------
        # factor k, predictor p:
        omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
        sigma_omega_kp = matrix(1/sqrt(rgamma(n = K*(p-1),
                                              shape = 1/2 + 1/2,
                                              rate = xi_omega_kp + omega2/2)), nrow = K)
        xi_omega_kp = matrix(rgamma(n = K*(p-1),
                                    shape = 1/2 + 1/2,
                                    rate = rep(1/lambda_omega_p^2, each = K) + 1/sigma_omega_kp^2), nrow = K)
        #----------------------------------------------------------------------------
        # predictor p:
        lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                       shape = 1/2 + K/2,
                                       rate = xi_omega_p + colSums(xi_omega_kp)))
        xi_omega_p = rgamma(n = p-1,
                            shape = 1/2 + 1/2,
                            rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
        #----------------------------------------------------------------------------
        # global:
        lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                       shape = 1/2 + (p-1)/2,
                                       rate = xi_omega_0 + sum(xi_omega_p)))
        xi_omega_0 = rgamma(n = 1,
                            shape = 1/2 + 1/2,
                            rate = 1 + 1/lambda_omega_0^2)
      }
    }
    #----------------------------------------------------------------------------
    # Step 10: Adjust the ordering
    #----------------------------------------------------------------------------
    #if(nsi == 10 && K > 1){adjOrder = order(tau_f_k, decreasing = TRUE); tau_f_k = tau_f_k[adjOrder]; Psi = Psi[,adjOrder]; Beta = as.matrix(Beta[,adjOrder])}

    # Store the MCMC output:
    if(nsi > nburn){
      # Increment the skip counter:
      skipcount = skipcount + 1

      # Save the iteration:
      if(skipcount > nskip){
        # Increment the save index
        isave = isave + 1

        # Save the MCMC samples:
        if(!is.na(match('beta', mcmc_params))) post.beta[isave,,] = Beta
        if(!is.na(match('fk', mcmc_params))) post.fk[isave,,] = Fmat
        if(!is.na(match('alpha', mcmc_params))) post.alpha[isave,,,] = alpha.arr
        if(!is.na(match('mu_k', mcmc_params))) post.mu_k[isave,] = mu_k
        if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et[isave,] = sigma_et
        if(!is.na(match('ar_phi', mcmc_params))) post.ar_phi[isave,] = ar_int
        if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat[isave,,] = Btheta # + sigma_e*rnorm(length(Y))
        if(!is.na(match('Ypred', mcmc_params))) post.Ypred[isave,,] = rnorm(n = T*m, mean = matrix(Btheta), sd = rep(sigma_et,m))
        if(forecasting) {post.Yfore[isave,] = Yfore; post.Yfore_hat[isave,] = Yfore_hat}
        if(computeDIC) post_loglike[isave] = sum(dnorm(matrix(Yna), mean = matrix(Btheta), sd = rep(sigma_et,m), log = TRUE), na.rm = TRUE)

        # And reset the skip counter:
        skipcount = 0
      }
    }
    computeTimeRemaining(nsi, timer0, nstot, nrep = 500)
  }

  # Store the results (and correct for rescaling by sdY):
  if(!is.na(match('beta', mcmc_params))) mcmc_output$beta = post.beta*sdY
  if(!is.na(match('fk', mcmc_params))) mcmc_output$fk = post.fk
  if(!is.na(match('alpha', mcmc_params))) mcmc_output$alpha = post.alpha*sdY
  if(!is.na(match('mu_k', mcmc_params))) mcmc_output$mu_k = post.mu_k*sdY
  if(!is.na(match('sigma_et', mcmc_params))) mcmc_output$sigma_et = post.sigma_et*sdY
  if(!is.na(match('ar_phi', mcmc_params))) mcmc_output$ar_phi = post.ar_phi
  if(!is.na(match('Yhat', mcmc_params))) mcmc_output$Yhat = post.Yhat*sdY
  if(!is.na(match('Ypred', mcmc_params))) mcmc_output$Ypred = post.Ypred*sdY
  if(forecasting) {mcmc_output$Yfore = post.Yfore*sdY; mcmc_output$Yfore_hat = post.Yfore_hat*sdY}

  if(computeDIC){
    # Log-likelihood evaluated at posterior means:
    loglike_hat = sum(dnorm(matrix(Yna),
                            mean = matrix(colMeans(post.Yhat)),
                            sd = rep(colMeans(post.sigma_et), m),
                            log = TRUE), na.rm=TRUE)

    # Effective number of parameters (Note: two options)
    p_d = c(2*(loglike_hat - mean(post_loglike)),
            2*var(post_loglike))
    # DIC:
    DIC = -2*loglike_hat + 2*p_d

    # Store the DIC and the effective number of parameters (p_d)
    mcmc_output$DIC = DIC; mcmc_output$p_d = p_d
  }

  print(paste('Total time: ', round((proc.time()[3] - timer0)/60), 'minutes'))

  return (mcmc_output);
}
#' MCMC Sampling Algorithm for the Dynamic Function-on-Scalars Regression Model with AR(1) Errors
#' and a Known Basis Expansion
#'
#' Runs the MCMC for the dynamic function-on-scalars regression model based on
#' a known basis expansion (splines or functional principal components).
#' Here, we assume the factor regression has AR(1) errors.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param X the \code{T x p} matrix of predictors; if NULL, only include an intercept
#' @param use_dynamic_reg logical; if TRUE, regression coefficients are dynamic
#' (with random walk models)
#' @param use_fpca logical; if TRUE, use functional principal components basis; otherwise use splines
#' @param use_shrinkage_priors logical; if TRUE, include shrinkage priors for the coefficients
#' @param nsave number of MCMC iterations to record
#' @param nburn number of MCMC iterations to discard (burin-in)
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param mcmc_params named list of parameters for which we store the MCMC output;
#' must be one or more of
#' \itemize{
#' \item "beta" (dynamic factors)
#' \item "fk" (loading curves)
#' \item "alpha" (regression coefficients; possibly dynamic)
#' \item "mu_k" (intercept term for factor k)
#' \item "ar_phi" (AR coefficients for each k under AR(1) model)
#' \item "sigma_et" (observation error SD; possibly dynamic)
#' \item "Yhat" (fitted values)
#' \item "Yfore" (one-step forecast; includes the estimate and the distribution)
#' }
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1}
#' @param use_obs_SV logical; when TRUE, include a stochastic volatility model
#' for the observation error variance
#' @param computeDIC logical; if TRUE, compute the deviance information criterion \code{DIC}
#' and the effective number of parameters \code{p_d}
#' @return A named list of the \code{nsave} MCMC samples for the parameters named in \code{mcmc_params}
#'
#'
#' @import  KFAS truncdist
#' @importFrom refund fpca.face
#' @export
dfosr_basis_ar = function(Y, tau, X = NULL,
                          use_dynamic_reg = TRUE,
                          use_fpca = TRUE, use_shrinkage_priors = FALSE,
                          nsave = 1000, nburn = 1000, nskip = 2,
                          mcmc_params = list("beta", "fk", "alpha", "mu_k", "ar_phi"),
                          X_Tp1 = 1,
                          use_obs_SV = FALSE,
                          computeDIC = TRUE){
  #----------------------------------------------------------------------------
  # Assume that we've done checks elsewhere
  #----------------------------------------------------------------------------
  # Convert tau to matrix, if necessary:
  tau = as.matrix(tau)

  # Compute the dimensions:
  T = nrow(Y); m = ncol(Y); d = ncol(tau)

  # Rescale observation points to [0,1]
  tau01 = apply(tau, 2, function(x) (x - min(x))/(max(x) - min(x)))

  # Rescale by observation SD (and correct parameters later):
  sdY = sd(Y, na.rm=TRUE);
  Y = Y/sdY;
  #----------------------------------------------------------------------------
  # Initialize the main terms:

  if(any(is.na(Y))) stop("Missing data not implemented for basis methods")
  Yna = Y # for consistency later

  if(use_fpca){
    if(d > 1) stop("FPCA only implemented for d = 1")

    Fmat = fpca.face(Y, center = TRUE,
                     argvals = as.numeric(tau01),
                     knots=  max(15, min(ceiling(floor(median(rowSums(!is.na(Y))))/4), 150)),
                     pve=0.99)$efunctions
  } else Fmat = getSplineInfo_d(tau = tau01, m_eff = floor(median(rowSums(!is.na(Y)))), orthonormalize = TRUE)$Bmat
  #} else Fmat = getSplineInfo_d(tau = tau01, m_eff = floor(0.1*median(rowSums(!is.na(Y)))), orthonormalize = TRUE)$Bmat

  # Initialize the factors
  Beta = tcrossprod(Y, t(Fmat))
  K = ncol(Beta) # to be sure we have the right value

  # Initialize the conditional expectation:
  Btheta = tcrossprod(Beta, Fmat)

  # Initialize the (time-dependent) observation error SD:
  if(use_obs_SV){
    svParams = initCommonSV(Y - Btheta)
    sigma_et = svParams$sigma_et
  } else {
    sigma_e = sd(Y - Btheta, na.rm=TRUE)
    sigma_et = rep(sigma_e, T)
  }
  #----------------------------------------------------------------------------
  # Predictors:
  if(!is.null(X)){
    # Assuming we have some predictors:
    X = as.matrix(X)

    # Remove any predictors which are constants/intercepts:
    const.pred = apply(X, 2, function(x) all(diff(x) == 0))
    if(any(const.pred)) X = as.matrix(X[,!const.pred])

    # Center and scale the (non-constant) predictors:
    # Note: may not be appropriate for intervention effects!
    #X = scale(X)
  }
  # Include an intercept:
  X = cbind(rep(1, T), X); #colnames(X)[1] = paste(intercept_model, "-Intercept", sep='')

  # Number of predictors:
  p = ncol(X)

  # Initialize the SSModel:
  X.arr = array(t(X), c(1, p, T))
  kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,1]), Zt = X.arr)

  # Identify the dynamic/non-dynamic components:
  if(p > 1 && !use_dynamic_reg) diag(kfas_model$R[,,1])[-1] = 0

  # Forecasting setup and checks:
  if(!is.na(match('Yfore', mcmc_params))){
    forecasting = TRUE # useful

    # Check the forecasting design points:
    if(length(X_Tp1) != p)
      stop("Dimension of predictor X_Tp1 for forecasting must align with alpha;
           try including/excluding an intercept or omit 'Yfore' from the mcmc_params list")
    X_Tp1 = matrix(X_Tp1, ncol = p)

    # Storage for the forecast estimate and distribution:
    alpha_fore_hat = alpha_fore = matrix(0, nrow = p, ncol = K)
  } else forecasting = FALSE
  #----------------------------------------------------------------------------
  # Overall mean term (and T x K case)
  mu_k = as.matrix(colMeans(Beta)); mu_tk = matrix(rep(mu_k, each =  T), nrow = T)

  # Prior SD term for mu_k:
  sigma_mu_k = rep(100, K) # This will stay fixed
  #a1_mu = 2; a2_mu = 3
  #delta_mu_k = sampleMGP(matrix(mu_k, ncol = K), rep(1,K), a1 = a1_mu, a2 = a2_mu)
  #sigma_mu_k = 1/sqrt(cumprod(delta_mu_k))
  #----------------------------------------------------------------------------
  # AR(1) Evolution Matrix
  G_alpha = diag(p) # Replace the intercept terms as needed

  # AR(1) coefficients:
  ar_int = apply(Beta - mu_tk, 2, function(x) lm(x[-1] ~ - 1 +  x[-length(x)])$coef)

  # Stationarity fix:
  ar_int[which(abs(ar_int) > 0.95)] = 0.8*sign(ar_int[which(abs(ar_int) > 0.95)])

  #----------------------------------------------------------------------------
  # Initialize the regression terms:
  alpha.arr = array(0, c(T, p, K))
  for(k in 1:K){
    # Update the evoluation matrix
    G_alpha[1,1] = ar_int[k]

    # Update the SSModel object given the new parameters
    kfas_model = update_kfas_model(Y.dlm = as.matrix(Beta[,k] - mu_k[k]),
                                   Zt = X.arr,
                                   Gt = G_alpha,
                                   kfas_model = kfas_model)
    # Run the sampler
    alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

    # Conditional mean from regression equation:
    Beta[,k] = mu_k[k] + rowSums(X*alpha.arr[,,k])
  }

  #----------------------------------------------------------------------------
  # Evolution error variance:
  Wt = array(diag(p), c(p, p, T)); W0 = diag(10^-4, p);

  # Intercept (or gamma) components:
  gamma_tk =  matrix(alpha.arr[,1,], nrow = T)

  # Then subtract the AR(1) part:
  eta_tk = gamma_tk[-1,] -  t(ar_int*t(gamma_tk[-T,]))

  # Initial variance term:
  sigma_eta_0k = abs(gamma_tk[1,])

  # Initial SD:
  sigma_eta_k = apply(eta_tk, 2, function(x) sd(x, na.rm = TRUE))

  # Update the error SD for gamma:
  sigma_eta_tk = matrix(rep(sigma_eta_k, each = T-1), nrow = T-1)
  #----------------------------------------------------------------------------
  # Non-intercept term:
  if(p > 1){

    if(use_dynamic_reg){

      # Dynamic setting:
      alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

      # Initial variance:
      sigma_alpha_0k = abs(alpha_reg[1,])

      # Innovation:
      omega = diff(alpha_reg)

      if(use_shrinkage_priors){
        # Initialize the shrinkage terms:
        sigma_omega_tpk = abs(omega);
        xi_omega_tpk = matrix(1, nrow = T-1, ncol = K*(p-1)) # PX term

        # predictor p, factor k
        lambda_omega_pk = colMeans(sigma_omega_tpk)
        xi_omega_pk = rep(1, (p-1)*K) # PX term

        # predictor p:
        lambda_omega_p = rowMeans(matrix(lambda_omega_pk, nrow = p-1))
        xi_omega_p = rep(1, (p-1)) # PX term

        # global:
        lambda_omega_0 = mean(lambda_omega_p)
        xi_omega_0 = 1 # PX term
      } else {
        # Initialize the SD terms:
        sigma_omega_pk = apply(omega, 2, function(x) sd(x, na.rm =TRUE))
        sigma_omega_tpk = matrix(rep(sigma_omega_pk, each = T-1), nrow = T-1)
      }

    } else{

      # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
      omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

      if(use_shrinkage_priors){
        # factor k, predictor p:
        sigma_omega_kp = abs(omega)
        xi_omega_kp = matrix(1, nrow = K, ncol = p-1) # PX term

        # predictor p:
        lambda_omega_p = colMeans(sigma_omega_kp)
        xi_omega_p = rep(1, (p-1)) # PX term

        # global:
        lambda_omega_0 = mean(lambda_omega_p)
        xi_omega_0 = 1 # PX term
      } else {
        # Initialize the SD terms:
        sigma_omega_p = apply(omega, 2, function(x) sd(x, na.rm =TRUE))
        sigma_omega_kp = matrix(rep(sigma_omega_p, each = K), nrow = K)
      }
    }
  }
  #----------------------------------------------------------------------------
  # Store the MCMC output in separate arrays (better computation times)
  mcmc_output = vector('list', length(mcmc_params)); names(mcmc_output) = mcmc_params
  if(!is.na(match('beta', mcmc_params))) post.beta = array(NA, c(nsave, T, K))
  if(!is.na(match('fk', mcmc_params))) post.fk = array(NA, c(nsave, m, K))
  if(!is.na(match('alpha', mcmc_params))) post.alpha = array(NA, c(nsave, T, p, K))
  if(!is.na(match('mu_k', mcmc_params))) post.mu_k = array(NA, c(nsave, K))
  if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et = array(NA, c(nsave, T))
  if(!is.na(match('ar_phi', mcmc_params))) post.ar_phi = array(NA, c(nsave, K))
  if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat = array(NA, c(nsave, T, m))
  if(!is.na(match('Ypred', mcmc_params))) post.Ypred = array(NA, c(nsave, T, m))
  if(forecasting) {post.Yfore = post.Yfore_hat = array(NA, c(nsave, m))}
  if(computeDIC) post_loglike = numeric(nsave)

  # Total number of MCMC simulations:
  nstot = nburn+(nskip+1)*(nsave)
  skipcount = 0; isave = 0 # For counting

  # Run the MCMC:
  timer0 = proc.time()[3] # For timing the sampler
  for(nsi in 1:nstot){

    #----------------------------------------------------------------------------
    # Step 1: Impute the data, Y:
    #----------------------------------------------------------------------------
    # Not implemented in this case
    #----------------------------------------------------------------------------
    # Step 2: Sample the FLCs
    #----------------------------------------------------------------------------
    # Not necessary here
    #----------------------------------------------------------------------------
    # Step 3: Sample the regression coefficients (and therefore the factors)
    #----------------------------------------------------------------------------
    # Pseudo-response and pseudo-variance:
    Y_tilde = tcrossprod(Y, t(Fmat)); sigma_tilde = sigma_et

    # Loop over each factor k = 1,...,K:
    for(k in 1:K){

      # Update the evoluation matrix:
      G_alpha[1,1] = ar_int[k]

      # Update the variances here:

      # Intercept/gamma:
      Wt[1,1,-T] = sigma_eta_tk[,k]^2; W0[1,1] = sigma_eta_0k[k]^2

      # Regression:
      if(p > 1){
        if(use_dynamic_reg){
          if(p == 2){
            # Special case: one predictor (plus intercept)
            Wt[2,2,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,1,k]
            W0[2,2] = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          } else {
            # Usual case: more than one predictor (plus intercept)
            for(j in 1:(p-1)) Wt[-1, -1,][j,j,-T] = array(sigma_omega_tpk^2, c(T-1, p-1, K))[,j,k]
            diag(W0[-1, -1]) = matrix(sigma_alpha_0k^2, nrow = p-1)[,k]
          }
        } else  W0[-1, -1] = diag(as.numeric(sigma_omega_kp[k,]^2), p - 1)
      }

      # Sanity check for Wt: if variances too large, KFAS will stop running
      Wt[which(Wt > 10^6, arr.ind = TRUE)] = 10^6; W0[which(W0 > 10^6, arr.ind = TRUE)] = 10^6

      # Update the SSModel object given the new parameters
      kfas_model = update_kfas_model(Y.dlm = as.matrix(Y_tilde[,k] - mu_k[k]),
                                     Zt = X.arr,
                                     sigma_et = sigma_tilde,
                                     Gt = G_alpha,
                                     Wt = Wt,
                                     W0 = W0,
                                     kfas_model = kfas_model)
      # Run the sampler
      alpha.arr[,,k] = simulateSSM(kfas_model, "states", nsim = 1, antithetics=FALSE, filtered=FALSE)[,,1]

      # Conditional mean from regression equation:
      Beta[,k] = mu_k[k] + rowSums(X*alpha.arr[,,k])

      # Sample from the forecasting distribution, if desired:
      if(forecasting && (nsi > nburn)){ # Only need to compute after burnin
        # Evolution matrix: assume diagonal
        evol_diag = diag(as.matrix(kfas_model$T[,,T-1]))

        # Evolution error SD: assume diagonal
        evol_sd_diag = sqrt(diag(as.matrix(kfas_model$R[,,1]))*diag(as.matrix(kfas_model$Q[,,T-1])))

        # Point forecasting estimate:
        alpha_fore_hat[,k] = evol_diag*alpha.arr[T,,k]

        # Sample from forecasting distribution:
        alpha_fore[,k] = alpha_fore_hat[,k] + rnorm(n = p, mean = 0, sd = evol_sd_diag)
      }
    }

    # Update the forecasting terms:
    if(forecasting){
      # Factors:
      Beta_fore_hat = mu_k + matrix(X_Tp1%*%alpha_fore_hat)
      Beta_fore = mu_k + matrix(X_Tp1%*%alpha_fore)

      # Curves:
      Yfore_hat = Fmat%*%Beta_fore_hat
      Yfore = Fmat%*%Beta_fore + sigma_et[T]*rnorm(n = m)
    }
    #----------------------------------------------------------------------------
    # Step 4: Sample the basis terms (if desired)
    #----------------------------------------------------------------------------
    # Not necessary, but update the conditional mean here
    Btheta = tcrossprod(Beta, Fmat)
    #----------------------------------------------------------------------------
    # Step 5: Sample the observation error variance
    #----------------------------------------------------------------------------
    if(use_obs_SV){
      svParams = sampleCommonSV(Y - Btheta, svParams)
      sigma_et = svParams$sigma_et
    } else {
      # Or use uniform prior?
      sigma_e = 1/sqrt(rgamma(n = 1, shape = sum(!is.na(Y))/2, rate = sum((Y - Btheta)^2, na.rm=TRUE)/2))
      sigma_et = rep(sigma_e, T)
    }
    #----------------------------------------------------------------------------
    # Step 6: Sample the intercept/gamma parameters (Note: could use ASIS)
    #----------------------------------------------------------------------------
    # Centerend and non-centered:
    gamma_tk =  matrix(alpha.arr[,1,], nrow = T)

    gamma_tk_c = gamma_tk + mu_tk

    # Sample the unconditional mean term:
    mu_k = sampleARmu(yt = gamma_tk_c,
                      phi_j = ar_int,
                      sigma_tj = sigma_eta_tk,
                      priorPrec = 1/sigma_mu_k^2)
    mu_tk = matrix(rep(mu_k, each =  T), nrow = T)

    # And update the non-centered parameter:
    gamma_tk = gamma_tk_c - mu_tk

    # AR(1) coefficients:
    ar_int = sampleARphi(yt = gamma_tk,
                         phi_j = ar_int,
                         sigma_tj = sigma_eta_tk,
                         prior_phi = c(5,2))
    #prior_phi = NULL)

    # Then subtract the AR(1) part:
    eta_tk = gamma_tk[-1,] -  t(ar_int*t(gamma_tk[-T,]))

    # Prior variance:
    sigma_eta_k = 1/sqrt(rgamma(n = K,
                                shape = 0.01 + (T-1)/2,
                                rate = 0.01 + colSums(eta_tk^2)/2))
    # Initial sd:
    sigma_eta_0k = 1/sqrt(rgamma(n = K,
                                 shape = 3/2 + 1/2,
                                 rate = 3/2 + gamma_tk[1,]^2/2))

    # Update the error SD for gamma:
    sigma_eta_tk = matrix(rep(sigma_eta_k, each = T-1), nrow = T-1)

    # Cap at machine epsilon:
    sigma_eta_tk[which(sigma_eta_tk < sqrt(.Machine$double.eps), arr.ind = TRUE)] = sqrt(.Machine$double.eps)
    #----------------------------------------------------------------------------
    # Step 7: Sample the non-intercept parameters:
    #----------------------------------------------------------------------------
    # Non-intercept term:
    if(p > 1){

      if(use_dynamic_reg){

        # Dynamic setting

        # Regression (non-intercept) coefficients
        alpha_reg = matrix(alpha.arr[,-1,], nrow = T)

        # Initial variance:
        sigma_alpha_0k = 1/sqrt(rgamma(n = K*(p-1),
                                       shape = 3/2 + 1/2,
                                       rate = 3/2 + alpha_reg[1,]^2/2))

        # Random walk, so compute difference for innovations:
        omega = diff(alpha_reg)

        if(use_shrinkage_priors){
          #----------------------------------------------------------------------------
          # tpk-specicif terms:
          omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
          sigma_omega_tpk = matrix(1/sqrt(rgamma(n = (T-1)*K*(p-1),
                                                 shape = 1/2 + 1/2,
                                                 rate = xi_omega_tpk + omega2/2)), nrow = T-1)
          xi_omega_tpk = matrix(rgamma(n = (T-1)*K*(p-1),
                                       shape = 1/2 + 1/2,
                                       rate = rep(1/lambda_omega_pk^2, each = T-1) + 1/sigma_omega_tpk^2), nrow = T-1)
          #----------------------------------------------------------------------------
          # predictor p, factor k
          lambda_omega_pk = 1/sqrt(rgamma(n = (p-1)*K,
                                          shape = 1/2 + (T-1)/2,
                                          rate = xi_omega_pk + colSums(xi_omega_tpk)))
          xi_omega_pk = rgamma(n = (p-1)*K,
                               shape = 1/2 + 1/2,
                               rate = rep(1/lambda_omega_p^2, times = K) + 1/lambda_omega_pk^2)
          #----------------------------------------------------------------------------
          # predictor p:
          lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                         shape = 1/2 + K/2,
                                         rate = xi_omega_p + rowSums(matrix(xi_omega_pk, nrow = p-1))))
          xi_omega_p = rgamma(n = p-1,
                              shape = 1/2 + 1/2,
                              rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
          #----------------------------------------------------------------------------
          # global:
          lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                         shape = 1/2 + (p-1)/2,
                                         rate = xi_omega_0 + sum(xi_omega_p)))
          xi_omega_0 = rgamma(n = 1,
                              shape = 1/2 + 1/2,
                              rate = (T-1) + 1/lambda_omega_0^2)
          #rate = 1 + 1/lambda_omega_0^2)
        } else {

          # Sample the SD terms:
          sigma_omega_pk = 1/sqrt(rgamma(n = (p-1)*K,
                                         shape = 0.01 + (T-1)/2,
                                         rate = 0.01 + colSums(omega^2)/2))
          sigma_omega_tpk = matrix(rep(sigma_omega_pk, each = T-1), nrow = T-1)

        }

      } else{

        # Non-dynamic setting: grab the first one (all the same) and store as (K x p-1) matrix
        omega = matrix(t(alpha.arr[1, -1, ]), nrow = K)

        if(use_shrinkage_priors){
          #----------------------------------------------------------------------------
          # factor k, predictor p:
          omega2 = omega^2; omega2 = omega2 + (omega2 < 10^-16)*10^-8
          sigma_omega_kp = matrix(1/sqrt(rgamma(n = K*(p-1),
                                                shape = 1/2 + 1/2,
                                                rate = xi_omega_kp + omega2/2)), nrow = K)
          xi_omega_kp = matrix(rgamma(n = K*(p-1),
                                      shape = 1/2 + 1/2,
                                      rate = rep(1/lambda_omega_p^2, each = K) + 1/sigma_omega_kp^2), nrow = K)
          #----------------------------------------------------------------------------
          # predictor p:
          lambda_omega_p = 1/sqrt(rgamma(n = p-1,
                                         shape = 1/2 + K/2,
                                         rate = xi_omega_p + colSums(xi_omega_kp)))
          xi_omega_p = rgamma(n = p-1,
                              shape = 1/2 + 1/2,
                              rate = rep(1/lambda_omega_0^2, p-1) + 1/lambda_omega_p^2)
          #----------------------------------------------------------------------------
          # global:
          lambda_omega_0 = 1/sqrt(rgamma(n = 1,
                                         shape = 1/2 + (p-1)/2,
                                         rate = xi_omega_0 + sum(xi_omega_p)))
          xi_omega_0 = rgamma(n = 1,
                              shape = 1/2 + 1/2,
                              rate = 1 + 1/lambda_omega_0^2)
        } else {

          # Sample the SD terms:
          sigma_omega_p = 1/sqrt(rgamma(n = (p-1),
                                        shape = 0.01 + (K-1)/2,
                                        rate = 0.01 + colSums(omega^2)/2))
          sigma_omega_kp = matrix(rep(sigma_omega_p, each = K), nrow = K)

        }
      }
    }
    #----------------------------------------------------------------------------
    # Step 10: Adjust the ordering
    #----------------------------------------------------------------------------
    # Not necessary here

    # Store the MCMC output:
    if(nsi > nburn){
      # Increment the skip counter:
      skipcount = skipcount + 1

      # Save the iteration:
      if(skipcount > nskip){
        # Increment the save index
        isave = isave + 1

        # Save the MCMC samples:
        if(!is.na(match('beta', mcmc_params))) post.beta[isave,,] = Beta
        if(!is.na(match('fk', mcmc_params))) post.fk[isave,,] = Fmat
        if(!is.na(match('alpha', mcmc_params))) post.alpha[isave,,,] = alpha.arr
        if(!is.na(match('mu_k', mcmc_params))) post.mu_k[isave,] = mu_k
        if(!is.na(match('sigma_et', mcmc_params)) || computeDIC) post.sigma_et[isave,] = sigma_et
        if(!is.na(match('ar_phi', mcmc_params))) post.ar_phi[isave,] = ar_int
        if(!is.na(match('Yhat', mcmc_params)) || computeDIC) post.Yhat[isave,,] = Btheta
        if(!is.na(match('Ypred', mcmc_params))) post.Ypred[isave,,] = rnorm(n = T*m, mean = matrix(Btheta), sd = rep(sigma_et,m))
        if(forecasting) {post.Yfore[isave,] = Yfore; post.Yfore_hat[isave,] = Yfore_hat}
        if(computeDIC) post_loglike[isave] = sum(dnorm(matrix(Yna), mean = matrix(Btheta), sd = rep(sigma_et,m), log = TRUE), na.rm = TRUE)

        # And reset the skip counter:
        skipcount = 0
      }
    }
    computeTimeRemaining(nsi, timer0, nstot, nrep = 500)
  }

  # Store the results (and correct for rescaling by sdY):
  if(!is.na(match('beta', mcmc_params))) mcmc_output$beta = post.beta*sdY
  if(!is.na(match('fk', mcmc_params))) mcmc_output$fk = post.fk
  if(!is.na(match('alpha', mcmc_params))) mcmc_output$alpha = post.alpha*sdY
  if(!is.na(match('mu_k', mcmc_params))) mcmc_output$mu_k = post.mu_k*sdY
  if(!is.na(match('sigma_et', mcmc_params))) mcmc_output$sigma_et = post.sigma_et*sdY
  if(!is.na(match('ar_phi', mcmc_params))) mcmc_output$ar_phi = post.ar_phi
  if(!is.na(match('Yhat', mcmc_params))) mcmc_output$Yhat = post.Yhat*sdY
  if(!is.na(match('Ypred', mcmc_params))) mcmc_output$Ypred = post.Ypred*sdY
  if(forecasting) {mcmc_output$Yfore = post.Yfore*sdY; mcmc_output$Yfore_hat = post.Yfore_hat*sdY}

  if(computeDIC){
    # Log-likelihood evaluated at posterior means:
    loglike_hat = sum(dnorm(matrix(Yna),
                            mean = matrix(colMeans(post.Yhat)),
                            sd = rep(colMeans(post.sigma_et), m),
                            log = TRUE), na.rm=TRUE)

    # Effective number of parameters (Note: two options)
    p_d = c(2*(loglike_hat - mean(post_loglike)),
            2*var(post_loglike))
    # DIC:
    DIC = -2*loglike_hat + 2*p_d

    # Store the DIC and the effective number of parameters (p_d)
    mcmc_output$DIC = DIC; mcmc_output$p_d = p_d
  }

  print(paste('Total time: ', round((proc.time()[3] - timer0)/60), 'minutes'))

  return (mcmc_output);
}
