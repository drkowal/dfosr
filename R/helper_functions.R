#----------------------------------------------------------------------------
#' Forecast with a dynamic function-on-scalars regression model
#'
#' Compute the one-step forecasting estimate
#' under a dynamic function-on-scalars regression model.
#'
#' @param X_Tp1 the \code{p x 1} matrix of predictors at the forecasting time point \code{T + 1};
#' if \code{NULL}, set to an intercept
#' @param post_sims a named list of posterior draws for the following parameters:
#' \itemize{
#' \item \code{alpha} (regression coefficients)
#' \item \code{fk} (loading curves)
#' \item \code{mu_k} (intercept term for factor k)
#' \item \code{ar_phi} (OPTIONAL; AR coefficients for each k under AR(1) model)
#' }
#' @param factor_model model for the (dynamic) factors;
#' must be one of
#' \itemize{
#' \item "IND" (independent errors)
#' \item "AR" (stationary autoregression of order 1)
#' \item "RW" (random walk model)
#' }
#'
#' @return \code{Yfore}, the \code{m x 1} curve forecasting estimate
#'
#' @examples
#' \dontrun{
#' # Simulate some data:
#' sim_data = simulate_dfosr(T = 200, m = 50, p_0 = 2, p_1 = 2)
#' Y = sim_data$Y; X = sim_data$X; tau = sim_data$tau
#' T = nrow(Y); m = ncol(Y); p = ncol(X) # Dimensions
#'
#' # Delete and store the last time point (for forecasting):
#' Y_Tp1 = Y[T,]; X_Tp1 = X[T,]
#' Y = Y[-T,]; X = X[-T,]; T = nrow(Y);
#'
#' # Run the MCMC w/ K = 6:
#' out = dfosr(Y = Y, tau = tau, X = X, K = 6,
#'            factor_model = 'AR',
#'            use_dynamic_reg = TRUE,
#'            mcmc_params = list("beta", "fk", "alpha", "mu_k", "ar_phi"))
#'
#' # Compute one-step forecasts:
#' Yfore = forecast_dfosr(X_Tp1 = X_Tp1,
#'                        post_sims = out,
#'                        factor_model = "AR")
#' # Plot the results:
#' plot(tau, Yfore, ylim = range(Yfore, Y_Tp1, Y[T,], na.rm=TRUE),
#'      main = 'One-Step Forecast',
#'      lwd = 8, col = "cyan", type = 'l')
#' # Add the most recent observed curve:
#' lines(tau, Y[T,], type='p', pch = 2)
#' # Add the realized curve:
#' lines(tau, Y_Tp1, type='p')
#' # And the true curve:
#' lines(tau, sim_data$Y_true[T+1,], lwd=8, col='black', lty=6)
#'}
#' @export
forecast_dfosr = function(X_Tp1 = NULL,
                          post_sims,
                          factor_model = 'AR'){
  # Convert to upper case, then check for matches to existing models:
  factor_model = toupper(factor_model);
  if(is.na(match(factor_model, c("RW", "AR", "IND"))))
    stop("The factor model must be one of 'RW', 'AR', or 'IND'")

  # Check: if AR, be sure that we've included the AR(1) parameters
  if(factor_model == 'AR' && is.na(match('ar_phi', names(post_sims))))
    stop("For AR, must include 'ar_phi' (autoregressive) parameters")

  # Check: is everything there?
  if(is.na(match('alpha', names(post_sims))))
    stop("Must include 'alpha' (regression coefficient) parameters")
  if(is.na(match('fk', names(post_sims))))
    stop("Must include 'fk' (loading curve) parameters")
  if(is.na(match('mu_k', names(post_sims))))
    stop("Must include 'mu_k' (intercept) parameters")

  # Compute dimensions locally:
  Nsims = dim(post_sims$fk)[1];
  m = dim(post_sims$fk)[2];
  K = dim(post_sims$fk)[3];
  T = dim(post_sims$alpha)[2]
  p = dim(post_sims$alpha)[3]

  # Check the matrix:
  if(is.null(X_Tp1)) X_Tp1 = 1
  if(length(X_Tp1) != p)
    stop("Dimension of predictor X_Tp1 must align with alpha; try including/exlcuding an intercept")
  X_Tp1 = matrix(X_Tp1, ncol = p)

  # Simple implementation: loop over samples
  Yfore = numeric(m) # Storage
  for(nsi in 1:Nsims){
    Betafore = numeric(K) # Forecast for each beta_k
    for(k in 1:K){
      if(factor_model=="RW")
        alpha_k_fore = post_sims$alpha[nsi,T,,k]
      if(factor_model=="AR")
        alpha_k_fore = c(post_sims$ar_phi[nsi,k]*post_sims$alpha[nsi,T,1,k], post_sims$alpha[nsi,T,-1,k])
      if(factor_model=="IND")
        alpha_k_fore = c(0, post_sims$alpha[nsi,T,-1,k])

      Betafore[k] = post_sims$mu_k[nsi,k] +  X_Tp1%*%alpha_k_fore
    }
    Yfore = Yfore + 1/Nsims*post_sims$fk[nsi,,]%*%Betafore
  }
  Yfore
}
#----------------------------------------------------------------------------
#' Simulate a dynamic function-on-scalars regression model
#'
#' Simulate data from a dynamic function-on-scalars regression model, allowing for
#' autocorrelated errors and (possibly) dynamic regression coefficients random effects.
#' The predictors are contemporaneously independent but (possibly) autocorrelated.
#'
#' @param T number of observed curves (i.e., number of time points)
#' @param m total number of observation points (i.e., points along the curve)
#' @param RSNR root signal-to-noise ratio
#' @param K_true rank of the model (i.e., number of basis functions used for the functional data simulations)
#' @param p_0 number of true zero regression coefficients
#' @param p_1 number of true nonzero regression coefficients
#' @param use_dynamic_reg logical; if TRUE, simulate dynamic regression coefficients; otherwise static
#' @param sparse_factors logical; if TRUE, then for each nonzero predictor j,
#' sample a subset of k=1:K_true factors to be nonzero
#' @param use_obs_SV logical; if TRUE, include stochastic volatility term for the error variance
#' @param ar1 AR(1) coefficient for time-correlated predictors
#' @param prop_missing proportion of missing data (between 0 and 1); default is zero
#'
#' @return a list containing the following:
#' \itemize{
#' \item \code{Y}: the simulated \code{T x m} functional data matrix
#' \item \code{X}: the simulated \code{T x p} design matrix
#' \item \code{tau}: the \code{m}-dimensional vector of observation points
#' \item \code{Y_true}: the true \code{T x m} functional data matrix (w/o noise)
#' \item \code{alpha_tilde_true} the true \code{T x p x m} array of regression coefficient functions
#' \item \code{alpha_arr_true} the true \code{T x p x K_true} array of (dynamic) regression coefficient factors
#' \item \code{Beta_true} the true \code{T x K_true} matrix of factors
#' \item \code{F_true} the true \code{m x K_true} matrix of basis (loading curve) functions
#' \item \code{sigma_true} the true observation error standard deviation
#' }
#'
#' @note The basis functions (or loading curves) are orthonormalized polynomials,
#' so large values of \code{K_true} are not recommended.
#'
#' @examples
#' # Example: simulate DFOSR
#' sim_data = simulate_dfosr()
#' Y = sim_data$Y; X = sim_data$X; tau = sim_data$tau
#'
#' @import truncdist
#' @export
simulate_dfosr = function(T = 200,
                         m = 100,
                         RSNR = 5,
                         K_true = 4,
                         p_0 = 2,
                         p_1 = 2,
                         use_dynamic_reg = TRUE,
                         sparse_factors = FALSE,
                         use_obs_SV = FALSE,
                         ar1 = 0,
                         prop_missing = 0){
  # Number of predictors:
  p = 1 + p_1 + p_0

  # Observation points:
  tau = seq(0, 1, length.out = m)

  # FLCs: orthonormalized polynomials
  F_true = cbind(1/sqrt(m),
                 poly(tau, K_true - 1))

  # Simulate the predictors:
  if(ar1 == 0){
    X = cbind(1, matrix(rnorm(n = T*(p-1)), nrow = T, ncol = p-1))
  } else X = cbind(1,
                   apply(matrix(0, nrow = T, ncol = p-1), 2, function(x)
                     arima.sim(n = T, list(ar = ar1), sd = sqrt(1-ar1^2))))

  # True coefficients:
  alpha_arr_true = array(0, c(T, p, K_true))

  # p = 1 Intercept coefficients: just use K:1
  alpha_arr_true[,1,] = matrix(rep(1/(1:K_true), each = T), nrow = T)

  # Factor standard deviation (or scale factor) for each k:
  #sd_k = 1/(1:K_true)
  sd_k = sqrt(.75^(1:K_true - 1))

  # p > 1: Possibly nonzero, possibly dynamic coefficients
  if(p_1 > 0){for(j in 1:p_1){
    # Which factors are nonzero for predictor j?
    if(sparse_factors){ # Truncated Poisson(1)
      k_p = sample(1:K_true, truncdist::rtrunc(n = 1, spec = 'pois', a = 1, b = K_true, lambda = 1))
    } else k_p = 1:K_true

    # Among nonzero factors, simulate dynamic (w/ sparse jumps) or static coefficients:
    for(k in k_p) {
      if(use_dynamic_reg){
        alpha_arr_true[,j+1,k] = sd_k[k]*rnorm(n = 1) + sd_k[k]*cumsum(rnorm(n = T)*(rbinom(n = T, size = 1, prob = 0.01)))
      } else alpha_arr_true[,j+1, k] = sd_k[k]*rnorm(n = 1)
    }
  }}

  # True regression coefficient functions:
  alpha_tilde_true = array(0, c(T, p, m))
  for(j in 1:p) alpha_tilde_true[,j,] = tcrossprod(alpha_arr_true[,j,], F_true)

  # Dynamic factors:
  Beta_true = matrix(0, nrow = T, ncol = K_true)
  for(k in 1:K_true) Beta_true[,k] = rowSums(X*alpha_arr_true[,,k]) + arima.sim(n = T, list(ar = 0.8), sd = sqrt(1-0.8^2)*sd_k[k])

  # True FTS:
  Y_true = tcrossprod(Beta_true, F_true)

  # Noise SD based on RSNR:
  sigma_true = sd(Y_true)/RSNR

  # Observed data:
  if(use_obs_SV){
    sigma_true = sigma_true*exp(1/2*arima.sim(n = T, list(ar = 0.9), sd = sqrt(1-0.9^2)))
    Y = Y_true + rep(sigma_true,  m)*rnorm(m*T)
  } else {
    Y = Y_true + sigma_true*rnorm(m*T)
  }

  # Remove any observation points:
  if(prop_missing > 0 ) Y[sample(1:length(Y), prop_missing*length(Y))] = NA

  list(Y = Y, X = X, tau = tau,
       Y_true = Y_true, alpha_tilde_true = alpha_tilde_true,
       alpha_arr_true = alpha_arr_true, Beta_true = Beta_true, F_true = F_true, sigma_true = sigma_true)
}
#----------------------------------------------------------------------------
#' Simulate a function-on-scalar regression model
#'
#' Simulate data from a function-on-scalar regression model, allowing for
#' subject-specific random effects. The predictors are multivariate normal with
#' mean zero and covariance \code{corr^abs(j1-j2)} for correlation parameter \code{corr}
#' between predictors \code{j1} and \code{j2}.
#' More predictors than observations (p > n) is allowed.
#'
#' @param n number of observed curves (i.e., number of subjects)
#' @param m total number of observation points (i.e., points along the curve)
#' @param RSNR root signal-to-noise ratio
#' @param K_true rank of the model (i.e., number of basis functions used for the functional data simulations)
#' @param p_0 number of true zero regression coefficients
#' @param p_1 number of true nonzero regression coefficients
#' @param sparse_factors logical; if TRUE, then for each nonzero predictor j,
#' sample a subset of k=1:K_true factors to be nonzero#'
#' @param corr correlation parameter for predictors
#' @param perc_missing percentage of missing data (between 0 and 1); default is zero
#'
#' @return a list containing the following:
#' \itemize{
#' \item \code{Y}: the simulated \code{n x m} functional data matrix
#' \item \code{X}: the simulated \code{n x p} design matrix
#' \item \code{tau}: the \code{m}-dimensional vector of observation points
#' \item \code{Y_true}: the true \code{n x m} functional data matrix (w/o noise)
#' \item \code{alpha_tilde_true} the true \code{m x p} matrix of regression coefficient functions
#' \item \code{alpha_arr_true} the true \code{K_true x p} matrix of regression coefficient factors
#' \item \code{Beta_true} the true \code{n x K_true} matrix of factors
#' \item \code{F_true} the true \code{m x K_true} matrix of basis (loading curve) functions
#' \item \code{sigma_true} the true observation error standard deviation
#' }
#'
#' @note The basis functions (or loading curves) are orthonormalized polynomials,
#' so large values of \code{K_true} are not recommended.
#'
#' @examples
#' # Example: simulate FOSR
#' sim_data = simulate_fosr(n = 100, m = 20, p_0 = 100, p_1 = 5)
#' Y = sim_data$Y; X = sim_data$X; tau = sim_data$tau
#'
#' @import truncdist
#' @export
simulate_fosr = function(n = 100,
                         m = 50,
                         RSNR = 5,
                         K_true = 4,
                         p_0 = 1000,
                         p_1 = 5,
                         sparse_factors = TRUE,
                         corr = 0,
                         perc_missing = 0){
  # Number of predictors:
  p = 1 + p_1 + p_0

  # Observation points:
  tau = seq(0, 1, length.out = m)

  # FLCs: orthonormalized polynomials
  F_true = cbind(1/sqrt(m),
                 poly(tau, K_true - 1))

  # Simulate the predictors:
  Xiid = matrix(rnorm(n = n*(p-1)), nrow = n, ncol = p-1)
  if(corr == 0){
    X = cbind(1,Xiid)
  } else {
    # Correlated predictors:
    ind_mat = matrix(rep(1:(p-1), p-1),nrow=p-1, byrow=FALSE);
    ind_diffs = abs(ind_mat - t(ind_mat))
    cov_mat = corr^ind_diffs
    # Cholesky:
    ch_cov_mat = chol(cov_mat)

    # Correlated predictors:
    X = cbind(1,
              t(crossprod(ch_cov_mat, t(Xiid))))
  }

  # True coefficients:
  alpha_arr_true = array(0, c(K_true, p))

  # p = 1 Intercept coefficients: decaying importance
  alpha_arr_true[,1] = 1/(1:K_true)

  # Simulate the nonzero factors
  # Nonzero indices: if correlated predictors, space out the true ones
  nonzero_ind = 1:p_1; if(corr != 0) nonzero_ind = round(seq(1, p-1, length.out = p_1))
  if(p_1 > 0){for(j in nonzero_ind){
    # Which factors are nonzero for predictor j?
    if(sparse_factors){ # Truncated Poisson(1)
      k_p = sample(1:K_true, truncdist::rtrunc(n = 1, spec = 'pois', a = 1, b = K_true, lambda = 1))
    } else k_p = 1:K_true

    # Simulate true values of the (nonzero) regression coefficients (decaying importance)
    alpha_arr_true[k_p, j+1] = rnorm(n = length(k_p), mean = 0, sd = 1/k_p)
  }}

  # True coefficient functions:
  alpha_tilde_true = F_true %*% alpha_arr_true # m x p

  # True factors: add Gaussian (subject-specific) noise (decaying importance)
  Beta_true = matrix(0, nrow = n, ncol = K_true)
  for(k in 1:K_true) Beta_true[,k] = X%*%alpha_arr_true[k,] + rnorm(n = n, sd = 1/k)

  # True FTS:
  Y_true = tcrossprod(Beta_true, F_true)

  # Noise SD based on RSNR:
  sigma_true = sd(Y_true)/RSNR

  # Observed data:
  Y = Y_true + sigma_true*rnorm(m*n)

  # Remove any observation points:
  if(perc_missing > 0 ) Y[sample(1:length(Y), perc_missing*length(Y))] = NA

  list(Y = Y, X = X, tau = tau,
       Y_true = Y_true, alpha_tilde_true = alpha_tilde_true,
       alpha_arr_true = alpha_arr_true, Beta_true = Beta_true, F_true = F_true, sigma_true = sigma_true)
}
#' Initialize the reduced-rank functional data model
#'
#' Compute initial values for the factors and loadings curves,
#' where the input matrix may have NAs.
#'
#' @param Y the \code{T x m} data observation matrix, where \code{T} is the number of time points and \code{m} is the number of observation points (\code{NA}s allowed)
#' @param tau the \code{m x d} matrix of coordinates of observation points
#' @param K the number of factors; if \code{NULL}, select using proportion of variability explained (0.99)
#' @param use_pace logical; if TRUE, use the PACE procedure for FPCA (only for \code{d} = 1);
#' otherwise use splines
#' @return a list containing
#' \itemize{
#' \item \code{Beta} the \code{T x K} matrix of factors
#' \item \code{Psi} the \code{J x K} matrix of loading curve basis coefficients,
#' where \code{J} is the number of spline basis functions
#' \item \code{splineInfo} a list with information about the spline basis
#' \item \code{Y0} the imputed data matrix
#' }
#'
#' @details The PACE procedure is useful for estimated a reduced-rank functional data model
#' in the case of sparsely-observed FD, i.e., many NAs in \code{Y}. However,
#' it is also only valid for univariate observation points
#'
#' @import fdapace
fdlm_init = function(Y, tau, K = NULL, use_pace = FALSE){

  # Convert to matrix, if necessary:
  tau = as.matrix(tau)

  # Rescale observation points to [0,1]
  tau01 = apply(tau, 2, function(x) (x - min(x))/(max(x) - min(x)))

  # And the dimensions:
  T = nrow(Y); m = ncol(Y); d = ncol(tau)

  # Compute basic quantities for the FLC splines:
  splineInfo = getSplineInfo_d(tau = tau01,
                               m_eff = floor(median(rowSums(!is.na(Y)))),
                               orthonormalize = TRUE)

  # For initialization: impute
  Y0 = matrix(NA, nrow = T, ncol = m) # Storage for imputed initialization data matrix
  allMissing.t = (rowSums(!is.na(Y))==0)   # Boolean indicator of times at which no points are observed

  # Use PACE, or just impute w/ splines:
  if((d==1) && use_pace && any(is.na(Y)) ){
    # To initialize, use FPCA via PACE:
    fit_fpca = FPCA(Ly =  apply(Y, 1, function(y) y[!is.na(y)]),
                    Lt = apply(Y, 1, function(y) tau01[!is.na(y)]),
                    optns = list(dataType = 'Sparse', methodSelectK = K))
    # Fitted FPCA curves:
    Yhat0 = fitted(fit_fpca); t0 = fit_fpca$workGrid

    # For all times at which we observe a curve, impute the full curve (across tau)
    Y0[!allMissing.t,] = t(apply(Yhat0[!allMissing.t,], 1, function(x) splinefun(t0, x, method='natural')(tau01)))

  } else {
    # For all times at which we observe a curve, impute the full curve (across tau)
    Y0[!allMissing.t,] = t(apply(Y[!allMissing.t,], 1, function(x) splinefun(tau01, x, method='natural')(tau01)))
  }

  # Next, impute any times for which no curve is observed (i.e., impute across time)
  Y0 = apply(Y0, 2, function(x){splinefun(1:T, x, method='natural')(1:T)})

  # Compute SVD of the (completed) data matrix:
    # (Complete) data matrix, projected onto basis:
  YB0 = Y0%*%splineInfo$Bmat%*%chol2inv(chol(splineInfo$BtB))
  singVal = svd(YB0)

  # If K is unspecified, select based on cpv
  if(is.null(K)){
    # Cumulative sums of the s^2 proportions (More reasonable when Y has been centered)
    cpv = cumsum(singVal$d^2/sum(singVal$d^2))
    K = max(2, which(cpv >= 0.99)[1])
  }

  # Check to make sure K is less than the number of basis coefficients!
  if(K >= ncol(YB0)){
    warning(paste("K must be less than the number of basis functions used; reducing from K =",K, "to K =", ncol(YB0) - 1))
    K = ncol(YB0) - 1
  }

  # Basis coefficients of FLCs:
  Psi0 = as.matrix(singVal$v[,1:K])

  # Initial FLCs:
  F0 = splineInfo$Bmat%*%Psi0

  # Factors:
  Beta0 = as.matrix((singVal$u%*%diag(singVal$d))[,1:K])

  # Initialize all curves to have positive sums (especially nice for the intercept)
  negSumk = which(colSums(F0) < 0); Psi0[,negSumk] = -Psi0[,negSumk]; Beta0[,negSumk] = -Beta0[,negSumk]

  list(Beta = Beta0, Psi = Psi0, splineInfo = splineInfo, Y0 = Y0)
}
#----------------------------------------------------------------------------
#' Compute the posterior distrubution for the regression coefficient functions
#'
#' Given posterior samples for the loading curves \code{fk} and the
#' regression coefficient factors \code{alpha_j} for a predictor \code{j},
#' compute the posterior distribution of the corresponding (dynamic) regression coefficient function.
#'
#' @param post_fk \code{Nsims x m x K} matrix of posterior draws of the loading curve matrix
#' @param post_alpha_j \code{Nsims x T x K} array or \code{Nsims x K} matrix
#' of posterior draws of the (dynamic) regression coefficient factors
#'
#' @return \code{Nsims x T x m} matrix of posterior draws of the regression coefficient function
#' @export
get_post_alpha_tilde = function(post_fk, post_alpha_j){

  # Compute dimensions:
  Nsims = dim(post_fk)[1]; m = dim(post_fk)[2]; K = dim(post_fk)[3]

  # Check: post_alpha_j might be (Nsims x K) or (Nsims x T x K)
  if(dim(post_alpha_j)[2] == K){
    # Non-dynamic case: (Nsims x K)
    post_alpha_tilde_j = array(0, c(Nsims, m))
    for(ni in 1:Nsims) post_alpha_tilde_j[ni,] = tcrossprod(post_alpha_j[ni,], post_fk[ni,,])
  } else {
    # Dynamic case: (Nsims x T x K)
    T = dim(post_alpha_j)[2]
    post_alpha_tilde_j = array(0, c(Nsims, T, m))
    for(ni in 1:Nsims) post_alpha_tilde_j[ni,,] = tcrossprod(post_alpha_j[ni,,], post_fk[ni,,])
  }

  post_alpha_tilde_j
}
#####################################################################################################
# Update (or initialize) the SSModel object used for sampling the factors
# Inputs:
# Y.dlm: (T x m0) matrix of response (w/ missing values); m0 is usually either K (fastImpute = TRUE) or m (fastImpute = FALSE)
# Zt: (m0 x K0) observation matrix or (m0 x K0 x T) observation array
# Optional inputs:
# Gt: (K0 x K0) evolution matrix or (K0 x K0 x T) array; if NULL, set as identity (for random walk)
# sigma_et: observation error SD(s): can be
# (T x m0) matrix, assuming time-dependent diagonal obs error variance (columns give diagonal elements)
# T-dimensional vector, assuming time-dependent scalar multiple of the identity
# 1-dimensional vector, assuming time-invariant scalar multiple of the identity
# if NULL, set as identity (for dimension purposes)
# Wt: (K0 x K0) matrix or (K0 x K0 x T) array of evolution error covariance; if NULL, set as identity (for dimension purposes)
# W0: (K0 x K0) matrix of initial evolution error covariance; if NULL, set as identity*10^4
# kfas_model: SSModel object from KFAS package; if NULL, construct model (might be slower!)
#####################################################################################################
#' @import KFAS
update_kfas_model = function(Y.dlm, Zt, sigma_et = NULL, Gt = NULL, Wt = NULL, W0 = NULL, kfas_model = NULL){

  if (!requireNamespace("KFAS", quietly = TRUE)) {
    stop("KFAS needed for this function to work. Please install it.",
         call. = FALSE)
  }

  # Compute these locally
  T = nrow(Y.dlm); m0 = ncol(Y.dlm) # m0 is usually either K (fastImpute = TRUE) or m (fastImpute = FALSE)
  K0 = ncol(Zt) # This is the dimension of the state vector

  # Evolution matrix: if NULL (unspecified), set as identity (for random walk model);
  # if (K0 x K0) matrix, transform to (K0 x K0 x T) array
  if(is.null(Gt)) {Gt = array(diag(K0), c(K0,K0,T))} else{if(length(dim(Gt)) == 2) Gt = array(Gt, c(K0, K0, T))}

  # Evolution error covariance matrix: if NULL (unspecified), set as identity (for dimension purposes only);
  # if (K0 x K0) matrix, transform to (K0 x K0 x T) array
  if(is.null(Wt)) {Wt = array(diag(K0), c(K0,K0,T))} else{if(length(dim(Wt)) == 2) Wt = array(Wt, c(K0, K0, T))}

  # Initial variance matrix (not an array in kfas!)
  if(is.null(W0)) W0 = diag(10^4, K0)

  # Observation error variance, which can be time-dependent (if NULL, just set it as the identity)
  Ht = array(diag(m0), c(m0, m0, T))
  if(!is.null(sigma_et)){
    if(length(sigma_et) == 1) sigma_et = rep(sigma_et, T);  #if(length(sigma_et) == 1) Ht = array(diag(sigma_et^2, m0), c(m0, m0, T))
    if(length(sigma_et) == T) sigma_et = tcrossprod(sigma_et, rep(1,m0))
    for(j in 1:m0) Ht[j,j,] = sigma_et[,j]^2 # for(j in 1:m0) Ht[j,j,] = sigma_et^2
  }

  # We can either initialize the SSModel object, kfas_model (when NULL), or update the parameters
  if(is.null(kfas_model)){
    kfas_model = SSModel(Y.dlm ~ -1 + (SSMcustom(Z = Zt, T = Gt, Q = Wt, a1 = matrix(0, nrow = K0), P1 = W0, n = T, index = 1:m0)), H = Ht)
  } else {kfas_model$y = Y.dlm; kfas_model$Z = Zt; kfas_model$T = Gt; kfas_model$Q = Wt; kfas_model$P1 = W0; kfas_model$H = Ht}


  # Check for errors
  if(!is.SSModel(kfas_model)) stop("Error: Model has incorrect dimensions")

  # Return the SSModel object
  kfas_model
}
#####################################################################################################
# getSplineInfo() initializes (and transforms) the spline basis
# Inputs:
# tau01: all observation points, scaled to [0,1]
# m_avg (=NULL): average number of obs points; if NULL, set m_avg = length(tau01)
# orthonormal: orthonormalize the basis (TRUE/FALSE)
# Notes:
# Uses quantile-based placement of knots for a cubic spline basis
# Enfoces a penalty on the integrated squared second derivative
# Computes the matrix of integrals for the orthonormality constraints
# Transforms the basis, penalty, and matrix of integrals so that:
# d_k is decomposed into linear (first 2 elements) and nonlinear components
# the resulting prior for d_k is diagonal, which helps with posterior mixing, and proper
# Follows Wand and Ormerod (2008)
# Note: for orthonormalized, this is no longer true
#####################################################################################################
getSplineInfo = function(tau01, m_avg = NULL, orthonormal = TRUE){

  # This is the linear component, which appears in both cases
  X<-cbind(1, tau01)

  m = length(tau01);

  # Average number of observation points
  if(is.null(m_avg)) m_avg = m

  # Low-rank thin plate spline

  # Number of knots: if m > 25, use fewer
  if(m > 25){
    num.knots = max(20, min(ceiling(m_avg/4), 150))
  } else num.knots = max(3, ceiling(m_avg/2))

  knots<-quantile(unique(tau01), seq(0,1,length=(num.knots+2))[-c(1,(num.knots+2))])

  # SVD-type reparam (see Ciprian's paper)
  Z_K<-(abs(outer(tau01,knots,"-")))^3; OMEGA_all<-(abs(outer(knots,knots,"-")))^3
  svd.OMEGA_all<-svd(OMEGA_all)
  sqrt.OMEGA_all<-t(svd.OMEGA_all$v %*%(t(svd.OMEGA_all$u)*sqrt(svd.OMEGA_all$d)))

  # The nonlinear component:
  Z<-t(solve(sqrt.OMEGA_all,t(Z_K)))

  # Now combine the linear and nonlinear pieces:
  Bmat = cbind(X, Z);

  # The penalty matrix:
  Omega = diag(c(rep(10^-8, 2), rep(1, (ncol(Bmat) - 2))))

  if(orthonormal){
    # QR decomposition:
    qrd = qr(Bmat, complete = TRUE);  R.t = t(qr.R(qrd));
    # Update hte basis and the penalty matrix:
    Bmat = qr.Q(qrd); Omega = forwardsolve(R.t, t(forwardsolve(R.t, Omega, upper.tri = FALSE)), upper.tri = FALSE)

    BtB = diag(1, ncol(Bmat))
  } else BtB = crossprod(Bmat)

  # Return the matrix, the penalty, and the cross product (of the basis)
  list(Bmat = Bmat, Omega = Omega, BtB = BtB)
}
#' Construct the spline basis and penalty matrices
#'
#' Given input points in \code{d} dimensions, construct a low-rank thin plate spline basis matrix
#' and penalty matrix.
#'
#' @param tau \code{m x d} matrix of coordinates, where \code{m} is the number of observation points and \code{d} is the dimension
#' @param m_eff the effective number of observation points;
#' (e.g., the median number of observation points when missing observations)
#' @param orthonormalize logical; if TRUE, orthornomalize the basis matrix
#'
#' @note The knot locations are selected using a space-filling algorithm.
#'
#' @import fields
getSplineInfo_d = function(tau, m_eff = NULL, orthonormalize = TRUE){

  # Just in case, reform as matrix
  tau = as.matrix(tau)

  # Number of observation points
  m = nrow(tau)

  # Dimension:
  d = ncol(tau)

  # Order of derivative in penalty:
  m_deriv = 2

  # This is the linear component
  X = cbind(1, tau)

  # Number of effective observation points:
  if(is.null(m_eff)) m_eff = m

  # Number of knots: if more than 25 effective observation points, likely can use fewer knots
  if(m_eff > 25){
    # Guaranteed to be between 20 and 150 knots (but adjust as needed)
    num_knots = max(20, min(ceiling(m_eff/4), 150))
  } else num_knots = max(3, m_eff)

  # Locations of knots:
  if(num_knots < m){
    # Usual case: fewer knots than TOTAL observation points
    if(d == 1){
      # d = 1, just use quantiles of the observed data points:
      knots = as.matrix(quantile(unique(tau), seq(0,1,length=(num_knots+2))[-c(1,(num_knots+2))]))
    } else {
      # d > 1, use space-filling algorithm:
      knots = cover.design(tau, num_knots)$design
    }
  } else knots = tau

  # For the penalty matrix, need to compute distances between obs. points and knots
  dist.mat <- matrix(0, num_knots, num_knots); dist.mat[lower.tri(dist.mat)] <- dist(knots); dist.mat <- dist.mat + t(dist.mat)
  if(d%%2 == 0){
    # Even dim:
    Omega = dist.mat^(2*m_deriv - d)*log(dist.mat)
  } else {
    # Odd dim:
    Omega = dist.mat^(2*m_deriv - d)
  }
  # For numerical stability:
  diag(Omega) = 0

  # Compute the "random effects" matrix
  Zk = matrix(0, nrow=m, ncol=num_knots)
  for (k in 1:num_knots){
    di = sqrt(rowSums((tau - matrix(rep(knots[k,], each = m), nrow=m))^2)) # di = 0; for(j in 1:d) di = di + (tau[,j] - knots[k,j])^2; di = sqrt(di)
    if(d%%2 == 0){# Even dim:
      Zk[,k] = di^(2*m_deriv - d)*log(di)
    } else { # Odd dim:
      Zk[,k] = di^(2*m_deriv - d)
    }
  }
  Zk[is.nan(Zk)] = 0

  # Natural constraints, if necessary:
  if(num_knots > m - 1){Q2 = qr.Q(qr(X), complete=TRUE)[,-(1:2)]; Zk = Zk%*%Q2; Omega = crossprod(Q2, Omega)%*%Q2}

  # SVD of penalty matrix
  # So that the "random effects" have diagonal prior variance
  svd.Omega = svd(Omega)
  sqrt.Omega = t(svd.Omega$v %*%(t(svd.Omega$u)*sqrt(svd.Omega$d)))
  Z = t(solve(sqrt.Omega,t(Zk)))

  # Now combine the linear and nonlinear pieces to obtain the matrix of basis functions evaluated at the obs. points
  Bmat = cbind(X, Z);

  # The penalty matrix:
  Omega = diag(c(rep(0, ncol(X)), rep(1, ncol(Z))))

  if(orthonormalize){
    # QR decomposition:
    qrd = qr(Bmat, complete = TRUE);  R.t = t(qr.R(qrd));
    # Update hte basis and the penalty matrix:
    Bmat = qr.Q(qrd); Omega = forwardsolve(R.t, t(forwardsolve(R.t, Omega, upper.tri = FALSE)), upper.tri = FALSE)

    BtB = diag(1, ncol(Bmat))
  } else BtB = crossprod(Bmat)

  # Return the matrix, the penalty, and the cross product (of the basis)
  list(Bmat = Bmat, Omega = Omega, BtB = BtB)
}
#----------------------------------------------------------------------------
#' Compute Global Bayesian P-Values
#'
#' Given posterior samples for the loading curves \code{fk} and the
#' regression coefficient factors \code{alpha},
#' compute Global Bayesian P-Values for all regression coefficient functions
#'
#' @param post_fk \code{Nsims x m x K} matrix of posterior draws of the loading curve matrix
#' @param post_alpha \code{Nsims x p x K} matrix of posterior draws of the regression coefficient factors
#'
#' @return \code{p x 1} vector of Global Bayesian P-Values
#'
#' @export
fosr_gbpv = function(post_fk, post_alpha){
  p = dim(post_alpha)[2]
  gbpv = numeric(p)
  for(j in 1:p){
    post_alpha_tilde_j = get_post_alpha_tilde(post_fk, post_alpha[,j,])
    gbpv[j] = min(simBaS(post_alpha_tilde_j))
  }
  gbpv
}
#####################################################################################################
#' Compute Simultaneous Credible Bands
#'
#' Compute (1-alpha)\% credible BANDS for a function based on MCMC samples using Crainiceanu et al. (2007)
#'
#' @param sampFuns \code{Nsims x m} matrix of \code{Nsims} MCMC samples and \code{m} points along the curve
#' @param alpha confidence level
#'
#' @return \code{m x 2} matrix of credible bands; the first column is the lower band, the second is the upper band
#'
#' @note The input needs not be curves: the simultaneous credible "bands" may be computed
#' for vectors. The resulting credible intervals will provide joint coverage at the (1-alpha)%
#' level across all components of the vector.
#'
#' @export
credBands = function(sampFuns, alpha = .05){

  N = nrow(sampFuns); m = ncol(sampFuns)

  # Compute pointwise mean and SD of f(x):
  Efx = colMeans(sampFuns); SDfx = apply(sampFuns, 2, sd)

  # Compute standardized absolute deviation:
  Standfx = abs(sampFuns - tcrossprod(rep(1, N), Efx))/tcrossprod(rep(1, N), SDfx)

  # And the maximum:
  Maxfx = apply(Standfx, 1, max)

  # Compute the (1-alpha) sample quantile:
  Malpha = quantile(Maxfx, 1-alpha)

  # Finally, store the bands in a (m x 2) matrix of (lower, upper)
  cbind(Efx - Malpha*SDfx, Efx + Malpha*SDfx)
}
#####################################################################################################
#' Compute Simultaneous Band Scores (SimBaS)
#'
#' Compute simultaneous band scores (SimBaS) from Meyer et al. (2015, Biometrics).
#' SimBaS uses MC(MC) simulations of a function of interest to compute the minimum
#' alpha such that the joint credible bands at the alpha level do not include zero.
#' This quantity is computed for each grid point (or observation point) in the domain
#' of the function.
#'
#' @param sampFuns \code{Nsims x m} matrix of \code{Nsims} MCMC samples and \code{m} points along the curve
#'
#' @return \code{m x 1} vector of simBaS
#'
#' @note The input needs not be curves: the simBaS may be computed
#' for vectors to achieve a multiplicity adjustment.
#'
#' @note The minimum of the returned value, \code{PsimBaS_t},
#' over the domain \code{t} is the Global Bayesian P-Value (GBPV) for testing
#' whether the function is zero everywhere.
#'
#' @export
simBaS = function(sampFuns){

  N = nrow(sampFuns); m = ncol(sampFuns)

  # Compute pointwise mean and SD of f(x):
  Efx = colMeans(sampFuns); SDfx = apply(sampFuns, 2, sd)

  # Compute standardized absolute deviation:
  Standfx = abs(sampFuns - tcrossprod(rep(1, N), Efx))/tcrossprod(rep(1, N), SDfx)

  # And the maximum:
  Maxfx = apply(Standfx, 1, max)

  # And now compute the SimBaS scores:
  PsimBaS_t = rowMeans(sapply(Maxfx, function(x) abs(Efx)/SDfx <= x))

  # Alternatively, using a loop:
  #PsimBaS_t = numeric(T); for(t in 1:m) PsimBaS_t[t] = mean((abs(Efx)/SDfx)[t] <= Maxfx)

  PsimBaS_t
}
#####################################################################################################
#' Estimate the remaining time in the MCMC based on previous samples
#' @param nsi Current iteration
#' @param timer0 Initial timer value, returned from \code{proc.time()[3]}
#' @param nsims Total number of simulations
#' @param nrep Print the estimated time remaining every \code{nrep} iterations
#' @return Table of summary statistics using the function \code{summary}
computeTimeRemaining = function(nsi, timer0, nsims, nrep=100){

  # Only print occasionally:
  if(nsi%%nrep == 0 || nsi==20) {
    # Current time:
    timer = proc.time()[3]

    # Simulations per second:
    simsPerSec = nsi/(timer - timer0)

    # Seconds remaining, based on extrapolation:
    secRemaining = (nsims - nsi -1)/simsPerSec

    # Print the results:
    if(secRemaining > 3600) {
      print(paste(round(secRemaining/3600, 1), "hours remaining"))
    } else {
      if(secRemaining > 60) {
        print(paste(round(secRemaining/60, 2), "minutes remaining"))
      } else print(paste(round(secRemaining, 2), "seconds remaining"))
    }
  }
}
#----------------------------------------------------------------------------
#' Summarize of effective sample size
#'
#' Compute the summary statistics for the effective sample size (ESS) across
#' posterior samples for possibly many variables
#'
#' @param postX An array of arbitrary dimension \code{(nsims x ... x ...)}, where \code{nsims} is the number of posterior samples
#' @return Table of summary statistics using the function \code{summary()}.
#'
#' @examples
#' # ESS for iid simulations:
#' rand_iid = rnorm(n = 10^4)
#' getEffSize(rand_iid)
#'
#' # ESS for several AR(1) simulations with coefficients 0.1, 0.2,...,0.9:
#' rand_ar1 = sapply(seq(0.1, 0.9, by = 0.1), function(x) arima.sim(n = 10^4, list(ar = x)))
#' getEffSize(rand_ar1)
#'
#' @import coda
#' @export
getEffSize = function(postX) {
  if(is.null(dim(postX))) return(effectiveSize(postX))
  summary(effectiveSize(as.mcmc(array(postX, c(dim(postX)[1], prod(dim(postX)[-1]))))))
}
#----------------------------------------------------------------------------
#' Compute the ergodic (running) mean.
#' @param x vector for which to compute the running mean
#' @return A vector \code{y} with each element defined by \code{y[i] = mean(x[1:i])}
#' @examples
#' # Compare:
#' ergMean(1:10)
#' mean(1:10)
#'
#'# Running mean for iid N(5, 1) samples:
#' x = rnorm(n = 10^4, mean = 5, sd = 1)
#' plot(ergMean(x))
#' abline(h=5)
#' @export
ergMean = function(x) {cumsum(x)/(1:length(x))}
#----------------------------------------------------------------------------
#' Compute a block diagonal matrix w/ constant blocks
#'
#' The function returns kronecker(diag(nrep), Amat), but is computed more efficiently
#' @param Amat matrix to populate the diagaonal blocks
#' @param nrep number of blocks on the diagonal
#----------------------------------------------------------------------------
blockDiag = function(Amat, nrep){
  nr1 = nrow(Amat); nc1 = ncol(Amat)
  fullMat = matrix(0, nrow = nr1*nrep, ncol = nc1*nrep)
  rSeq = seq(1, nr1*nrep + nr1, by=nr1) # row sequence
  cSeq = seq(1, nc1*nrep + nc1, by=nc1) # col sequence
  for(i in 1:nrep) fullMat[rSeq[i]:(rSeq[i+1] - 1),  cSeq[i]:(cSeq[i+1] - 1)] = Amat

  fullMat
}
#----------------------------------------------------------------------------
#' Plot a curve given posterior samples
#'
#' Plot the posterior mean, simultaneous and pointwise 95\% credible bands
#' for a curve given draws from the posterior distribution
#'
#' @param post_f \code{Ns x m} matrix of \code{Ns} posterior simulations
#' of the curve at \code{m} points
#' @param tau \code{m x 1} vector of observation points
#' @param alpha confidence level for the bands
#' @param include_joint logical; if TRUE, include joint bands (as well as pointwise)
#' @param main title text (optional)
#' @param ylim range of y-axis (optional)
#'
#' @importFrom graphics abline lines par plot polygon
#' @import coda
#'
#' @export
plot_curve = function(post_f, tau = NULL, alpha = 0.05, include_joint = TRUE,
                      main = "Posterior Mean and Credible Bands", ylim = NULL){

  Ns = nrow(post_f); m = ncol(post_f)

  if(is.null(tau)) tau = 1:m

  #par(mfrow = c(1, 1), mai = c(1, 1, 1, 1))

  # Pointwise intervals:
  #dcip = dcib = HPDinterval(as.mcmc(post_f), prob = 1 - alpha);
  dcip = dcib = t(apply(post_f, 2, quantile, c(alpha/2, 1 - alpha/2)));

  # Joint intervals, if necessary:
  if(include_joint) dcib = credBands(post_f, alpha = alpha)

  f_hat = colMeans(post_f)

  plot(tau, f_hat, type = "n", ylim = range(dcib, dcip, ylim, na.rm = TRUE),
       xlab = expression(tau), ylab = "", main = main,
       cex.lab = 1.5, cex.main = 1.5, cex.axis = 1.5)
  if(include_joint) polygon(c(tau, rev(tau)), c(dcib[, 2], rev(dcib[, 1])), col = "gray50",
                            border = NA)
  polygon(c(tau, rev(tau)), c(dcip[, 2], rev(dcip[, 1])), col = "grey",
          border = NA)
  lines(tau, f_hat, lwd = 8, col = "cyan")
}
#----------------------------------------------------------------------------
#' Plot the factors
#'
#' Plot posterior mean of the factors together with the simultaneous and pointwise
#' 95\% credible bands.
#'
#' @param post_beta the \code{Nsims x T x K} array of \code{Nsims} draws from the posterior
#' distribution of the \code{T x K} matrix of factors, \code{beta}
#' @param dates \code{T x 1} vector of dates or labels corresponding to the time points
#'
#' @importFrom grDevices dev.new
#' @importFrom graphics abline lines  par plot polygon
#' @import coda
#' @export
plot_factors = function(post_beta, dates = NULL){
  K = dim(post_beta)[3] # Number of factors
  if(is.null(dates)) dates = seq(0, 1, length.out = dim(post_beta)[2])

  #dev.new();
  par(mai = c(.8,.9,.4,.4), bg = 'gray90');
  plot(dates, post_beta[1,,1], ylim = range(post_beta), xlab = 'Dates', ylab = '', main = paste('Dynamic Factors', sep=''), type='n', cex.lab = 2, cex.axis=2,cex.main=2)
  abline(h = 0, lty=3, lwd=2);
  for(k in K:1){
    cb = credBands(as.mcmc(post_beta[,,k])); ci = HPDinterval(as.mcmc(post_beta[,,k]));
    polygon(c(dates, rev(dates)), c(cb[,2], rev(cb[,1])), col='grey50', border=NA);
    polygon(c(dates, rev(dates)), c(ci[,2], rev(ci[,1])), col='grey', border=NA);
    lines(dates,colMeans(post_beta[,,k]), lwd=8, col=k)
  }
}
#----------------------------------------------------------------------------
#' Plot the factor loading curves
#'
#' Plot posterior mean of the factor loading curves together with the simultaneous
#' and pointwise 95\% credible bands.
#'
#' @param post_fk the \code{Nsims x m x K} array of \code{Nsims} draws from the posterior
#' distribution of the \code{m x K} matrix of FLCs, \code{fk}
#' @param tau \code{m x 1} vector of observation points
#'
#' @importFrom graphics abline lines  par plot polygon
#' @import coda
#' @export
plot_flc = function(post_fk, tau = NULL){
  K = dim(post_fk)[3] # Number of factors
  if(is.null(tau)) tau = seq(0, 1, length.out = dim(post_fk)[2])

  #dev.new();
  par(mai = c(.9,.9,.4,.4), bg = 'gray90');
  plot(tau, post_fk[1,,1], ylim = range(post_fk), xlab = expression(tau), ylab = '', main = 'Factor Loading Curves', type='n', cex.lab = 2, cex.axis=2,cex.main=2)
  abline(h = 0, lty=3, lwd=2);
  for(k in K:1){
    # Credible intervals:
    ci = HPDinterval(as.mcmc(post_fk[,,k]));
    # Credible bands (w/ error catch):
    cb = try(credBands(as.mcmc(post_fk[,,k])), silent = TRUE)
    if(class(cb) == "try-error") cb = ci
    polygon(c(tau, rev(tau)), c(cb[,2], rev(cb[,1])), col='grey50', border=NA);
    polygon(c(tau, rev(tau)), c(ci[,2], rev(ci[,1])), col='grey', border=NA);
    lines(tau,colMeans(post_fk[,,k]), lwd=8, col=k)
  }
}
#----------------------------------------------------------------------------
#' Plot the Bayesian curve fitted values
#'
#' Plot the curve posterior means with posterior credible intervals (pointwise and joint),
#' the observed data, and true curves (if known)
#'
#' @param y the \code{T x 1} vector of time series observations
#' @param mu the \code{T x 1} vector of fitted values, i.e., posterior expectation of the mean
#' @param postY the \code{nsims x T} matrix of posterior draws from which to compute intervals
#' @param y_true the \code{T x 1} vector of points along the true curve
#' @param t01 the observation points; if NULL, assume \code{T} equally spaced points from 0 to 1
#' @param include_joint_bands logical; if TRUE, compute simultaneous credible bands
#'
#' @import coda
#' @export
plot_fitted = function(y, mu, postY, y_true = NULL, t01 = NULL, include_joint_bands = FALSE){

  # Time series:
  T = length(y);
  if(is.null(t01)) t01 = seq(0, 1, length.out=T)

  # Credible intervals/bands:
  #dcip = HPDinterval(as.mcmc(postY)); dcib = credBands(postY)
  dcip = dcib = t(apply(postY, 2, quantile, c(0.05/2, 1 - 0.05/2)));
  if(include_joint_bands) dcib = credBands(postY)

  # Plot
  #dev.new();
  par(mfrow=c(1,1), mai = c(1,1,1,1))
  plot(t01, y, type='n', ylim=range(dcib, y, na.rm=TRUE), xlab = 't', ylab=expression(paste("y"[t])), main = 'Fitted Values: Conditional Expectation', cex.lab = 2, cex.main = 2, cex.axis = 2)
  polygon(c(t01, rev(t01)), c(dcib[,2], rev(dcib[,1])), col='gray50', border=NA)
  polygon(c(t01, rev(t01)), c(dcip[,2], rev(dcip[,1])), col='grey', border=NA)
  if(!is.null(y_true))  lines(t01, y_true, lwd=8, col='black', lty=6);
  lines(t01, y, type='p');
  lines(t01, mu, lwd=8, col = 'cyan');
}
#----------------------------------------------------------------------------
#' Univariate Slice Sampler from Neal (2008)
#'
#' Compute a draw from a univariate distribution using the code provided by
#' Radford M. Neal. The documentation below is also reproduced from Neal (2008).
#'
#' @param x0    Initial point
#' @param g     Function returning the log of the probability density (plus constant)
#' @param w     Size of the steps for creating interval (default 1)
#' @param m     Limit on steps (default infinite)
#' @param lower Lower bound on support of the distribution (default -Inf)
#' @param upper Upper bound on support of the distribution (default +Inf)
#' @param gx0   Value of g(x0), if known (default is not known)
#'
#' @return  The point sampled, with its log density attached as an attribute.
#'
#' @note The log density function may return -Inf for points outside the support
#' of the distribution.  If a lower and/or upper bound is specified for the
#' support, the log density function will not be called outside such limits.
uni.slice <- function (x0, g, w=1, m=Inf, lower=-Inf, upper=+Inf, gx0=NULL)
{
  # Check the validity of the arguments.

  if (!is.numeric(x0) || length(x0)!=1
      || !is.function(g)
      || !is.numeric(w) || length(w)!=1 || w<=0
      || !is.numeric(m) || !is.infinite(m) && (m<=0 || m>1e9 || floor(m)!=m)
      || !is.numeric(lower) || length(lower)!=1 || x0<lower
      || !is.numeric(upper) || length(upper)!=1 || x0>upper
      || upper<=lower
      || !is.null(gx0) && (!is.numeric(gx0) || length(gx0)!=1))
  {
    stop ("Invalid slice sampling argument")
  }

  # Keep track of the number of calls made to this function.
  #uni.slice.calls <<- uni.slice.calls + 1

  # Find the log density at the initial point, if not already known.

  if (is.null(gx0))
  { #uni.slice.evals <<- uni.slice.evals + 1
    gx0 <- g(x0)
  }

  # Determine the slice level, in log terms.

  logy <- gx0 - rexp(1)

  # Find the initial interval to sample from.

  u <- runif(1,0,w)
  L <- x0 - u
  R <- x0 + (w-u)  # should guarantee that x0 is in [L,R], even with roundoff

  # Expand the interval until its ends are outside the slice, or until
  # the limit on steps is reached.

  if (is.infinite(m))  # no limit on number of steps
  {
    repeat
    { if (L<=lower) break
      #uni.slice.evals <<- uni.slice.evals + 1
      if (g(L)<=logy) break
      L <- L - w
    }

    repeat
    { if (R>=upper) break
      #uni.slice.evals <<- uni.slice.evals + 1
      if (g(R)<=logy) break
      R <- R + w
    }
  }

  else if (m>1)  # limit on steps, bigger than one
  {
    J <- floor(runif(1,0,m))
    K <- (m-1) - J

    while (J>0)
    { if (L<=lower) break
      #uni.slice.evals <<- uni.slice.evals + 1
      if (g(L)<=logy) break
      L <- L - w
      J <- J - 1
    }

    while (K>0)
    { if (R>=upper) break
      #uni.slice.evals <<- uni.slice.evals + 1
      if (g(R)<=logy) break
      R <- R + w
      K <- K - 1
    }
  }

  # Shrink interval to lower and upper bounds.

  if (L<lower)
  { L <- lower
  }
  if (R>upper)
  { R <- upper
  }

  # Sample from the interval, shrinking it on each rejection.

  repeat
  {
    x1 <- runif(1,L,R)

    #uni.slice.evals <<- uni.slice.evals + 1
    gx1 <- g(x1)

    if (gx1>=logy) break

    if (x1>x0)
    { R <- x1
    }
    else
    { L <- x1
    }
  }

  # Return the point sampled, with its log density attached as an attribute.

  attr(x1,"log.density") <- gx1
  return (x1)

}

.onUnload <- function (libpath) {
  library.dynam.unload("dfosr", libpath)
}

# Just add these for general use:
#' @importFrom stats predict quantile rgamma rnorm sd splinefun var rexp runif arima arima.sim dbeta dgamma dist dnorm dunif fitted lm median poly rbinom dnbinom rnbinom rpois
NULL

#' @useDynLib dfosr
#' @importFrom Rcpp sourceCpp
NULL
