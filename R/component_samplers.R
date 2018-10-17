#' Factor Loading Curve Sampling Algorithm
#'
#' Sample the factor loading curve basis coefficients subject to an orthonormality constraint.
#' Additional linear constraints may be included as well.
#'
#' @param BtY \code{J x T} matrix \code{B.t()*Y} for basis matrix B
#' @param Beta \code{T x K} matrix of factors
#' @param Psi \code{J x K} matrix of previous factor loading curve coefficients
#' @param BtB \code{J x J} matrix of \code{B.t()*B}
#' @param Omega \code{J x J} prior precision/penalty matrix
#' @param BtCon (optional) \code{J x Jc} matrix of additional constraints, pre-multiplied by B.t()
#' @param lambda \code{K}-dimensional vector of prior precisions
#' @param sigmat2 \code{T}-dimensional vector of time-dependent observation error variances
#' @return Psi \code{J x K} matrix of factor loading curve coefficients
#'
#' @note This is a wrapper for Rcpp functions for the special cases of
#' \code{K = 1} and whether or not additional (linear) constraints are included,
#' i.e., whether or not \code{BtCon} is non-\code{NULL}.
#' @export
fdlm_flc = function(BtY, Beta, Psi, BtB, Omega, BtCon = NULL, lambda, sigmat2){

  # Obtain the dimensions, in order of appearance:
  J = nrow(BtY); T = ncol(BtY); K = ncol(Beta);

  # Allow for scalar variance input
  if(length(sigmat2) == 1) sigmat2 = rep(sigmat2, T)

  # Check dimensions:
  if( (nrow(Beta) != T) ||
      (nrow(Psi) != J) || (ncol(Psi) != K) ||
      (nrow(BtB) != J) || (ncol(BtB) != J) ||
      (nrow(Omega) != J) || (ncol(Omega) != J) ||
      (length(lambda) != K) ||
      (length(sigmat2) != T)
  ) stop("Mismatched dimensions in FLC sampler")

  # No additional constraints (besides orthonormality of FLCs themselves)
  if(is.null(BtCon)){

    if(K == 1){
      # Special case: (FLC) orthogonality not necessary
      Psi = sampleFLC_1(BtY = BtY, Beta = Beta, Psi = Psi, BtB = BtB, Omega = Omega, lambda = lambda, sigmat2 = sigmat2)
    } else {
      Psi = sampleFLC(BtY = BtY, Beta = Beta, Psi = Psi, BtB = BtB, Omega = Omega, lambda = lambda, sigmat2 = sigmat2)
    }


  } else {
    # Additional constraints: orthogonal to BtCon


    # Special case: (FLC) orthogonality not necessary
    if(K == 1){
      Psi = sampleFLC_cons_1(BtY = BtY, Beta = Beta, Psi = Psi, BtB = BtB, Omega = Omega, BtCon = BtCon, lambda = lambda, sigmat2 = sigmat2)
    } else {
      Psi = sampleFLC_cons(BtY = BtY, Beta = Beta, Psi = Psi, BtB = BtB, Omega = Omega, BtCon = BtCon, lambda = lambda, sigmat2 = sigmat2)
    }

  }
}
#' Factor loading curve smoothing parameter sampler
#'
#' Sample the smoothing parameters for each factor loading curve.
#'
#' @param lambda \code{K}-dimensional vector of smoothing parameters (prior precisions)
#' from previous MCMC iteration
#' @param Psi \code{J x K} matrix of basis coefficients, where \code{J} is the number of
#' basis functions and \code{K} is the number of factors
#' @param Omega \code{J x J} penalty matrix; if NULL, assume it is diag(0, 0, 1,...,1)
#' @param d dimension of \code{tau}; default is 1
#' @param uniformPrior logical; when TRUE, use a uniform prior on prior standard deviations,
#' \code{1/sqrt{lambda[k]}}; otherwise use independent Gamma(0.001, 0.001) prior for each \code{lambda[k]}
#' @param orderLambdas logical; when TRUE, enforce the ordering constraint \code{lambda[1] > ... > lambda[K]}
#' for identifiability

#' @return The \code{K}-dimensional vector of samoothing parameters, \code{lambda}.
#####################################################################################################
#' @export
sample_lambda = function(lambda, Psi, Omega = NULL, d = 1, uniformPrior = TRUE, orderLambdas = TRUE){
  J = nrow(Psi); K = ncol(Psi)

  if(uniformPrior){shape0 = (J - d + 1 + 1)/2} else shape0 = (J - d - 1)/2 + 0.001; # for Gamma(0.001, 0.001) prior
  #if(uniformPrior){shape0 = (J + 1)/2} else shape0 = (J - 2)/2 + 0.001; # for Gamma(0.001, 0.001) prior

  for(k in 1:K){
    if(is.null(Omega)){rate0 = crossprod(Psi[-(1:(d+1)),k])/2} else rate0 = crossprod(Psi[,k], Omega)%*%Psi[,k]/2
    #if(is.null(Omega)){rate0 = crossprod(Psi[-(1:2),k])/2} else rate0 = crossprod(Psi[,k], Omega)%*%Psi[,k]/2

    if(!uniformPrior) rate0 = rate0 + 0.001  # for Gamma(0.001, 0.001) prior

    # Lower and upper bounds, w/ ordering constraints (if specified):
    if(orderLambdas){
      lam.l = 10^-8; lam.u = Inf; if(k != 1) lam.u = lambda[k-1];  # if(k != K) lam.l = lambda[k+1];
      lambda[k] = truncdist::rtrunc(1, 'gamma', a=lam.l, b=lam.u, shape=shape0, rate=rate0) # more stable, possibly faster
    } else lambda[k] = rgamma(1, shape = shape0, rate = rate0)
  }
  lambda
}
#' Sample the autoregressive coefficients in an AR(1) Model
#'
#' Compue one draw of the autoregressive coefficients \code{phi} in an AR(1) model.
#' The sampler also applies to a multivariate case with independent components.
#'
#' Sample the AR(1) coefficients \code{phi_j} using the model
#'
#' \code{y_tj = mu_j + phi_j(y_{t-1,j} - mu_j) + e_tj},
#'
#' with \code{e_tj ~ N(0, sigma[j]^2)}
#'
#' @param yt the \code{T x p} matrix of centered multivariate time series
#' (i.e., the time series minus the unconditional means, \code{mu})
#' @param phi_j the \code{p x 1} vector of previous AR(1) coefficients
#' @param sigma_tj the \code{(T-1) x p} matrix or \code{p x 1} vector of error standard deviations
#' @param prior_phi the parameters of the prior for the AR(1) coefficients \code{phi_j};
#' either \code{NULL} for uniform on [-0.99,0.99] or a 2-dimensional vector of (shape1, shape2) for a Beta prior
#' on \code{[(phi_j + 1)/2]}
#'
#' @return \code{p x 1} vector of sampled AR(1) coefficient(s)
#'
#' @note For the standard AR(1) case, \code{p = 1}. However, the function applies more
#' generally for sampling \code{p > 1} independent AR(1) processes (jointly).
#'
#' @import truncdist
#' @export
sampleARphi = function(yt, phi_j, sigma_tj, prior_phi = NULL){

  # Just in case:
  yt = as.matrix(yt)

  # Store dimensions locally:
  T = nrow(yt); p = ncol(yt)

  if(length(sigma_tj) == p) sigma_tj = matrix(rep(sigma_tj, each = T-1), nrow = T-1)

  # Loop over the j=1:p
  for(j in 1:p){

    # Compute "regression" terms for dhs_phi_j:
    y_ar = yt[-1,j]/sigma_tj[,j] # Standardized "response"
    x_ar = yt[-T,j]/sigma_tj[,j] # Standardized "predictor"

    # Using Beta distribution:
    if(!is.null(prior_phi)){

      # Check to make sure the prior params make sense
      if(length(prior_phi) != 2) stop('prior_phi must be a numeric vector of length 2')

      phi01 = (phi_j[j] + 1)/2 # ~ Beta(prior_phi[1], prior_phi[2])

      # Slice sampler when using Beta prior:
      phi01 = uni.slice(phi01, g = function(x){
        -0.5*sum((y_ar - (2*x - 1)*x_ar)^2) +
          dbeta(x, shape1 = prior_phi[1], shape2 = prior_phi[2], log = TRUE)
      }, lower = 0.005, upper = 0.995)[1]

      phi_j[j] = 2*phi01 - 1

    } else {
      # For phi_j ~ Unif(-0.99, 0.99), the posterior is truncated normal
      phi_j[j] = rtrunc(n = 1, spec = 'norm',
                        a = -0.99, b = 0.99,
                        mean = sum(y_ar*x_ar)/sum(x_ar^2),
                        sd = 1/sqrt(sum(x_ar^2)))
    }
  }
  return(phi_j)
}
#' Sample the unconditional mean in an AR(1) Model
#'
#' Compue one draw of the unconditional mean \code{mu} in an AR(1) model assuming a
#' Gaussian prior (with mean zero).
#'
#' Sample the unconditional mean \code{mu} using the model
#'
#' \code{y_tj = mu_j + phi_j(y_{t-1,j} - mu_j) + e_tj},
#'
#' with \code{e_tj ~ N(0, sigma[j]^2)} and prior \code{mu ~ N(0, 1/priorPrec[j])}
#'
#' @param yt the \code{T x p} matrix of multivariate time series
#' @param phi_j the \code{p x 1} vector of AR(1) coefficients
#' @param sigma_tj the \code{(T-1) x p} matrix or \code{p x 1} vector of error standard deviations
#' @param priorPrec the \code{p x 1} vector of prior precisions;
#' if \code{NULL}, use \code{rep(10^-6, p)}
#'
#' @return The \code{p x 1} matrix of unconditional means.
#' @export
sampleARmu = function(yt, phi_j, sigma_tj, priorPrec = NULL){

  # Just in case:
  yt = as.matrix(yt)

  # Store dimensions locally:
  T = nrow(yt); p = ncol(yt)

  if(length(sigma_tj) == p) sigma_tj = matrix(rep(sigma_tj, each = T-1), nrow = T-1)

  # Prior precision:
  if(is.null(priorPrec)) priorPrec = rep(10^-6, p)

  # Now, form the "y" and "x" terms in the (auto)regression
  y_mu = (yt[-1,] - matrix(rep(phi_j, each = T-1), nrow = T-1)*yt[-T,])/sigma_tj
  x_mu = matrix(rep(1 - phi_j, each = T-1), nrow = T-1)/sigma_tj

  # Posterior SD and mean:
  postSD = 1/sqrt(colSums(x_mu^2) + priorPrec)
  postMean = (colSums(x_mu*y_mu))*postSD^2
  mu = rnorm(n = p, mean = postMean, sd = postSD)

  return(mu)
}
#--------------------------------------------------------------
#' Multiplicative Gamma Process (MGP) Sampler
#'
#' Sample the global parameters, delta.h, with tau.h = cumprod(delta.h), from
#' the MGP prior for factor models
#'
#' @param theta.jh the \code{p x K} matrix with entries theta.jh ~ N(0, tau.h^-1), j=1:p, h=1:K
#' @param delta.h the \code{K}-dimensional vector of previous delta.h values, tau.h[h] = prod(delta.h[1:h])
#' @param a1 the prior parameter for factor 1: delta.h[1] ~ Gamma(a1, 1)
#' @param a2 the prior parameter for factors 2:K: delta.h[h] ~ Gamma(a2, 1) for h = 2:K
#' @return \code{delta.h}, the \code{K}-dimensional vector of multplicative components,
#' where tau.h[h] = prod(delta.h[1:h])
#'
#' @note The default \code{a1 = 2} and \code{a2 = 3} appears to offer the best performance
#' in Durante (2017).
#' @export
sampleMGP = function(theta.jh, delta.h, a1 = 2, a2 = 3){

  # Just in case:
  theta.jh = as.matrix(theta.jh)

  # Store the dimensions locally
  p = nrow(theta.jh); K = ncol(theta.jh)

  # Sum over the (squared) replicates:
  sum.theta.l = colSums(theta.jh^2)

  # h = 1 case is separate:
  tau.not.1 = cumprod(delta.h)/delta.h[1]
  delta.h[1] = rgamma(n = 1, shape = a1 + p*K/2,
                      rate = 1 + 1/2*sum(tau.not.1*sum.theta.l))
  # h > 1:
  if(K > 1){for(h in 2:K){
    tau.not.h = cumprod(delta.h)/delta.h[h]
    delta.h[h] = rgamma(n = 1, shape = a2 + p/2*(K - h + 1),
                        rate = 1 + 1/2*sum(tau.not.h[h:K]*sum.theta.l[h:K]))
  }}
  delta.h #list(tau.h = cumprod(delta.h), delta.h = delta.h)
}
#----------------------------------------------------------------------------
#' Sample a Gaussian vector using the fast sampler of BHATTACHARYA et al.
#'
#' Sample from N(mu, Sigma) where Sigma = solve(crossprod(Phi) + solve(D))
#' and mu = Sigma*crossprod(Phi, alpha):
#'
#' @param Phi \code{n x p} matrix (of predictors)
#' @param Ddiag \code{p x 1} vector of diagonal components (of prior variance)
#' @param alpha \code{n x 1} vector (of data, scaled by variance)
#' @return Draw from N(mu, Sigma), which is \code{p x 1}, and is computed in \code{O(n^2*p)}
#' @note Assumes D is diagonal, but extensions are available
#' @export
sampleFastGaussian = function(Phi, Ddiag, alpha){
  
  # Dimensions:
  Phi = as.matrix(Phi); n = nrow(Phi); p = ncol(Phi)
  
  # Step 1:
  u = rnorm(n = p, mean = 0, sd = sqrt(Ddiag))
  delta = rnorm(n = n, mean = 0, sd = 1)
  
  # Step 2:
  v = Phi%*%u + delta
  
  # Step 3:
  w = solve(crossprod(sqrt(Ddiag)*t(Phi)) + diag(n), #Phi%*%diag(Ddiag)%*%t(Phi) + diag(n)
            alpha - v)
  
  # Step 4:
  theta =  u + Ddiag*crossprod(Phi, w)
  
  # Return theta:
  theta
}