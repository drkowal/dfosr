% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/component_samplers.R
\name{sampleARmu}
\alias{sampleARmu}
\title{Sample the unconditional mean in an AR(1) Model}
\usage{
sampleARmu(yt, phi_j, sigma_tj, priorPrec = NULL)
}
\arguments{
\item{yt}{the \code{T x p} matrix of multivariate time series}

\item{phi_j}{the \code{p x 1} vector of AR(1) coefficients}

\item{sigma_tj}{the \code{(T-1) x p} matrix or \code{p x 1} vector of error standard deviations}

\item{priorPrec}{the \code{p x 1} vector of prior precisions;
if \code{NULL}, use \code{rep(10^-6, p)}}
}
\value{
The \code{p x 1} matrix of unconditional means.
}
\description{
Compue one draw of the unconditional mean \code{mu} in an AR(1) model assuming a
Gaussian prior (with mean zero).
}
\details{
Sample the unconditional mean \code{mu} using the model

\code{y_tj = mu_j + phi_j(y_{t-1,j} - mu_j) + e_tj},

with \code{e_tj ~ N(0, sigma[j]^2)} and prior \code{mu ~ N(0, 1/priorPrec[j])}
}
