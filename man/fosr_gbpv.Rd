% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helper_functions.R
\name{fosr_gbpv}
\alias{fosr_gbpv}
\title{Compute Global Bayesian P-Values}
\usage{
fosr_gbpv(post_fk, post_alpha)
}
\arguments{
\item{post_fk}{\code{Nsims x m x K} matrix of posterior draws of the loading curve matrix}

\item{post_alpha}{\code{Nsims x p x K} matrix of posterior draws of the regression coefficient factors}
}
\value{
\code{p x 1} vector of Global Bayesian P-Values
}
\description{
Given posterior samples for the loading curves \code{fk} and the
regression coefficient factors \code{alpha},
compute Global Bayesian P-Values for all regression coefficient functions
}
