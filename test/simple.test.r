library(multivariate.sv)
NR <- 20
NC <- 5
X <- matrix(rnorm(NR*NC),NR,NC)

res <- multivariate.sv(X,iterations=1e4L, burn=1e4L, adapt=1e3L, thin=5L)


