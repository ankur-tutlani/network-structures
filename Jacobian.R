# Load the necessary library (if not already loaded)
#install.packages("matlib")  # Uncomment this line if you haven't installed the 'matlib' package
rm(list = ls())
library(matlib)

# Define the equilibrium point and parameters. Parameters imply the payoff values in the matrix. 
equilibrium <- c(0.5,0.5)
parameters <- c(1,0.9, 0.9, 1)

a <- parameters
states <- sqrt(length(a))
A <- matrix(a, states, byrow = TRUE)
A <- t(A)
A


Replicator <- function(state, parameters) {
  a <- parameters
  states <- sqrt(length(a))
  A <- matrix(a, states, byrow = TRUE)
  A <- t(A)
  
  dX <- c()
  
  for(i in 1:states) {
    dX[i] <- sum(state * A[i, ])
  }
  
  avgFitness <- sum(dX * state)
  
  for(i in 1:states) {
    dX[i] <- state[i] * (dX[i] - avgFitness)
  }
  
  return(list(dX))
}

f<- Replicator


# Compute the Jacobian matrix
delta <- 1e-6
J <- matrix(0, nrow = 2, ncol = 2)
for (i in 1:2) {
  for (j in 1:2) {
    J[i, j] <- (f(state = equilibrium + delta * diag(2)[j, ], parameters = parameters)[[1]][i] -
                  f(state = equilibrium, parameters = parameters)[[1]][i]) / delta
  }
}

cat("Jacobian matrix at equilibrium:\n")
print(J)

# Compute the eigenvalues
eigenvalues <- eigen(J)$values
cat("Eigenvalues:", eigenvalues, "\n")



Logit <- function(state, parameters) {
  eta <- parameters[length(parameters)]
  
  a <- parameters[-length(parameters)]
  states <- sqrt(length(a))
  A <- matrix(a, states, byrow = TRUE)
  A <- t(A)
  
  dX <- c()
  
  for(i in 1:states) {
    dX[i] <- sum(state * A[i, ])
  }
  
  etaVals <- c()
  
  for(i in 1:states) {
    etaVals <- sum(etaVals, exp(eta^(-1) * dX[i]))
  }
  
  if(is.infinite(etaVals)) {
    stop("Due to internal restrictions of R, please choose a greater value of 
         eta.")
  }
  
  for(i in 1:states) {
    dX[i] <- (exp(eta^(-1) * dX[i])) / etaVals - state[i]
  }
  
  return(list(dX))
}

### the last parameter value is the eta value of the logit function. 
equilibrium <- c(0.5, 0.5)
parameters <- c(1,0.9, 0.9, 1,0.99)

a <- parameters
states <- sqrt(length(a))
A <- matrix(a, states, byrow = TRUE)
A <- t(A)
A


f<- Logit


# Compute the Jacobian matrix
delta <- 1e-6
J <- matrix(0, nrow = 2, ncol = 2)
for (i in 1:2) {
  for (j in 1:2) {
    J[i, j] <- (f(state = equilibrium + delta * diag(2)[j, ], parameters = parameters)[[1]][i] -
                  f(state = equilibrium, parameters = parameters)[[1]][i]) / delta
  }
}


# Print the Jacobian matrix at equilibrium
cat("Jacobian matrix at equilibrium:\n")
print(J)

# Compute the eigenvalues
eigenvalues <- eigen(J)$values
cat("Eigenvalues:", eigenvalues, "\n")
