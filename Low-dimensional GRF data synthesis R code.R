library(grf)
library(faux)


########################## @ setting ##########################
n_samples = 2000 # number of observations
n_features = 10  # number of covariates

set.seed(12)
X = matrix(data=rnorm(n=n_samples*n_features, mean=0, sd=1), nrow=n_samples, ncol=n_features) # covariate matrix
W = rbinom(n=n_samples, size=1, prob=1/(1+exp(-rnorm(n=n_samples, mean=0, sd=1)))) # binary treatment, uncorrelated with X

set.seed(123)
e = rnorm(n=n_samples, mean=0, sd=1) # gaussian noise 1, used in Y
set.seed(1234)
h = rnorm(n=n_samples, mean=0, sd=1) # gaussian noise 2, used in mu



########################## @ no TE & no HTE ##########################
set.seed(12345)
mu = 2*h
Y = mu + e

cor(Y,W); cor(mu,X); cor(Y,X)



########################## @ TE & no HTE ##########################
set.seed(12345)
mu = rnorm_pre(W, mu=0, sd=1, r=0.5) + h 
mu = as.matrix(mu, nrow=n_samples)
Y = rowMeans(mu %*% W) + e

cor(Y,W); cor(mu,X); cor(Y,X)



########################## @ TE & HTE (only protective factors) ##########################
set.seed(12345)
mu = rnorm_pre(W, mu=0, sd=1, r=0.5) + rowMeans(X[, 1:2]) + h 
Y = mu*W + e

cor(Y,W); cor(mu,X); cor(Y,X)



########################## @ TE & HTE (only risk factors) ##########################
set.seed(12345)
mu = rnorm_pre(W, mu=0, sd=1, r=0.5) - rowMeans(X[, 3:4]) + h 
Y = mu*W + e

cor(Y,W); cor(mu,X); cor(Y,X)



########################## @ TE & HTE (both protective & risk factors) ##########################
set.seed(12345)
mu = rnorm_pre(W, mu=0, sd=1, r=0.5) + rowMeans(X[, 1:2]) - rowMeans(X[, 3:4]) + h 
Y = mu*W + e

cor(Y,W); cor(mu,X); cor(Y,X)



########################## @ GRF code ##########################
# z-scaling of continuous variables
Y = as.vector(scale(Y))
for(i in 1:ncol(X)){X[,i] = as.vector(scale(X[,i]))}


# grow five forests using the set of seeds
num_seeds = 5
seeds = c(1419, 8475, 7150, 7847, 2108) # create using "sample(1:10000, size=num_seeds, replace=FALSE)"
# seeds =  c(1532, 3980, 1818, 7652, 7973) # check seed-robustness
# seeds = c(6052, 8153, 6478, 1936, 8322)  # check seed-robustness
forest_list = list()

for(seed_ind in 1:length(seeds)){
  print(paste('seed_ind', seed_ind, sep=' = '))
  forest = causal_forest(X, Y, W, 
                         Y.hat=NULL, 
                         W.hat=NULL, 
                         compute.oob.predictions = FALSE, 
                         #clusters = site_id,
                         num.trees = 2000, 
                         tune.parameters = 'all',
                         num.threads = 16,
                         seed = seeds[seed_ind])
  forest_list[[paste0('seed', seed_ind)]] = forest
}
forest_merged = merge_forests(forest_list, compute.oob.predictions = TRUE)


# ATE estimation
ate = average_treatment_effect(forest_merged, method='AIPW', target.sample='all')
ate_t = ate[1] / ate[2]
ate_p = 1.96 * (1 - pnorm(abs(ate_t)))
ate_ci_low = ate[1] - 1 * qnorm(0.975) * ate[2] 
ate_ci_high = ate[1] + 1 * qnorm(0.975) * ate[2] 
ate_results = data.frame(ate[1], ate[2], ate_t, ate_p, ate_ci_low, ate_ci_high)
rownames(ate_results) = 'ATE'
colnames(ate_results) = c("estimate", "std.error", "t-score", "P-value", "[95% ", " CI]")
print(ate_results)


# performance of the causal forest
test_cal = test_calibration(forest_merged)
print(test_cal)


# feature importance
var_imp = c(variable_importance(forest_merged))
names(var_imp) = as.character(seq(1:10))
sorted_var_imp = sort(var_imp, decreasing=TRUE)
sorted_var_imp = data.frame(variable=names(sorted_var_imp), importance=sorted_var_imp)


# best_linear_projection
best_linear_projection(forest_merged, X)

