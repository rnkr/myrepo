# 1. Deleting the observations
lm(medv ~ ptratio + rad, data=BostonHousing, na.action=na.omit)

2. Deleting the variable

3. Imputation with mean / median / mode
library(Hmisc)
impute(BostonHousing$ptratio, mean)  # replace with mean
impute(BostonHousing$ptratio, median)  # median
impute(BostonHousing$ptratio, 20)  # replace specific number
# or if you want to impute manually
BostonHousing$ptratio[is.na(BostonHousing$ptratio)] <- mean(BostonHousing$ptratio, na.rm = T)  # not run

#Lets compute the accuracy when it is imputed with mean

library(DMwR)
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- rep(mean(BostonHousing$ptratio, na.rm=T), length(actuals))
regr.eval(actuals, predicteds)

# 4 . Prediction
## 4.1. kNN Imputation

library(DMwR)
knnOutput <- knnImputation(BostonHousing[, !names(BostonHousing) %in% "medv"])  # perform knn imputation.
anyNA(knnOutput)
#> FALSE

#Lets compute the accuracy.

actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- knnOutput[is.na(BostonHousing$ptratio), "ptratio"]
regr.eval(actuals, predicteds)


# 4.2 rpart
library(rpart)
class_mod <- rpart(rad ~ . - medv, data=BostonHousing[!is.na(BostonHousing$rad), ], method="class", na.action=na.omit)  # since rad is a factor
anova_mod <- rpart(ptratio ~ . - medv, data=BostonHousing[!is.na(BostonHousing$ptratio), ], method="anova", na.action=na.omit)  # since ptratio is numeric.
rad_pred <- predict(class_mod, BostonHousing[is.na(BostonHousing$rad), ])
ptratio_pred <- predict(anova_mod, BostonHousing[is.na(BostonHousing$ptratio), ])

#Lets compute the accuracy for ptratio
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- ptratio_pred
regr.eval(actuals, predicteds)


# 4.3 mice  == mice short for Multivariate Imputation by Chained Equations
library(mice)
miceMod <- mice(BostonHousing[, !names(BostonHousing) %in% "medv"], method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
anyNA(miceOutput)
#> FALSE

# Lets compute the accuracy of ptratio.

actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- miceOutput[is.na(BostonHousing$ptratio), "ptratio"]
regr.eval(actuals, predicteds)




##### pare 2 We therefore check for features (columns) and samples (rows) where more than 5% of the data is missing using a simple function

pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(data,2,pMiss)
apply(data,1,pMiss)


#The mice package provides a nice function md.pattern() to get a better understanding of the pattern of missing data

library(mice)
md.pattern(data)

#more helpful visual representation can be obtained using the VIM package as follows

library(VIM)
aggr_plot <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, 
                  gap=3, ylab=c("Histogram of missing data","Pattern"))



# Imputing the missing data  The mice() function takes care of the imputing process

tempData <- mice(data,m=5,maxit=50,meth='pmm',seed=500)
summary(tempData)
mice(data = data, m = 5, method = "pmm", maxit = 50, seed = 500)