utils::
caret::
datasets::
mlbench::
AppliedPredictiveModeling::
  
graphics::
lattice::
ggplot2::

Amelia::
corrplot::
  
klaR::  
Hmisc::
  
data(iris)
summary(iris)

# data inspection
sapply(BostonHousing,class)
table(PimaIndiansDiabetes$diabetes)

y <- PimaIndiansDiabetes$diabetes
cbind(freq=table(y), percentage=prop.table(table(y))*100)

##min, median, 1st quar, 3rd quar, max

### IMPUTE with mean/median etc
#library(Hmisc)
impute(BostonHousing$ptratio, mean)  # replace with mean
impute(BostonHousing$ptratio, median)  # median
impute(BostonHousing$ptratio, 20)  # replace specific number
# or if you want to impute manually
BostonHousing$ptratio[is.na(BostonHousing$ptratio)] <- mean(BostonHousing$ptratio, na.rm = T)  # not run




summary()

sapply(PimaIndiansDiabetes[,1:8],sd)

apply(PimaIndiansDiabetes[,1:8], 2,skewness)

cor(PimaIndiansDiabetes[,1:8])

#visualization
par(mfrow=c(1,4))
for(i in 1:4) {
  hist(iris[,i], main=names(iris)[i])
}

for (i in 1:8) { 
  plot(density(PimaIndiansDiabetes[,i]), main = names(PimaIndiansDiabetes)[i] )
}

for (i in 1:4) { 
  boxplot(iris[,i], main = names(iris)[i] )
  }

for (i in 2:9 ) { 
  barplot(table(BreastCancer[,i]),main = names(BreastCancer)[i])
  }

missmap(Soybean, col = c("black","darkred"), legend=FALSE)  #missing vale

#multivariate
corrplot(cor(iris[,1:4]), method= "circle")  #correlation plot

pairs(iris) #scatterplot
pairs(Species~., data=iris, col=iris$Species)


# PRE PROCESSING
library(caret)
preproc <- preProcess(iris[,1:4], method = c("scale"))   #divide values by standard deviation.
preproc <- preProcess(iris[,1:4], method = c("center"))  #subtract mean from values
preprocessParams <- preProcess(iris[,1:4], method=c("center", "scale"))  # standardize data -attribute will have mean value of 0 and a standard deviation of 1
preprocessParams <- preProcess(iris[,1:4], method=c("range"))  # Normalize - dataData values can be scaled into the range of [0, 1]

#When an attribute has a Gaussian-like distribution but is shifted, this is called a skew.
#The distribution of an attribute can be shifted to reduce the skew and make it more Gaussian. 
#The BoxCox transform can perform this operation (assumes all values are positive).
preprocessParams <- preProcess(PimaIndiansDiabetes[,7:8], method=c("BoxCox"))  # Box-Cox Transformli

#Yeo-Johnson Transform - The YeoJohnson transform another power-transform like Box-Cox,
#but it SUPPORTS raw values that are equal to ZERO and NEGATIVE.
preprocessParams <- preProcess(PimaIndiansDiabetes[,7:8], method=c("YeoJohnson"))

preprocessParams <- preProcess(iris, method=c("center", "scale", "pca"))   # Principal Component Analysis Transform   PCA
preprocessParams <- preProcess(PimaIndiansDiabetes[,1:8], method=c("center", "scale","ica"), n.comp=5)  # Independent Component Analysis Transform


# summarize transform parameters
transformed <- predict(preproc, iris[,1:4])


#### Resampling ######
# 1 data split

# define an 80%/20% train/test split of the dataset
trainIndex <- createDataPartition(iris$Species, times=1, p=0.80, list=FALSE) 
dataTrain <- iris[ trainIndex,]
dataTest <- iris[-trainIndex,]
fit <- NaiveBayes(Species~., data=dataTrain)
# make predictions
predictions <- predict(fit, dataTest[,1:4])
# summarize results
confusionMatrix(predictions$class, dataTest$Species)











