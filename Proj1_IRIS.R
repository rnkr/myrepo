# R Project Template

# 1. Prepare Problem
# a) Load libraries
library(caret)

# b) Load dataset
data(iris)
dataset <- iris
  ## filename <- "iris.csv"   ##-- UCI Machine Learning Repository
  ## dataset <- read.csv(filename, header=FALSE)
  ## colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")


# c) Split-out validation dataset
validationIndex <- createDataPartition(dataset$Species, p = 0.80, list=FALSE)
validation <- dataset[-validationIndex,]
dataset <- dataset[validationIndex,]

# 2. Summarize Data
# a) Descriptive statistics
dim(dataset)
sapply(dataset, class)
head(dataset)
levels(dataset$Species)
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq = table(dataset$Species), percentage)
summary(dataset)

# b) Data visualizations
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(dataset)[i])
  }
# barplot for class breakdown
plot(y)
# scatter plot matrix
featurePlot(x=x, y=y, plot="ellipse")
# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)



# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms

# a) Test options and evaluation metric
# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method = "cv", number =10)
metric <- "Accuracy"

# b) Spot Check Algorithms

#LDA Linear Discriminant Analysis (LDA)
set.seed(17)
fit.lda <- train(Species~., data = dataset, method = "lda", metric = metric, trControl = trainControl)

#CART Classification and REgression Trees 
set.seed(17)
fit.cart <- train(Species~., data = dataset, method = "rpart", metric = metric, trControl = trainControl)

#KNN K Nearest Neighbours
set.seed(17)
fit.knn <- train(Species~., data = dataset, method = "knn", metric = metric, trControl = trainControl)

#SVM Support Vector Machines
set.seed(17)
fit.svm <- train(Species~., data = dataset, method = "svmRadial", metric = metric, trControl = trainControl)

#Random Forest
set.seed(17)
fit.rf <- train(Species~., data = dataset, method = "rf", metric = metric, trControl = trainControl)

# c) Compare Algorithms


#summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)


# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

# 6. Finalize Model
# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use
