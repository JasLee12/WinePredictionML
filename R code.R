# ========================================================================================================
# Title:        Application and Analysis of ML Algorithms into Predicting Wine Quality and Type
# Data Source:  UC Irvine Machine Learning Repository
# Date:         3rd April 2024
#=========================================================================================================
# Link for UCI dataset (code to download data not specified as necessary)
url <- 'https://archive.ics.uci.edu/dataset/186/wine+quality'

# Read dataset
red <- read.csv('winequality-red.csv', sep = ';')
white <- read.csv('winequality-white.csv', sep = ';')
  
# Data Pre-processing
library(dplyr)
set.seed(2020)
# Add wine type as colour (red or white)
red$type <- 'red'
white$type <- 'white'

# Merge wine datasets together
wine <- merge(red, white, all = TRUE)

wine$type <- as.factor(wine$type)

# Missing and duplicated values
sum(is.na(wine))
# No missing values found
sum(duplicated(wine))
# Noted on duplicate values

# Exploratory Data Analysis
summary(wine)
apply(wine[,sapply(wine, is.numeric)],2,var)
apply(wine[,sapply(wine, is.numeric)],2,sd)

# QQ plots
library(car)
 wine.num <- wine[1:12]

par(mar=c(1,1,1,1))
par(mfrow = c(3,4))
for (i in 1:ncol(wine.num)) {
qqplot <- qqPlot(wine.num[,i], main = names(wine.num)[i])
}

# Correlation
library(corrplot)

par(mfrow=c(1,1))
cor.wine <- cor(wine[,-13])
corrplot(cor.wine, method = 'number', type = 'lower')

# Variable Importance for feature selection
library(caret)

filterVarImp(x= wine[,1:11], y = wine$type)
filterVarImp(x= wine[,1:11], y= wine$quality)

# Density plots
library(ggplot2)
library(ggridges)

colname <- c(colnames(wine)[1:11])
# By wine type
d_ggplot <- function(column){
  ggplot(data=wine) +
    geom_density(aes_string(x= column, fill= 'type'), alpha= 0.5) +
    ggtitle(column)
}

lapply(colname, d_ggplot)

colname <- colnames(wine[,c(1:11,13)])
# By quality
dr_ggplot <- function(column){
  ggplot(data=wine) +
    geom_density_ridges(aes_string(group= 'quality',x= column, y= 'quality', fill= 'quality'), alpha= 0.5) +
    ggtitle(column)
}
lapply(colname, dr_ggplot)

# Boxplot
boxplot <- function(column){
  ggplot(data=wine) +
    geom_boxplot(aes_string(group= 'quality',x= 'quality', y= column, fill= 'quality'), alpha= 0.5) +
    ggtitle(column)
}
lapply(colname, boxplot)

boxplot(wine$quality)


# Unsupervised Learning
# Deal with Numeric and non-numerical data
numeric.variables <- sapply(wine, is.numeric)
numeric <- wine[,numeric.variables]
factor <- wine[, !numeric.variables]
# Standardise features
scaled <- scale(numeric[,-which(names(numeric)=='quality')])
wine.scaled <- cbind(scaled, factor)

# 1. PCA for dimension reduction (Feature Selection for Quality Regression Analysis)
library(FactoMineR)
library(factoextra) 

# Perform PCA
pca <- prcomp(wine.scaled, center = TRUE, scale. = TRUE)
summary(pca)

pr.var=pca$sdev^2
pve=pr.var/sum(pr.var)

# Plot 
par(mfrow=c(1,2))
plot(pve, type = "b", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained")
plot(cumsum(pve), xlab="Principal Component", 
            ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')

print('The first 4 components explain about 73% of variance in the data')

# Visualisation of variable contribution
fviz_pca_var(pca, col.var = 'contrib', gradient.cols= c('green','red'))

# Select 4 features for quality prediction
print('Variables selected: density, factor= type, total sulfur dioxide, 
      residual sugar')


# 2. K-means Clustering (Feature Selection for Wine Type Classification Analysis)

# Determine optimal number of clusters (k) using elbow method

wss <- numeric(10)
for (i in 1:10) {
  kmeans_model <- kmeans(scaled, centers = i)
  wss[i] <- sum(kmeans_model$withinss)
}
par(mfrow=c(1,1))
plot(1:10, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters (k)", ylab = "Within Sum of Squares (WSS)",
     main = "Elbow Method for Optimal k")
print('Selected k: 3')

# Perform K-means with selected k
k <- 3
kmeans_model <- kmeans(scaled, centers = k, nstart = 25)
print(kmeans_model)

# Plot
fviz_cluster(kmeans_model, data = wine[1:11], palette= c('black','blue','red'),
             ggtheme = theme_bw())

# Identifying homogeneous population groups
# Using Centroid Values
features.type <- apply(wine[1:11], 2, function(x) {
  centroid_values <- tapply(x, kmeans_model$cluster, mean)
  selected_feature <- names(centroid_values)[which.max(centroid_values)]
  return(selected_feature)
}) 

print(sort(features.type, decreasing = FALSE))

# Feature selection for wine type prediction
print('Variables selected: fixed Acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol')
features.type <- select(wine, -c('quality','type'))


# Regression Analysis

# Train test split, stratified with quality
library(splitstackshape)
train <- stratified(wine,c("quality"), 
                    0.7, bothSets = TRUE)
trainset <- subset(train$SAMP1)
testset <- subset(train$SAMP2)

# 1. Multi-Linear Regression

# Train model using features selected from PCA dimension reduction
lr1 <- lm(quality ~ density+ type+ total.sulfur.dioxide+ residual.sugar, data = trainset)
summary(lr1)
sqrt(mean(residuals(lr1)^2))
## RMSE = 0.8021657

# Predict on testset
predict.lr <- predict(lr1, newdata = testset)

# Performance metric
lr.error <- testset$quality - predict.lr
MSE.lr <- mean(lr.error^2)
RMSE.lr <- sqrt(mean(lr.error^2))
## MSE = 0.6578543
## RMSE = 0.8110822

# Train model using all features
lr2 <- lm(quality ~., data = trainset)
summary(lr2)
# Noted insignificant and less significant variables: citric acid, chlorides
sqrt(mean(residuals(lr2)^2))
## RMSE = 0.7319389

# Predict on testset
predict2.lr <- predict(lr2, newdata = testset)

# Performance metric
lr.error2 <- testset$quality - predict2.lr
MSE.lr2 <- mean(lr.error2^2)
RMSE.lr2 <- sqrt(mean(lr.error2^2))
## MSE = 0.5411637
## RMSE = 0.7356383 (better)

# Train model without citric acid and chlorides
lr3 <- lm(quality~.-citric.acid-chlorides, data = trainset)
summary(lr3)
sqrt(mean(residuals(lr3)^2))
# RMSE = 0.7323 (higher RMSE than with these variables)
# No need to remove variables, error measure not improved

# With 10-fold CV
n <- nrow(wine)
folds <- 10

RMSE.train <- seq(1:folds)
RMSE.test <- seq(1:folds)
MSE.test <- seq(1:folds)

# Generate indices of holdout observations (testset pieces) in a list
testlist <- split(sample(1:n), 1:folds)

# Loop training and testing with 10-fold cv
for (i in 1: folds) {
  trainset = wine[-testlist[[i]],]
  testset = wine[testlist[[i]],]
  # Model training
  m.train <- lm(quality ~ ., data = trainset) 
  # Predict on testset
  predict.m.test <- predict(m.train, newdata = testset)
  # Performance metric
  RMSE.train[i] <- sqrt(mean(residuals(m.train)^2))
  testset.error <- testset$quality - predict.m.test
  MSE.test[i] <- mean(testset.error^2)
  RMSE.test[i] <- sqrt(mean(testset.error^2))
}

# Performance metric
mean(RMSE.train)
## RMSE = 0.7322495
MSE.test.mean <- mean(MSE.test)
RMSE.test.mean <- mean(RMSE.test)
## MSE = 0.5395028
## RMSE = 0.7338793

# Multicollinearity
vif(lr2)
# There are variables with high collinearity

# 2. Ridge/Lasso
library(glmnet)

x <- model.matrix(quality~., trainset)[,-1]
y <- trainset$quality
test_x <- model.matrix(quality~., testset)[,-1]

# Finding best lambda using cross validation
ridge <- cv.glmnet(as.matrix(x), y, alpha=0)
sqrt(mean(ridge$cvm)^2)
# RMSE = 0.6566574

# Select lambda
ridge$lambda.min
ridge$lambda.1se
CF <- ridge$lambda.1se
CF2 <- ridge$lambda.min

# Train model subject to 1SE lambda
ridge <- glmnet(as.matrix(x), y, alpha = 0, lambda = CF)
# Predict on testset
predict.ridge <- predict(ridge, s= CF, newx= test_x)
# Performance metric
ridge.error <- testset$quality - predict.ridge
MSE.ridge <- mean(ridge.error^2)
RMSE.ridge <- sqrt(mean(ridge.error^2))
## MSE = 0.5999168
## RMSE = 0.774543

# Train model subject to minimum lambda
ridge2 <- glmnet(as.matrix(x),y,alpha = 0, lambda = CF2)
predict.ridge2 <- predict(ridge2, s= CF2, newx= test_x)
ridge2.error <- testset$quality - predict.ridge2
MSE.ridge2 <- mean(ridge2.error^2)
RMSE.ridge2 <- sqrt(mean(ridge2.error^2))
## MSE = 0.5900543
## RMSE = 0.7681499


# 3. CART
library(rpart)
library(rpart.plot)

# Grow tree
cart1 <- rpart(quality ~ ., data = trainset, method = 'anova', control = rpart.control(minsplit = 2, cp = 0))
print(cart1)
printcp(cart1, digits = 3)
plotcp(cart1)

predict <- predict(cart1, newdata = testset)
# Performance metric
cart.error <- testset$quality - predict
MSE.cart <- mean(cart.error^2)
RMSE.cart <- sqrt(mean(cart.error^2))
## MSE = 0.7596302
## RMSE = 0.8715677

# Prune tree
CV.errorcap <- cart1$cptable[which.min(cart1$cptable[,'xerror']),'xerror'] + cart1$cptable[which.min(cart1$cptable[,'xerror']),'xstd']

i <- 1; j <- 4
while (cart1$cptable[i,j] > CV.errorcap) {
  i <- i + 1
}

cp.opt <- ifelse(i>1, sqrt(cart1$cptable[i,1] * cart1$cptable[i-1,1]),1)

cart2 <- prune(cart1, cp = cp.opt)
print(cart2)
printcp(cart2, digits = 3)

## Root node error: 4445/5848 = 0.76
## cart2 trainset MSE = 0.00406 * 0.76 = 0.0030856
# cart2 trainset RMSE = sqrt(0.0030856) = 0.0555481 
## cart2 CV MSE = 0.76464 * 0.76 = 0.5811285
# cart2 CV RMSE = sqrt(0.594377) = 0.7623178

# Predict on testset
predict <- predict(cart2, newdata = testset)
# Performance metric
cart2.error <- testset$quality - predict
MSE.cart2 <- mean(cart2.error^2)
RMSE.cart2 <- sqrt(mean(cart2.error^2))
## MSE = 0.5985617
## RMSE = 0.7736677

# Significance of variables
summary(m.train)

# Comparison of Regression models (Error Measure: RMSE)
df_ra <- data.frame(Final_Model = c("Linear Regression", "Ridge Regression", "CART"),
                MSE = c(MSE.test.mean,MSE.ridge2,MSE.cart2),    
                RMSE = c(RMSE.test.mean,RMSE.ridge2, RMSE.cart2))
df_ra

# Classification Analysis
type <- wine$type
data.ca <- cbind(features.type, type)

# Check and change baseline to type = white (majority class)
levels(data.ca$type)
data.ca$type <- relevel(data.ca$type, ref = "white")
levels(data.ca$type)

# Train test split, stratified with type
train <- stratified(data.ca,c("type"), 
                    0.7, bothSets = TRUE)
trainset <- subset(train$SAMP1)
testset <- subset(train$SAMP2)

# 1. Logistic Regression
library(caTools)

# Model 1
m.logr <- glm(type~., family= binomial, data= trainset)
summary(m.logr)
# Noted some variables as less and not significant, to remove in Model 2 

# Predict Wine type with testset
predict <- predict(m.logr, type = 'response', newdata= testset)

# Setting threshold as standard 0.5 for 2 possible outcomes
threshold <- 0.5
type.hat <- ifelse(predict > threshold, 'red', 'white')

# Confusion matrix
table(predicted= type.hat, observed= testset$type)
# Performance metrics 
accuracy.mlogr <- mean(type.hat == testset$type)
precision.mlogr <- 1464/(1464+4)
sensivity.mlogr <- 1464/(1464+5)
## Accuracy = 0.9953822
## Precision = 0.9972752
## Sensitivity = 0.9965963

# Model 2
m.logr2 <- glm(type~.-sulphates-citric.acid-fixed.acidity-pH, family = binomial, data = trainset)
summary(m.logr2)
predict2 <- predict(m.logr2, type= 'response', newdata= testset)
type.hat <- ifelse(predict2 > threshold, 'red', 'white')
table(predicted= type.hat, observed= testset$type)
## Same as model 1

# Noted no difference in performance measures regardless of less and insignificant variables presence

# 2. Naive Bayes Classifier
library(naivebayes)

nb1 <- naive_bayes(type~. ,data = trainset)
summary(nb1)

# Predict on testset
predict <- predict(nb1, newdata= testset)

# Confusion matrix
table(predicted= predict, observed= testset$type)
# Performance metrics
accuracy.nb <- mean(predict == testset$type)
precision.nb <- 1438/(1438+14)
sensivity.nb <- 1438/(1438+31)
## Accuracy = 0.9769112
## Precision = 0.9903581
## Sensitivity = 0.9788972

# With Laplace Smoothing

# Finding optimal alpha for laplace value
alpha <- seq(0,1, by= 0.1)

accuracy.values <- numeric(length(alpha))

for (i in seq_along(alpha)) {
  nb.model <- naive_bayes(type~., data= trainset, laplace= alpha)
  predictions <- predict(nb.model, newdata= testset)
  
  accuracy.values[i] <- mean(predictions == testset$type)
}

opt.alpha <- alpha[which.max(accuracy.values)]
opt.alpha

# Best laplace smoothing is alpha = 0 (default)

# 3. Random Forest
library(randomForest)

# Model 1
m.rf <- randomForest(type~., data = trainset)
mean(m.rf$predicted == trainset$type)
# Accuracy = 0.9940633

# Predict Wine type with testsest
predict <- predict(m.rf, newdata= testset)

# Confusion matrix
table(predicted= predict, observed= testset$type)
# Performance metrics
accuracy.rf1 <- mean(predict == testset$type)
precision.rf1 <- 1468/(1468+7)
sensitivity.rf1 <- 1468/(1468+1)
## Accuracy = 0.9958953
## Precision = 0.9952542
## Sensitivity = 0.9993193

# With hyperparameter tuning
hyperparameter <- expand.grid(
  mtry = c(2, 3, 4, 5, 6)
 , ntree = c(100, 250, 550, 750, 1000) 
)
# Perform grid search with cv
cv <- trainControl(method = 'cv', number = 5, search = 'grid')
rf.models <- train(type~., data= trainset, method= 'rf', 
                   trControl= cv, tune_Grid= hyperparameter)
# Select best parameters
best_ntree <- rf.models$bestTune$ntree
best_mtry <- rf.models$bestTune$mtry
# best ntrees = null
# best mtry = 2

# Model 2
m.rf2 <- randomForest(type~., data = trainset, mtry= best_mtry)
mean(m.rf2$predicted == trainset$type)
# Accuracy = 0.9949428

# Predict Wine type with testsest
predict <- predict(m.rf2, newdata= testset)

# Confusion matrix
table(predicted= predict, observed= testset$type)
# Performance metrics
accuracy.rf2 <- mean(predict == testset$type)
precision.rf2 <- 1468/(1468+5)
sensitivity.rf2 <- 1468/(1468+1)
## Accuracy = 0.9969215
## Precision = 0.9966056
## Sensitivity = 0.9993193

# Significance of variables
varImpPlot(m.rf2)
## Total sulfur dioxide and chlorides are most significant

# Comparison of Classification models (Error Measure: Accuracy)
df_ca <- data.frame(Model = c("Logistic Regression", "Naive Bayes Classifier", "Random Forest"),
                Accuracy = round(c(accuracy.mlogr, accuracy.nb, accuracy.rf2),3),
                Precision = round(c(precision.mlogr, precision.nb, precision.rf2),3),
                Sensitivity = round(c(sensivity.mlogr, sensivity.nb, sensitivity.rf2),3))
df_ca
