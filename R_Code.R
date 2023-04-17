# Loading packages

install.packages("DescTools")
install.packages("psych")
install.packages("car")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("RColorBrewer")
install.packages("HH")
install.packages("glmnet")
install.packages("Metrics")

library(ggplot2)
library(dplyr)
library(car)
library(psych)
library(corrplot)
library(RColorBrewer)
library(leaps)
library(DescTools)
library(HH)
library(glmnet)
library(Metrics)

# Loading Loan Dataset

dataset<-read.csv("//Users//aswathyvinod//Desktop//train_ctrUa4K.csv", header = TRUE, sep = ",")
dataset

# EDA and Descriptive Statistics

dim(dataset)
str(dataset)
head(dataset)
bruceR::Describe(dataset, all.as.numeric = TRUE)
names(dataset)
summary(dataset)

# Checking for missing values

is.na(dataset)

# Replacing NA with mean value of the variables

dataset$Gender <- ifelse(dataset$Gender == "",Mode(dataset$Gender,na.rm=TRUE),dataset$Gender)
dataset$Married <- ifelse(dataset$Married == "",Mode(dataset$Married,na.rm=TRUE),dataset$Married)
dataset$Dependents <- ifelse(dataset$Dependents == "",Mode(dataset$Dependents,na.rm=TRUE),dataset$Dependents)
dataset$Self_Employed <- ifelse(dataset$Self_Employed == "",Mode(dataset$Self_Employed,na.rm=TRUE),dataset$Self_Employed)
dataset$Credit_History <- ifelse(is.na(dataset$Credit_History) == TRUE,Mode(dataset$Credit_History,na.rm=TRUE),dataset$Credit_History)
dataset$LoanAmount <- ifelse(is.na(dataset$LoanAmount) == TRUE,round(mean(dataset$LoanAmount,na.rm=TRUE),2),round(dataset$LoanAmount,2))
dataset$Loan_Amount_Term <- ifelse(is.na(dataset$Loan_Amount_Term) == TRUE,round(mean(dataset$Loan_Amount_Term,na.rm=TRUE),0),round(dataset$Loan_Amount_Term,0))

is.na(dataset)

# visualisation

# Histogram

par(mfrow = c(2, 2))
hist(dataset$LoanAmount, main = "Histogram of Loan amount", xlab = "Loan Amount")
hist(dataset$ApplicantIncome, main = "Histogram of Applicant's income", xlab = "Applicant Income")
hist(dataset$Loan_Amount_Term, main = "Histogram of Loan Term", xlab = "Loan term")

# Bar plot

ggplot(dataset, aes(x=Gender, y=LoanAmount)) + geom_bar(stat = "identity") + ggtitle("Loan Amount Gender wise")
ggplot(dataset, aes(factor(Dependents),fill = factor(Dependents))) +geom_bar() + ggtitle("Number of dependents")
ggplot(dataset, aes(factor(Property_Area),fill = factor(Property_Area))) +geom_bar() + ggtitle("Dwelling region of the applicants")

# Box plot

boxplot(LoanAmount ~ Married, data = dataset, 
        xlab = "Marital status",
        ylab = "Loan Amount", 
        main = " Loan amount vs Marital status",
        notch = TRUE, 
        varwidth = TRUE, 
        col = c("blue","orange"),
        names = c("No","Yes")
)



############################################################################################################
# GLM
############################################################################################################

## Splitting the dataset into train and test

set.seed(23061997)
trainIndex <- sort(sample(x = nrow(dataset), size = nrow(dataset) * 0.7)) 
train <- dataset[trainIndex,-1] 
test <- dataset[-trainIndex,-1] 


model <- glm(Loan_Status ~ ., data = train, family = binomial(link = "logit"))
summary(model)


model2 <- glm(Loan_Status ~ CoapplicantIncome + Credit_History  + Property_Area , data = train, family = binomial(link = "logit"))
summary(model2)

vif(model2)

predicted_train <- plogis(predict(model2, newdata = train))  # predicted scores


prop.table(table(train$Loan_Status))

optimalCutoff(train$Loan_Status, as.numeric(predicted_train)[1])
optCutOff <- optimalCutoff(test$Loan_Status, predicted_train[1]) 


## Predicted probabilities for Train
train_prob <- predict(model2, train, type = 'response')
train.predicted <- as.factor(ifelse(train_prob >= optCutOff,"Y","N"))
train.confusionMatrix <- confusionMatrix(train.predicted, train$Loan_Status, positive = 'Y')



precision <- train.confusionMatrix$byClass['Pos Pred Value']    
recall <- train.confusionMatrix$byClass['Sensitivity']


## Predicted probabilities for Test

test_prob <- predict(model2, test, type = 'response')
test.predicted <- as.factor(ifelse(test_prob >= optCutOff,"Y","N"))
confusionMatrix(test.predicted, test$Loan_Status, positive = 'Y')


#ROC

ROC <- roc(test$Loan_Status,test_prob)
plot(ROC, col = 'blue', ylab = 'Sensitivity - TP Rate', xlab = 'Specificity - FP Rate')

## Area under ROC curve - AUC 
auc(ROC)


############################################################################################################
# Regression to predict loan amount to be sanctioned based on other variables
############################################################################################################

# Producing Correlation Matrix for Numeric Variables

a<-dataset[,unlist(lapply(dataset,is.numeric))]
a
names(a)
cor1<-cor(a) 
cor1

# Plot for Correlation Matrix

corrplot(cor1,type='upper',tl.cex=0.5)

# Scatterplot for SalePrice against Highest,lowest correlation

# Highest

scatterplot(LoanAmount~ApplicantIncome,data=dataset)

# Lowest

scatterplot(LoanAmount~Credit_History,data=dataset)

# Fitting  a Regression Model with continuous variables

lg<- lm(LoanAmount~ApplicantIncome+CoapplicantIncome+Loan_Amount_Term+Credit_History, data = dataset)
lg
summary(lg)
summary(lg)$adj.r.squared
BIC(lg)
AIC(lg)

# Plotting the Regression Model using Plot() Function

par(mfrow=c(2,2))
plot(lg)
dev.off()

# Multicollinearity Findings

vif(lg)

# Outliers Findings

outlierTest(lg)

# High leverage Points

hat.plot <- function(lg){
  p <- length(coefficients(lg))
  n <- length(fitted(lg))
  plot(hatvalues(lg), main="Index Plot of Hat Values")
  abline(h=c(2,3)*p/n, col="red", lty=2)
  identify(1:n, hatvalues(lg), names(hatvalues(lg)))
}
hat.plot(lg)

#Influential Observations

cutoff <- 4/(nrow(df)-length(lg$coefficients)-2)
plot(lg, which=4, cook.levels=cutoff)
abline(h=cutoff, lty=2, col="red")

# Changes with Normality of the Dependent variable Loan Amount and Outliers removal

dataset<- dataset[-c(410,131,562,186,605,582,370,178,524,488), ]
dataset

# correction of normality

hist(dataset$LoanAmount,main='Histogram before Normality Loan Amount')
dataset$LoanAmount<-sqrt(dataset$LoanAmount)
hist(dataset$LoanAmount,main='Histogram after Normality Loan Amount')

# New regression model  with normality and without outliers and removel of non significant variable

str(dataset)
lg1<-lm(LoanAmount~ApplicantIncome+CoapplicantIncome+Loan_Amount_Term, data = dataset)
summary(lg1)
summary(lg1)$adj.r.squared
BIC(lg1)
AIC(lg1)

par(mfrow=c(2,2))
plot(lg1)
dev.off()

# All regression subset to find the best model

ss<-regsubsets(LoanAmount~ApplicantIncome+CoapplicantIncome+Loan_Amount_Term+Credit_History, data = dataset,nbest=4)
plot(ss,scale="adjr2")

summary(ss)

############################################################################################################
#Chi-Square Test for independence
############################################################################################################

# Hypothesis

#H0: Loan Status is independent of the no. of Dependents
#H1: Loan Status is  dependent on the no. of Dependents

alpha <- 0.05

mtrx <- matrix(table(dataset$Dependents, dataset$Loan_Status),nrow = 4, byrow = FALSE) 
rownames(mtrx) <- c("0","1","2","3+")
colnames(mtrx) <- c("N","Y")

result <- chisq.test(mtrx)
result

ifelse(result$p.value > alpha, "Fail to Reject Null Hypothesis","Reject the Null Hypothesis")

############################################################################################################
#ANOVA Test for independence
############################################################################################################

# step 1 - form the hypothesis

# The hypotheses for the Married  are (Factor A)

# H0: There is no difference between the means of Loan amount amount for the married or single customers 
# H1: There is a difference between the means of Loan amount amount for the married or single customers 

# The hypotheses for the self_employed  are (Factor B)

# H0: There is no difference between the means of Loan amount for customers' self-employed status.
# H1: There is a difference between the means of  Loan amount for customers' self-employed status.

# The hypotheses for the interaction are

# H0: There is no interaction effect between Married  and self_employed in Loan amount
# H1: There is interaction effect between Married and self_employed in Loan amount.

# step 2 - Compute the test value

#  set significance level

alpha <- 0.05

# step 3-  Run Two-way Anova - test

dataset$Married <- as.factor(dataset$Married)
dataset$Self_Employed <- as.factor(dataset$Self_Employed)

anova <- aov(LoanAmount ~ Married + Self_Employed , data = dataset )
a.summary <- summary(anova)
a.summary

# Two-way ANOVA with interaction effect

inter_anova <- aov(LoanAmount ~ Married + Self_Employed + Married:Self_Employed, data = dataset)

summary(inter_anova)

# Extreact the P value from the summary for mean

pMarried <- summary(anova)[[1]][["Pr(>F)"]][1]
pSelf_Employed <- summary(anova)[[1]][["Pr(>F)"]][2]
pMarried
pSelf_Employed

# Extreact the P value from the summary for interaction

pint <- summary(inter_anova)[[1]][["Pr(>F)"]][3]

pint

# step 4- Make the decision

#  Compare the p-value to the alpha and make the decision.

ifelse(pMarried> alpha , " fail to reject null hypothesis" , "Reject the hypothesis ")

ifelse(pSelf_Employed> alpha , " fail to reject null hypothesis" , "Reject the hypothesis ")

ifelse(pint> alpha , " fail to reject null hypothesis" , "Reject the hypothesis ")

# visualizing the results

# interaction

interaction.plot(x.factor = dataset$Married, #x-axis variable
                 trace.factor = dataset$Self_Employed, #variable for lines
                 response = dataset$LoanAmount, #y-axis variable
                 fun = median, #metric to plot
                 ylab = "Loan amount",
                 xlab = "Married",
                 col = c("pink", "blue"),
                 lty = 1, #line type
                 lwd = 2, #line width
                 trace.label = "SelfEmployed")


############################################################################################################
# Regularization - Ridge & LASSO
############################################################################################################

# Regularization _ Ridge

# Create Train and Test set - maintain % of event rate (70/30 split)

set.seed(1234)
trainIndex <- sample(x = nrow(dataset), size = nrow(dataset)* 0.7)
train <- dataset[trainIndex,]
test <- dataset[- trainIndex,]

train_x <- model.matrix(LoanAmount ~ Gender+ Married +Dependents +Education +
                          Self_Employed + ApplicantIncome + CoapplicantIncome +
                          Credit_History +Property_Area + Loan_Status, train)[,-1]

test_x <- model.matrix(LoanAmount ~  Gender+ Married +Dependents +Education +
                         Self_Employed + ApplicantIncome + CoapplicantIncome +
                         Credit_History +Property_Area + Loan_Status, test)[,-1]


train_y <- train$LoanAmount
test_y <- test$LoanAmount

#fit ridge regression model

model <- glmnet(train_x, train_y, alpha = 0)
plot(model, xvar = "lambda", label = TRUE)

# find the best lambda using cross-Validation

# dev.off()

cv.ridge <- cv.glmnet(train_x, train_y,alpha = 0,nfolds = 10) 
cv.ridge$lambda.min
cv.ridge$lambda.1se
plot(cv.ridge)

# Analyze Final Model

model.ridge.min <- glmnet(train_x,train_y , alpha = 0 , lambda = cv.ridge$lambda.min)
coef(model.ridge.min)

model.ridge.1se <- glmnet(train_x,train_y , alpha = 0 , lambda = cv.ridge$lambda.1se)
coef(model.ridge.1se)

# Train set predictions

# Determine the performance of the fit model against the training set by using RMSE and lambda min

pred.ridge.min.train <- predict(model.ridge.min , newx = train_x)
rmse.ridge.min.train <- rmse(train_y , pred.ridge.min.train)
rmse.ridge.min.train

pred.ridge.1se.train <- predict(model.ridge.1se , newx = train_x)
rmse.ridge.1se.train <- rmse(train_y , pred.ridge.1se.train)
rmse.ridge.1se.train

# Determine the performance of the fit model against the test set by using RMSE and lambda min

pred.ridge.min.test <- predict(model.ridge.min , newx = test_x)
rmse.ridge.min.test <- rmse(test_y , pred.ridge.min.test)
rmse.ridge.min.test

pred.ridge.1se.test <- predict(model.ridge.1se , newx = test_x)
rmse.ridge.1se.test <- rmse(test_y , pred.ridge.1se.test)
rmse.ridge.1se.test


#Fit Lasso regression model

model <- glmnet(train_x, train_y, alpha = 1)
plot(model, xvar = "lambda", label = TRUE)

# find the best lambda using cross-Validation

dev.off()
cv.lasso <- cv.glmnet(train_x, train_y,alpha = 1,nfolds = 10) 
cv.lasso$lambda.min
cv.lasso$lambda.1se
plot(cv.lasso)

# Analyze Final Model

model.lasso.min <- glmnet(train_x,train_y , alpha = 1 , lambda = cv.lasso$lambda.min)
coef(model.lasso.min)

model.lasso.1se <- glmnet(train_x,train_y , alpha = 1 , lambda = cv.lasso$lambda.1se)
coef(model.lasso.1se)

# Train set predictions
# Determine the performance of the fit model against the training set by using RMSE and lambda min

pred.lasso.min.train <- predict(model.lasso.min , newx = train_x)
rmse.lasso.min.train <- rmse(train_y , pred.lasso.min.train)
rmse.lasso.min.train

pred.lasso.1se.train <- predict(model.lasso.1se , newx = train_x)
rmse.lasso.1se.train <- rmse(train_y , pred.lasso.1se.train)
rmse.lasso.1se.train

# Determine the performance of the fit model against the test set by using RMSE and lambda min

pred.lasso.min.test <- predict(model.lasso.min , newx = test_x)
rmse.lasso.min.test <- rmse(test_y , pred.lasso.min.test)
rmse.lasso.min.test

pred.lasso.1se.test <- predict(model.lasso.1se , newx = test_x)
test.lasso.rmse.1se.test <- rmse(test_y , pred.lasso.1se.test)
test.lasso.rmse.1se.test







