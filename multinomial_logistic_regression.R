
rm(list=ls())
library(glmnet)
library(stats)
library(caret)
library(nnet)
library(ggplot2)
library(reshape2)


data <- read.csv("/Users/yangzining/Desktop/nolinear optimization/sleep_scoring_ground_true.csv")
str(data)
# remove all the NAs
data_clean <- na.omit(data)
data_clean <- data_clean[,-which(names(data_clean)%in% c("AnimalName","Geno","Date"))]

x <- data_clean[ ,-which(names(data_clean)%in% c("sleep_labels"))]
x <- as.matrix(x)
x <- scale(x)
y <- data_clean[ ,which(names(data_clean)%in% c("sleep_labels"))]
y <- as.factor(y)

grid <- 10^ seq (10,-2, length =100)
mult_mod <- glmnet(x,y,alpha =1,lambda =grid,family="multinomial")

# doing cross validation
set.seed(123)  
data_sample <- data_clean[sample(nrow(data_clean), size = 3000), ]  # 
x_sample <- as.matrix(data_sample[, -which(names(data_sample) == "sleep_labels")])
y_sample <- as.factor(data_sample$sleep_labels)

cv_fit_mult <- cv.glmnet(x_sample, y_sample, alpha = 1, family = "multinomial", nfold = 10)


#predict the result on sleep class
pred_mult<- predict(mult_mod,s=cv_fit_mult$lambda.min ,newx=x, type = "class")


# getting the confusion matirx
conf_matrix <- table(Predicted = pred_mult, Actual = y)

# Normalize the confusion matrix by rows to get probabilities
conf_matrix_probs <- sweep(conf_matrix, 1, rowSums(conf_matrix), FUN="/")

# View the normalized confusion matrix
print(conf_matrix_probs)

# Melt the data for ggplot
conf_matrix_long <- melt(conf_matrix_probs)

# Plot using ggplot2
ggplot(conf_matrix_long, aes(x = Predicted, y = Actual, fill = value)) +
  geom_tile(color = "white") +  # add tile geometry
  geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 4) +  # add text annotations
  scale_fill_gradient(low = "white", high = "blue") +  # color gradient
  labs(x = "Predicted Class", y = "Actual Class", fill = "Probability") +  # labels
  theme_minimal()  # minimal theme

