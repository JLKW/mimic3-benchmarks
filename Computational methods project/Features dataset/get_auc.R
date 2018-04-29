#Put this file in the same directory as "Xy_train.csv" and "Xy_test.csv". Then run the whole code

require(caret)
require(AUC)

train_tbl <- read.csv("Xy_train.csv", stringsAsFactor=F)[-1]
test_tbl <- read.csv("Xy_test.csv", stringsAsFactor=F)[-1]

inds <- which(colSums(is.na(train_tbl)) < 300)

train_tbl <- train_tbl[inds]
test_tbl <- test_tbl[inds]

#pred <- factor(ifelse(predict(logfit, newdata=train_tbl, type="response") > 0.5, 1, 0), levels=0:1)
#confusionMatrix(data=pred, reference=as.factor(train_tbl$y))

logfit <- glm(y~., family=binomial, data=train_tbl)
rocfit <- roc(predict(logfit, newdata=test_tbl, type="response"), as.factor(test_tbl$y))
plot(rocfit)
auc(rocfit)

