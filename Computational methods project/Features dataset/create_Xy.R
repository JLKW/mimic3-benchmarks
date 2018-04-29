train_X <- read.csv("features_train.csv", stringsAsFactor=F)
test_X <- read.csv("features_test.csv", stringsAsFactor=F)
train_y <- read.csv("listfile_train.csv", stringsAsFactor=F)
test_y <- read.csv("listfile_test.csv", stringsAsFactor=F)

train_X <- train_X[order(train_X$id),]
test_X <- test_X[order(test_X$id),]
train_y <- train_y[order(train_y$stay),]
test_y <- test_y[order(test_y$stay),]

all(train_X$id == gsub("([^_]*_[^_]*)_.*$", "\\1", train_y$stay))
all(test_X$id == gsub("([^_]*_[^_]*)_.*$", "\\1", test_y$stay))

train_X$y <- train_y$y_true
test_X$y <- test_y$y_true

write.csv(train_X, "Xy_train.csv", row.names=F)
write.csv(test_X, "Xy_test.csv", row.names=F)