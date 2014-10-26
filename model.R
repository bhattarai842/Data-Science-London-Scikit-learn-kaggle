setwd("~/Desktop/personal/kaggle/Data Science London + Scikit-learn/")
library(caret)
library(doMC)

#load data
rawTrain <- read.csv(file="./data/train.csv",header=T)
names(rawTrain)[-1] <- paste0("V",1:40)
test <- read.csv("./data/test.csv",header=F)

###########################
# Prediator transformation
############################
# combine_data <- rbind(rawTrain[,-1],test)
# preprocessModel <- preProcess(combine_data, method = "BoxCox")
# combine_data <- predict(preprocessModel, combine_data)
# train <- combine_data[1:1000,]
# train[,1] <- as.factor(ifelse(rawTrain[,1] == 0, "zero", "one"))
# names(train)[1] <- "label"
# test <- combine_data[1001:10000,]

############## Preprocessing
preprocessModel <- preProcess(test, method = "BoxCox")
train <- predict(preprocessModel, rawTrain[, -1])
train[,1] <- as.factor(ifelse(rawTrain[,1] == 0, "zero", "one"))
names(train)[1] <- "label"
test <- predict(preprocessModel, test)


#######################
## Feature Selection ##
#######################
registerDoMC(10)
rfeFuncs <- rfFuncs
rfeFuncs$summary <- twoClassSummary
rfe.control <- rfeControl(rfeFuncs, method = "repeatedcv", number=10 ,repeats = 4, verbose = FALSE, returnResamp = "final")
rfe.rf <- rfe(train[,-1], train[,1], sizes = 10:15, rfeControl = rfe.control, metric="ROC")

train <- train[,c("label",predictors(rfe.rf))]
test <- test[,predictors(rfe.rf)]
save(train,test,file="trainDatamodel.RData")

################### Using H2O
library(h2o)
localH2O <- h2o.init(max_mem_size = '4g')

train.h2o <- as.h2o(localH2O, train, key = 'train.h2o')
test.h2o <- as.h2o(localH2O, test, key = 'test.h2o')

modelI <- h2o.deeplearning(x = 2:16,
                          y = "label",
                          data = train.h2o,
                          nfolds = 10,
                          activation = "Tanh",
                          hidden = c(50, 50, 50),
                          epochs = 100,
                          balance_classes = FALSE)

predictI <- h2o.predict(modelI, test.h2o) 
predictI.df <- as.data.frame(predictI)
### semi-supervised model
newtrainindex <- which(predictI.df$one >= 0.97 | predictI.df$one <= 0.03)
newtrain <- test[newtrainindex,]
newtrain$label <- as.factor(ifelse(predictI.df$one[newtrainindex] < 0.5, "zero","one"))
train <- rbind(train,newtrain[,names(train)])

## Phase II model
trainII.h2o <- as.h2o(localH2O, train, key = 'train.h2o')
#trainSplit <- h2o.splitFrame(trainII.h2o, ratios = 0.8, shuffle = TRUE)


modelII <- h2o.deeplearning(x = 2:15,
                           y = "label",
                           data = trainII.h2o,
                           nfolds = 10,
                           activation = "Tanh",
                           hidden = c(50, 50, 50),
                           epochs = 100,
                           balance_classes = FALSE)

predictII <- h2o.predict(modelII, test.h2o) 
predictII.df <- as.data.frame(predictII)

test.pred <- (predictI.df$one + predictII.df$one)/2
write.csv(data.frame(id=1:length(test.pred),solution=ifelse(test.pred < 0.5, 0 ,1)),file="~/Desktop/submission.csv",row.names=F)


