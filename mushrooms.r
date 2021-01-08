library(readr)
library(caret)
library(rpart)

mushrooms <- read_csv("mushrooms.csv", col_types = cols("class" = col_factor(), "gill-attachment" = col_factor(), "veil-type" = col_skip()))
# This will convert all remaining columns to factors - not neccessary for training models, but usefull for debugging
mushrooms <- mushrooms %>% mutate_if(is.character,as.factor)

# Renaming columns to avoid bugs
colnames(mushrooms) <- make.names(colnames(mushrooms))

# Classic Train test split
train.index <-  createDataPartition(mushrooms$class,p = 0.75, list = FALSE)
train <- mushrooms[train.index, ]
test <- mushrooms[-train.index, ]

# Training Binomial Logistic Regression
multinomMushi <- multinom(class ~ ., data=train)
predictions <- predict(multinomMushi,test)
print(confusionMatrix(predictions,test$class)$overall["Accuracy"])

# Training normal Logistic Regression
logisticModel <- glm(class ~ ., family = binomial(), data = train, control = list(maxit = 100))
mean((predict(logisticModel,test, type ="response")>0.5)==(test$class=="e"))

# Building decision tree
decisionTree <- rpart(class ~ ., 
               data = train, 
               parms = list(split = "information"), 
               method = "class",
               control = rpart.control(minsplit = 1, xval = 1, minbucket = 1, cp = 0.005))
prp(decisionTree)

predictions <- predict(decisionTree,test,type="class")
print(confusionMatrix(predictions,test$class)$overall["Accuracy"])

# Below the better way: Proper RepatedCrossValidation with caret
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

multinomModels <- train(class ~ ., data = mushrooms, method = "multinom",
               trControl = train.control)
logisticRegressionModels <- train(class ~ ., data = mushrooms, method = "glm",
      trControl = train.control, control = list(maxit = 100))

decionTreeModels <- train(class ~ ., data = mushrooms, method = "rpart",
               trControl = train.control)

# Strangely this model does not look nice
prp(model$finalModel)

# Training a single decsion tree again with the best parameters
decisionTree <- rpart(class ~ ., 
                      data = train, 
                      parms = list(split = "information"), 
                      method = "class",
                      control = rpart.control(cp = model$bestTune["cp"]))
predictions <- predict(decisionTree,test,type="class")
print(confusionMatrix(predictions,test$class)$overall["Accuracy"])
prp(decisionTree)

