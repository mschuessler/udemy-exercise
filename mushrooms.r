library(readr)
mushrooms <- read_csv("mushrooms.csv", col_types = cols("class" = col_factor(), "gill-attachment" = col_factor(), "veil-type" = col_skip()))
train.index <-  createDataPartition(mushrooms$class,p = 0.75, list = FALSE)
train <- mushrooms[train.index, ]
test <- mushrooms[-train.index, ]
multinomMushi <- multinom(class ~ ., data=train)
predictions <- predict(multinomMushi,test)
print(confusionMatrix(predictions,test$class)$overall["Accuracy"])