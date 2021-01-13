library(naivebayes)
library(caret)
library(readr)
library(tm)
spam <- read_csv("spam.csv", col_types = cols(type = col_factor(levels = c("ham", "spam"))))
train.index <- createDataPartition(spam$type, p = 0.75, list = FALSE)
train <- spam[train.index, ]
test <- spam[-train.index, ]
corpus.train <- Corpus(VectorSource(train$message))
corpus.test <- Corpus(VectorSource(test$message))

freq <- findFreqTerms(DocumentTermMatrix(corpus.train), lowfreq = 2) #, highfreq = 200

dtm.train <- DocumentTermMatrix(corpus.train, control = list(dictionary = freq))
dtm.test <- DocumentTermMatrix(corpus.test, control = list(dictionary = freq))
model <- multinomial_naive_bayes(as.matrix(dtm.train), train$type) #dtm.train
pred <- predict(model,as.matrix(dtm.test), type ="class")
print(confusionMatrix(pred, test$type)$overall["Accuracy"])
length(freq)
