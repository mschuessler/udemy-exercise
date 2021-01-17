library(keras)
# Note data is formated as 28x28 instead of 1x784
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape - into Matrix format
# Note that we use the array_reshape() function rather than the dim<-() function to reshape the array.
# This is so that the data is re-interpreted using row-major semantics 
# (as opposed to R’s default column-major semantics), which is in turn compatible 
# with the way that the numerical libraries called by Keras interpret array dimensions.
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255


#The y data is an integer vector with values ranging from 0 to 9. To prepare this data for training 
# we one-hot encode the vectors into binary class matrices using the Keras to_categorical() function:
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


plot(history)
model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test)

lenet <- keras_model_sequential() 
lenet %>% layer_conv_2d(filters = 6, kernel_size = 5, activation = "sigmoid", padding = "same") %>%
  layer_average_pooling_2d(pool_size = 2,strides = 2) %>%
  layer_conv_2d(filters = 16, kernel_size = 5, activation = "sigmoid") %>% 
  layer_average_pooling_2d(pool_size = 2,strides = 2) %>%
  layer_flatten() %>%
  layer_dense(84, activation = "sigmoid") %>%
  layer_dense(10)

lenet %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- lenet %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)
