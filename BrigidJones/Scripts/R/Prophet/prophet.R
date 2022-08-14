#!/usr/bin/env Rscript 
#arg 1 window size
args = commandArgs(trailingOnly=TRUE)
library(prophet)
winSize <- args[1]
predict <- data.frame(yhat = c())
start.Date <- as.Date("2020/1/1")
train <- read.csv(header = FALSE, file = 'Scripts/R/Prophet/train.csv')
test <- read.csv(header = FALSE, file = 'Scripts/R/Prophet/test.csv')

numpred = length(test)
date.range <- seq.Date(start.Date,along.with = train,	by = "month")
df <- data.frame(ds = unlist(date.range), y = unlist(train))

start.time <- Sys.time()
iter <- as.integer(numpred / winSize)


if (!numpred %% winSize == 0){
  iter <- iter+1

}
horizon <- 0
for(i in 0:(iter-1)){
  if(numpred -winSize*i < winSize){
    horizon <- numpred - winSize*i
  }
  else{
    horizon <- winSize
  }
  m <- prophet(df)
  future <- make_future_dataframe(m, periods = horizon)
  forecast <- predict(m, future)

  if(!i==iter-1){
    appender <- data.frame(ds = tail(future["ds"],winSize) ,y = unlist(test)[(i*winSize+1):(i*winSize+winSize)])
    df <- rbind(df, appender)
  }
  predict <- rbind(predict,tail(forecast["yhat"],horizon))
}


end.time <- Sys.time()
time.taken <- end.time - start.time

write(time.taken, file = "Scripts\R\Prophet\runtime.txt",append = FALSE)
write.csv(predict,file='Scripts/R/Prophet/out.csv',row.names=F)
