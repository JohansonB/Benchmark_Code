args = commandArgs(trailingOnly=TRUE)
library(prophet)
#args[1] prediction length
#args[2] starting date
#args[3] step_size between predictions
#args[4] trend type ("linear" or "flat")
numpred <- as.integer(args[1])
predict <- data.frame(yhat = c())
start.Date <- as.POSIXct(args[2])
train <- read.csv(header = FALSE, file = 'C:/Users/41766/IdeaProjects/BrigidJones/Scripts/R/Prophet/train.csv')

step = gsub("_"," ",args[3])
date.range <- seq.POSIXt(start.Date,along.with = train,	by = step)
df <- data.frame(ds = unlist(date.range), y = unlist(train))

start.time <- Sys.time()

m <- prophet(growth = args[4])
m <- fit.prophet(m, df)
future <- make_future_dataframe(m, periods = numpred, freq = "m")
forecast <- predict(m, future)


predict <- tail(forecast["yhat"],numpred)



end.time <- Sys.time()
time.taken <- end.time - start.time

write(time.taken, file = "C:/Users/41766/IdeaProjects/BrigidJones/Scripts/R/Prophet/runtime.txt",append = FALSE)
write.csv(predict,file='C:/Users/41766/IdeaProjects/BrigidJones/Scripts/R/Prophet/out.csv',row.names=F)
plot(m,forecast)
