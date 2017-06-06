stock=read.csv("./training_test_data.csv")
set.seed(333)
index<-sample(seq_len(nrow(stock)),size=0.75*nrow(stock))
training.set<-stock[index,]
test.set<-stock[-index,]


#Prediction on model

model<-glm(snp_log_return_positive~snp_log_return_1+snp_log_return_2+snp_log_return_3+
             nyse_log_return_1+nyse_log_return_2+nyse_log_return_3+
             djia_log_return_1+djia_log_return_2+djia_log_return_3+
             nikkei_log_return_0+nikkei_log_return_1+nikkei_log_return_2+
             hangseng_log_return_0+hangseng_log_return_1+hangseng_log_return_2+
             ftse_log_return_0+ftse_log_return_1+ftse_log_return_2+
             dax_log_return_0+dax_log_return_1+dax_log_return_2+
             aord_log_return_0+aord_log_return_1+aord_log_return_2,
           data=training.set,family = binomial)
scores <- predict(model, newdata = test.set, type= "response")
library(ROCR)
pred.fit.glm <- ifelse(scores>0.5, 1, 0)
library(caret)
confusionMatrix(pred.fit.glm, test.set$snp_log_return_positive, positive = "1")

# Accuracy of 72 %
##########################################

##########################################
# Model 2
model<-glm(snp_log_return_positive~snp_log_return_1+
             nyse_log_return_1+
             djia_log_return_1+
             nikkei_log_return_0+
             hangseng_log_return_0+
             ftse_log_return_0+
             dax_log_return_0+
             aord_log_return_0,
           data=training.set,family = binomial)
scores <- predict(model, newdata = test.set, type= "response")
pred.fit.glm <- ifelse(scores>0.5, 1, 0)
confusionMatrix(pred.fit.glm, test.set$snp_log_return_positive, positive = "1")

# Accuracy of 72.5 %
#########################################

########################################

stock=read.csv("./closing_date_scaled")
set.seed(333)
index<-sample(seq_len(nrow(stock)),size=0.75*nrow(stock))
training.set<-stock[index,]
test.set<-stock[-index,]

################################################
# Model 3
model<-lm(snp_close_scaled~nyse_close_scaled+djia_close_scaled+nikkei_close_scaled+
             hangseng_close_scaled+ftse_close_scaled+
             dax_close_scaled+aord_close_scaled,
           data=training.set)

scores <- predict(model, newdata = test.set)
plot(1:362,test.set$snp_close_scaled)
lines(1:362,scores,col='red')

SSE=sum((scores-test.set$snp_close_scaled)^2)   ## SSE = 0.08123113
SST=sum((mean(training.set$snp_close_scaled)-test.set$snp_close_scaled)^2)   ##  SST  = 9.001377
R2=1-SSE/SST             ## R2 = 0.9909757 
RMSE=sqrt(SSE/nrow(training.set))         #3 RMSE = 0.008652595

# Accuracy = 99.1%
#########################################

########################################
# Model 4
model<-lm(snp_close_scaled~nyse_close_scaled+djia_close_scaled,
          data=training.set)


scores <- predict(model, newdata = test.set)
plot(1:362,test.set$snp_close_scaled)
lines(1:362,scores,col='red')

SSE=sum((scores-test.set$snp_close_scaled)^2)   ## SSE = 0.08123113
SST=sum((mean(training.set$snp_close_scaled)-test.set$snp_close_scaled)^2)   ##  SST  = 9.001377
R2=1-SSE/SST             ## R2 = 0.9909757 
RMSE=sqrt(SSE/nrow(training.set))         #3 RMSE = 0.008652595

# Accuracy = 99.1 %
#######################################################
