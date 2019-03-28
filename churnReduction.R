
rm(list=ls(all=T))
# set working Directory
setwd("D:/MY FIRST EDWISOR PROJECT/My first Edwisor Projects in R")
#install.packages(c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced","C50","dummies",
#    "e1071","MASS","rpart","gbm","ROSE"))
## Loading Imortant Libraries
x=c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced","C50","dummies",
    "e1071","MASS","rpart","gbm","ROSE")

# Loading Packages
lapply(x,require,character.only=T)


### Read Train churn data
churn_train=read.csv("Train_data.csv")


# Read test churn data
churn_test=read.csv("Test_data.csv")


# ## Explore the data
# str(churn_train)
# class(churn_train$number.vmail.messages)
# class(churn_train$total.day.minutes)
# 
### Missing value Analysis
# create data frame with missing percentage
miss_val=data.frame(apply(churn_train,2,function(x){sum(is.na(x))}))
miss_val            # No missing value found


# Check Unique values
unique(churn_train$state)
unique(churn_train$account.length)

# TRAIN DATA:-Data manupulation: convert string categories into factor numeric'

for(i in 1:ncol(churn_train)){
  if(class(churn_train[,i])=="factor"){
    churn_train[,i]=factor(churn_train[,i],
                           labels = (1:length(levels(factor(churn_train[,i])))))
  }
}

# TEST DATA:- Data manupulation: convert string categories into factor numeric'

for(i in 1:ncol(churn_test)){
  if(class(churn_test[,i])=="factor"){
    churn_test[,i]=factor(churn_test[,i],
                           labels = (1:length(levels(factor(churn_test[,i])))))
  }
}


#############################################################################
############################# outlier Analysis ##############################
# seperate numeric columns
numeric_index=sapply(churn_train,is.numeric)
numeric_data=churn_train[,numeric_index]
cnames=colnames(numeric_data)
cnames


# numeric_index=sapply(churn_train_deleted,is.numeric)
# numeric_data=churn_train_deleted[,numeric_index]
# cnames=colnames(numeric_data)
# cnames
# 

## Normality check before removing outliers
# Histogram using ggplot

for(i in 1:length(cnames))
{
  assign(paste0("gn",i),ggplot(churn_train,aes_string(x=cnames[i]))+
           geom_histogram(fill="cornsilk",colour="black")+
           geom_density()+
           theme_bw()+
           labs(x=cnames[i])+
           ggtitle(paste("Histogram of ",cnames[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,gn11,gn12,ncol=2)
gridExtra::grid.arrange(gn13,gn14,gn15,gn16,ncol=2)


## outlier Analysis
# Boxplot distribution and outlier check

for(i in 1:length(cnames)){
  assign(paste0("gn",i),ggplot(churn_train,aes_string(y=cnames[i],x="Churn",
                              fill=churn_train$Churn))+
           geom_boxplot(outlier.colour = "red",fill="skyblue",outlier.shape = 18,
                        outlier.size = 1,notch = F)+
           theme_bw()+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box Plot of Churn for",cnames[i])))
}


# 
# for(i in 1:length(cnames)){
#   assign(paste0("gn",i),ggplot(churn_train_deleted,aes_string(y=cnames[i],x="Churn",
#                                                       fill=churn_train_deleted$Churn))+
#            geom_boxplot(outlier.colour = "red",fill="skyblue",outlier.shape = 18,
#                         outlier.size = 1,notch = F)+
#            theme_bw()+
#            labs(y=cnames[i],x="Churn")+
#            ggtitle(paste("Box Plot of Churn for",cnames[i])))
# }
# 

# Plotting the BoxPlot together
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,gn11,gn12,ncol=2)
gridExtra::grid.arrange(gn13,gn14,gn15,gn16,ncol=2)


##### Remove outliers using boxplot
df=churn_train
# #churn_train=df
# # Loop to remove oultiers from all the variables
# for(i in cnames){
#   print(i)
#   val=churn_train_deleted[,i][churn_train_deleted[,i] %in%
#                         boxplot.stats(churn_train_deleted[,i])$out]
#   print(length(val))
#   churn_train_deleted=churn_train_deleted[which(!churn_train_deleted[,i]%in% val),]
# }
# 



for(i in cnames){
  print(i)
  val=churn_train[,i][churn_train[,i] %in%
                                boxplot.stats(churn_train[,i])$out]
  print(length(val))
  churn_train=churn_train[which(!churn_train[,i]%in% val),]
}


## Normality check after removing outliers
# Histogram using ggplot

for(i in 1:length(cnames))
{
  assign(paste0("gn",i),ggplot(churn_train,aes_string(x=cnames[i]))+
           geom_histogram(fill="cornsilk",colour="black")+
           geom_density()+
           theme_bw()+
           labs(x=cnames[i])+
           ggtitle(paste("Histogram of ",cnames[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,gn11,gn12,ncol=2)
gridExtra::grid.arrange(gn13,gn14,gn15,gn16,ncol=2)



############################################################################
############################ Feature Selection #############################  
# selecting numeric variable 
numeric_index=sapply(churn_train,is.numeric)
cnames

#### correlation plot
corrgram(churn_train,order = F,lower.panel=panel.shade,
         upper.panel = panel.pie,text.panel = panel.txt,
         main="Correlation Plot")

#correlation :- 
#     (Total day min - Total day charge), (Total eve min- Total eve charge)
#     (Total Night min - Total Night Charge) , (Total Int min-Total int charge)



## Chi square test of independence
factor_index=sapply(churn_train,is.factor)
factor_data=churn_train[,factor_index]
for(i in 1:5){
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}
# Remove Phone. numbebr

###### TRAIN DATA:- Dimension Reduction
# Reduct all the minutes
#churn_train_deleted=subset(churn_train,select=-c(phone.number, total.day.minutes,
#                    total.eve.minutes, total.night.minutes,total.intl.minutes))

# Reduct all the charges
churn_train_deleted=subset(churn_train,select=-c(phone.number, total.day.charge,
                     total.eve.charge, total.night.charge,total.intl.charge))


###### TEST DATA:- Dimension Reduction
# Reduct all the minutes
#churn_test_deleted=subset(churn_test,select=-c(phone.number, total.day.minutes,
#                                                 total.eve.minutes, total.night.minutes,total.intl.minutes))
# Reduct all the charges
churn_test_deleted=subset(churn_test,select=-c(phone.number, total.day.charge,
                                                 total.eve.charge, total.night.charge,total.intl.charge))


#### correlation plot after dimension reduction
corrgram(churn_train_deleted,order = F,lower.panel=panel.shade,
         upper.panel = panel.pie,text.panel = panel.txt,
         main="Correlation Plot")
#############################################################################
############################## Feature scaling ##############################

## Normality check 
# Histogram using ggplot
numeric_index_deleted=sapply(churn_train_deleted,is.numeric)
numeric_data_deleted=churn_train_deleted[,numeric_index_deleted]
cnames2=colnames(numeric_data_deleted)
cnames2
for(i in 1:length(cnames2))
{
  assign(paste0("gn",i),ggplot(churn_train_deleted,aes_string(x=cnames2[i]))+
           geom_histogram(fill="cornsilk",colour="black")+
           geom_density()+
           theme_bw()+
           labs(x=cnames2[i])+
           ggtitle(paste("Histogram of ",cnames2[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,gn11,gn12,ncol=2)

# classify which features are normally distributed

#  normally Distributed               Non uniformally Distributed
# 1)account.length                      1) area.code
# 2)total.day.calls                     2) number.vmail.message
# 3)total.day.charge                    3) total.intl.calls
# 4)total.eve.calls                     4) number.customer.services
# 5)total.eve.charge
# 6)total.night.calls
# 7)total.night.charge
# 8)total.intl.charge


## TRAIN DATA:- Normalization for Non uniformly distributed features

cnames3=c("area.code","number.vmail.messages","total.intl.calls",
          "number.customer.service.calls")

for(i in cnames3){
  print(i)
  churn_train_deleted[,i]=(churn_train_deleted[,i]-min(churn_train_deleted[,i]))/
    (max(churn_train_deleted[,i]-min(churn_train_deleted[,i])))
}

# for(i in cnames3){
#   print(i)
#   churn_train[,i]=(churn_train[,i]-min(churn_train[,i]))/
#     (max(churn_train[,i]-min(churn_train[,i])))
# }
# 


## TEST DATA:-Normalization for Non uniformly distributed features



for(i in cnames3){
  print(i)
  churn_test_deleted[,i]=(churn_test_deleted[,i]-min(churn_test_deleted[,i]))/
    (max(churn_test_deleted[,i]-min(churn_test_deleted[,i])))
}


## TRAIN DATA:- Standardization of uniformally distributed data 
# #cnames3=c("account.length","total.day.calls","total.day.charge",
#           "total.eve.calls","total.eve.charge","total.night.calls",
#           "total.night.charge","total.intl.charge")

cnames3=c("account.length","total.day.calls","total.day.minutes",
          "total.eve.calls","total.eve.minutes","total.night.calls",
          "total.night.minutes","total.intl.minutes")
# #cnames3=c("account.length","total.day.calls","total.day.minutes",
#           "total.eve.calls","total.eve.minutes","total.night.calls",
#           "total.night.minutes","total.intl.minutes","total.day.calls","total.day.charge",
#           "total.eve.calls","total.eve.charge","total.night.calls",
#           "total.night.charge","total.intl.charge")

for(i in cnames3){
  print(i)
  churn_train_deleted[,i]=(churn_train_deleted[,i]-mean(churn_train_deleted[,i]))/
    sd(churn_train_deleted[,i])
}

## TEST DATA:- Standardization of uniformally distributed data 
for(i in cnames3){
  print(i)
  churn_test_deleted[,i]=(churn_test_deleted[,i]-mean(churn_test_deleted[,i]))/
    sd(churn_test_deleted[,i])
}

# 
#write.csv(churn_test_deleted,"churn_test_deleted.csv",row.names = F)



###############################################################################
##############################  Decision Tree  ################################
############################################################################

# Clean the environment
# #install.packages("DataCombine")
# library(DataCombine)
# rmExcept(c("churn_train_deleted","churn_test_deleted"))

# ## Decision tree
# # Develop model on training data
# C5.0_model=C5.0(Churn~.,churn_train_deleted,trials=100,rules=T)
# 
# 
# 
# # Summary of data
# summary(C5.0_model)
# 
# 
# # Write rules into disk
# write(capture.output(summary(C5.0_model)),"C50Rules_churn.txt")
# 
# 
# # Lets Predict for the test cases
# C5.0_predictions=predict(C5.0_model,churn_test_deleted[-16],type = "class")
# 
# ############################################################################
# ################### Evaluate the performance of model ######################
# conf_matrix_churn=table(churn_test_deleted$Churn,C5.0_predictions)
# 
# confusionMatrix(conf_matrix_churn)


########## by excluding all the minutes (day,night,eve,international) ####              

# Accuracy: 91.78%

# False negative Rate= 0%
#FPR=FP/(FP+TP)
#FPR=0/(0+1143)

# Flase positive Rate=61.16%
#FNR=FN/(FN+TN)
#FNR=137/(137+87)




# Without scaling & with outliers
#C5.0_predictions
#    1    2
#1 1436    7
#2   64  160
#64/(64+160)=FPR=28.57%
#Accuracy : 0.9574          


###########  by excluding all the charges (day,night,eve,international) ##

# Accuracy: 93.40%

# False Positive rate= 0.2%
#FPR=FP/(FP+TP)
#FPR=3/(3+1140)

# False negative Rate= 47.76%
#FNR=FN/(FN+TN)
#FNR=107/(107+117)



#####################  Without excluding any variable     ###########################

# Accuracy=87.46%

# error= 12.44%

#FPR=1.52% - should be decreased

# Rcall=1-FNR= 98.48% - Should be increased (Positive case)

# FNR=83.48% - should be decreased

# Specificity=1-FPR=16.52% - Should be increased (Negative case)



# Without Removing outliers
#confusionMatrix(conf_matrix_churn)

#C5.0_predictions
#    1    2
#1  307 1136
#2   10  214
#Accuracy : 0.3125    




#C5.0_predictions
#    1    2
#1 1413   30
#2  174   50

#Accuracy : 0.8776         

#174/(174+50)

#############by excluding all the minutes (day,night,eve,international) ####              
# by removing outliers 
#Confusion Matrix and Statistics

#C5.0_predictions
#     1    2
#1 1442    1
#2  104  120

#Accuracy : 0.937           



#########  by excluding all the charges (day,night,eve,international) #####
## bY REMOVING OUTLIERS
#C5.0_predictions
#    1    2
#1 1442    1
#2  107  117

#Accuracy : 0.9352    



#########  by excluding all the charges (day,night,eve,international) #####
## WITHOUT REMOVING OUTLIERS
#C5.0_predictions
#    1    2
#1 1438    5
#2   66  158

#Accuracy : 0.9574 



#########  by excluding all the charges (day,night,eve,international) #####
## WITHOUT REMOVING OUTLIERS but with feature scaling
#C5.0_predictions
#     1    2
#1 1353   90
#2   97  127
#97/(97+127)=FNR=43.30
#Accuracy : 0.8878          



## By excluding all the minutes and apply all data preprocessing in sequence
#C5.0_predictions
#     1    2
#1 1442    1
#2  108  116

#Accuracy : 0.9346      



# By excluding all the charges and apply all data preprocessing in sequence
#C5.0_predictions
#     1    2
#1 1440    3
#2  107  117

#Accuracy : 0.934          

##############################################################################
######################## Random Forest #######################################
#############################################################################
# # When we run Model next time our prediction will be same
# set.seed(1234)
# 
# # Random Forest
# RF_model=randomForest(Churn~.,churn_train_deleted,importance=T,ntree=500)
# 
# RF_model
# 
# # Extarct rules from RF
# # transform rf object to an trees format
# library(inTrees)
# treelist=RF2List(RF_model)
# 
# # Extract rules
# exec=extractRules(treelist,churn_train_deleted[,-16])
# 
# # Visualize some rules
# exec[1:2,]
# 
# # Make rules for more readable
# readableRules=presentRules(exec,colnames(churn_train_deleted))
# 
# readableRules[1:2,]
# 
# # Get rules metrices
# ruleMetric=getRuleMetric(exec,churn_train_deleted[,-16],churn_train_deleted$Churn)
# 
# # Evaluate few rules
# ruleMetric[1:2,]
# 
# # predict test data using random forest model
# 
# RF_predictions=predict(RF_model,churn_test_deleted[,-16])
# 
# 
# 
# 

#################################   Model Evaluation of Random Forest  #############################
# Evaluate the performance of classification model
# 
# conf_matrix_RF=table(churn_test_deleted$Churn,RF_predictions)
# 
# confusionMatrix(conf_matrix_RF)

# False Negative rate
#FNR=FN/(FN+TP)=45.53

# # Evaluation
# Accuracy : 92.26 
# 
# RF_predictions
# 1    2
# 1 1416   27
# 2  102  122

######## by excluding all the minutes (day,night,eve,international)  #############
# without scaling
# RF_predictions  # with 100 trees
# 1    2
# 1 1424   19
# 2   64  160

# Accuracy : 0.9502          
# FPR= 0.2857143
# 
# # with 500 trees
# RF_predictions
# 1    2
# 1 1423   20
# 2   66  158
# 
# Accuracy : 0.9484     
# 
# 
# #########by excluding all the minutes (day,night,eve,international) ####              
# # by removing outliers 
# RF_predictions
# 1    2
# 1 1436    7
# 2  115  109
# 
# Accuracy : 0.9268    
# 
# 

#########  by excluding all the charges (day,night,eve,international) #####
## bY REMOVING OUTLIERS
# RF_predictions
# 1    2
# 1 1432   11
# 2  113  111
# 
# Accuracy : 0.9256   


#########  by excluding all the charges (day,night,eve,international) #####
## WITHOUT REMOVING OUTLIERS
# RF_predictions
#     1    2         #  ntrees= 500
# 1 1422   21
# 2   67  157
# 
# Accuracy : 0.9472          


       
#########  by excluding all the charges (day,night,eve,international) #####
## WITHOUT REMOVING OUTLIERS but with feature scaling
# RF_predictions
#     1    2
# 1 1341  102
# 2  105  119
# 
# Accuracy : 0.8758         
############################################################################
######################## Logistic regression  ##############################
#############################################################################
# # Logistic regression
# Logit_model=glm(Churn~.,data = churn_train_deleted,family = "binomial")
# 
# # Summary
# summary(Logit_model)
# 
# 
# # predict using logistic regression
# 
# logit_predictions=predict(Logit_model,newdata=churn_test_deleted,
#                           type="response")
# 
# # convert prob
# logit_predictions=ifelse(logit_predictions>0.5,1,0)
# 
# # evaluate the performance of classification
# 
# conf_matrix_LR=table(churn_test_deleted$Churn,logit_predictions)
# conf_matrix_LR
# # False Negative rate
# FNR=FN/(FN+TP)
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling
# conf_matrix_LR
# logit_predictions
# Accuracy:87.04%
# 0    1
# 1 1396   47
# 2  169   55
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling and by removing outliers
# logit_predictions
# 0    1
# 1 1412   31
# 2  164   60
# Accuracy:88.54%
# 
# 
# 
# ### by excluding all the charges (day,night,eve,international)  #############
# # without scaling and without removing outliers
# logit_predictions
# 0    1
# 1 1396   47
# 2  169   55
# 
# 
# 
# ### by excluding all the charges (day,night,eve,international)  #############
# #  without removing outliers but with feature scaling
# logit_predictions
#     0    1
# 1 1416   27
# 2  187   37
###########################################################################################################
################################         KNN       #############################################################
############################################################################################################
# Knn Implementation
library(class)

# predict test data
knn_predictions=knn(churn_train_deleted[,1:16],churn_test_deleted[,1:16],
                    churn_train_deleted$Churn,k=5)

# confusion matrix

conf_matrix_knn=table(knn_predictions,churn_test_deleted$Churn)

confusionMatrix(conf_matrix_knn)

###############################  Check  Accuracy   of KNN    ##################################

#By excluding all the charges and apply all data preprocessing in sequence
#     
#                knn_predictions    1    2
#                               1 1435  175         for k= 3
#                               2    8   49                   
#                               Accuracy : 0.8902         
#                             FNR=8/(8+49)=14.03%             
#                             FPR=175/(175+1435)=10.86% 

#             knn_predictions      1    2 
#                             1  1443  190             for k=5
#                             2    0   34
# 
#                           Accuracy : 0.886        
#                           FNR=0%
#                           FPR=11.69%

# sum(daig(conf_matrix_knn))/nrow(churn_test_deleted)
# sum(daig(conf_matrix_knn))/nrow(churn_test)
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling
# knn_predictions    1    2
#                1 1398  201
#                2   45   23
# 
# Accuracy : 0.8524 
# 
# # False Negative rate
# FNR=FN/(FN+TP)
# 
# # without removing outliers
# knn_predictions    1    2
# 1 1054  148
# 2  389   76
# 
# Accuracy : 0.6779       
# 
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling and by removing outliers
# knn_predictions    1    2
# 1 1420  203
# 2   23   21
# 
# Accuracy : 0.8644    
# 
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling and by removing outliers
# knn_predictions    1    2
# 1 1412  166
# 2   31   58
# 
# Accuracy : 0.8818   
# 
# 
# 
# 
# ### by excluding all the charges (day,night,eve,international)  #############
# # without scaling and removing outliers
# 
# n_predictions    1    2
# 1 1389  162
# 2   54   62
# 
# Accuracy : 0.8704       
# 
# 
# 
# 
# ### by excluding all the charges (day,night,eve,international)  #############
# # with scaling and without removing outliers
# knn_predictions    1    2
#                1 1426  168
#                2   17   56
# 
# Accuracy : 0.889          
# 17/(17+56)
# 
# 
# #### by excluding all the charges (day,night,eve,international)  #############
# # with scaling and  removing outliers not in sequence
# knn_predictions    1    2
#                1 1436  172
#                2    7   52
#                Accuracy : 0.8926  
#                FNR=7/(7+52)=11.86
#                FPR=172/(172+1436)=10.69
#                           
#  
#              
#                  
# 
###########################################################################
############################  Naive Bays  #################################
###########################################################################
# # Naive Bayes
# library(e1071)
# 
# # Develop Model
# NB_model=naiveBayes(Churn~.,data=churn_train_deleted)
# 
# # Predict on test case Draw
# NB_predictions=predict(NB_model,churn_test_deleted[,1:16],type = "class")
# 
# # Look at confusion matrix
# conf_matrix_NV=table(observed=churn_test_deleted[,16],
#                      predicted=NB_predictions)
# 
# 
# confusionMatrix(conf_matrix_NV)
# 
# # Statical way:- Accuracy
# mean(NB_predictions==churn_test_deleted$Churn)
# 
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling
# 
#           predicted
# observed    1    2
# 1 1405   38
# 2  160   64
# 
# Accuracy : 0.8812   
#                          
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling and by removing outliers
# 
#           predicted
# observed    1    2
#   1        1427   16
#   2         171   53
# 
# Accuracy : 0.8878      
# 
# 
# 
# 
# ### by excluding all the minutes (day,night,eve,international)  #############
# # without scaling and by removing outliers
# predicted
# observed    1    2
#          1 1427   16
#          2  171   53
# 
# Accuracy : 0.8878          
# 
# 
# 
# ### by excluding all the charges (day,night,eve,international)  #############
# # without scaling and removing outliers.
#           predicted
# observed    1    2
#         1 1405   38
#         2  160   64
# 
# Accuracy : 0.8812      
# 
# 
# 
# ### by excluding all the charges (day,night,eve,international)  #############
# # with scaling and without removing outliers.
# predicted
# observed    1    2
# 1 1336  107
# 2  111  113
# 
# Accuracy : 0.8692      