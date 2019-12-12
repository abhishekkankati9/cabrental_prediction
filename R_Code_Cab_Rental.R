#To clean the environment
rm(list=ls())

#To set up our working directory
setwd("E:/edWisor/Project2/Cab Rental/train_cab")

#To check if our working directory is correct
getwd()

#To load the necessary packages required for our project
lib_req = c('readxl','caret','dplyr','class','randomForest','leaps','cowplot','ggplot2','rpart','DMwR','corrgram')
#This statement will return true if the library is found and loaded successfully
lapply(lib_req, require, character.only = TRUE)

#To load our data both train and test
cab = read.csv("train_cab.csv", header=T, na.strings=c(" ","","NA","430-"))
cab_test = read.csv("test.csv") #test dataset
cab = as.data.frame(cab) #transforming the data into a dataframe
cab_test = as.data.frame(cab_test)

###########################################################EXPLORING THE DATA####################################################

#Understanding of data viz., view the data, strucuture, dimension, summary, column names of the data respectively
View(cab) #To view the data
str(cab) #All the variables are given as numericals except for fare_amount(actually numerical) and pickup_datetime which are to be converted
dim(cab) #16067 rows 7 columns
summary(cab) #To check the basic quantitative stats
colnames(cab)#to check all the column names
#We have observed that a missing value exists in our pickup_datetime variable let's remove it
cab=cab[-1328,] #as date column is invalid here

#train_cab$pickup_latitude > 90 have 1 observation
#Na values for fare_amount<0
#na values for passengercount < 1 and  > 6
#subsetting of the data without latitudes and longitudes = 0
cab = subset(cab,cab$pickup_latitude >= -90 & cab$pickup_latitude <= 90)
dim(cab)#16065 7
cab = subset(cab,cab$dropoff_latitude >= -90 & cab$dropoff_latitude <= 90)
dim(cab)#16065 7
cab = subset(cab,cab$pickup_longitude >= -180 & cab$pickup_longitude <= 180)
dim(cab)#16065 7
cab = subset(cab,cab$dropoff_longitude >= -180 & cab$dropoff_longitude <= 180)
dim(cab)#16065 7(NA values are 80 as of now)
#cab_train$fare_amount[cab_train$fare_amount>1000]=NA
#Let us set the limits of passenger_count and fare_amount
cab$passenger_count[cab$passenger_count > 6]=NA
cab$passenger_count[cab$passenger_count < 1]=NA
cab$fare_amount[cab$fare_amount < 0]=NA
dim(cab)#missing values as of now 161 with dim as 16065 7
sum(is.na(cab))

#################################################Data Types Conversion######################################################
#fare_amount should be a numerical type
#pickup_datetime should be converted to date type and thereby extracting day,month,year and hour as new variables
#By using latitude and longitudes, let's find the new variable distance_travelled and drop the four variables
cab$fare_amount = as.numeric(cab$fare_amount) #Converting from factor to numeric type
cab$pickup_datetime = strptime(cab$pickup_datetime,"%Y-%m-%d %H:%M:%S", "UTC")
cab$Year = format(cab$pickup_datetime,"%Y")
cab$Month = format(cab$pickup_datetime,"%m")
cab$Date = format(cab$pickup_datetime,"%d")
cab$Hour = format(cab$pickup_datetime,"%H")
dim(cab) #(16065 11)
View(cab)

#Let us find the distance using latitudes and longitudes of pickup and drop by haversine formula

haversine <- function(long1, lat1, long2, lat2)
{
  
  # convert degrees to radians 
  radian <- function(deg)
  {
    rad = (deg*22)/(7*180)
    return (rad)
  }
  
  long1 = radian(long1)
  lat1 = radian(lat1)
  long2 = radian(long2)
  lat2 = radian(lat2)
  
  
  # haversine formula 
  dlon = long2 - long1 
  dlat = lat2 - lat1 
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a)) 
  r = 6371 # Radius of earth in kilometers. Use 3956 for miles
  return (c * r)
}

#To create a variable distance_travelled and fill the values of that variable
cab$distance_travelled = haversine(cab$pickup_longitude,cab$pickup_latitude,cab$dropoff_longitude,cab$dropoff_latitude)
cab$distance_travelled = as.numeric(cab$distance_travelled) #To convert the new variable to numeric type
dim(cab)#16065 12 with 161 NA values

#Now let us remove the latitude and longitude variables and pickup_datetime as new variables have been derived from them
cab = cab[,-c(2,3,4,5,6)]
dim(cab) #16065 7
View(cab)

#Now we need to replace zeros with NA values
cab[cab == 0] = NA # We are having 621 NA values in our dataset
View(cab)

#Numeric Variables and Categorical variables
num_var = c('fare_amount','passenger_count','distance_travelled')
cat_var = c('Year','Month','Date','Hour')

#Transforming categorical variables to factor type
for (k in cat_var){
  cab[,k] = factor(cab[,k],labels = 1:length(levels(factor(cab[,k]))))
}

#str(cab) 3 cont and 4 cate
#####################################################MISSING VALUE ANALYSIS###########################################
x = sapply(cab,function(x){sum(is.na(x))})
missing_values = data.frame(x)

#To get the row names as new column
missing_values$Variables = row.names(missing_values)

#Reset the rownames i.e., setting the index values
row.names(missing_values) = NULL

#Rename the column 1 name
names(missing_values)[1] = "Missing_percentage"

#Calculate the missing values percentage and if it is > 30%, we should remove that column from our data
missing_values$Missing_percentage= ((missing_values$Missing_percentage/nrow(cab))*100)

#Reorder the columns
missing_values = missing_values[,c(2,1)]

#Sorting the rows in descending order w.r.t. missing percentage values
missing_values = missing_values[order(-missing_values$Missing_percentage),]
View(missing_values) # We have found missing values in distance_travelled and passenger_count

#Bar plot to visualise top 7 missing percentage values
ggplot(data = missing_values[1:3,], mapping = aes(x = reorder(Variables, -Missing_percentage),y = Missing_percentage)) + 
  geom_bar(stat = 'identity', fill = 'blue') + xlab("Variable Name") + ylab("Missing Percentage") + 
  ggtitle("Missing Value Percentages") + theme_classic()

#Imputing the missing values
#Creating a missing value and check which method will give us the nearest value
#cab[["fare_amount"]][522] #Actual value = 5.5
#cab[["fare_amount"]][522] = NA #Creating a missing value to check the best method for missing value imputation
#cab[["fare_amount"]][522] = mean(cab[["fare_amount"]], na.rm = T) #Mean value = 15.02
#cab[["fare_amount"]][522] = median(cab[["fare_amount"]], na.rm = T) #Median value = 8.5
cab = knnImputation(cab,k=3,meth = "median", distData = NULL) #Knn value = 13.166, hence median 8.5

#Now our cab_train is free from NA values
#################################################EXPLORE THE DISTRIBUTION OF VARIABLES USING THE PLOTS#############################################

#Get the data with only numeric columns
num_index = sapply(cab, is.numeric)
numeric_data = cab[,num_index]
View(numeric_data)
View(cab) 

#Get the data with only category data(factor) columns
cat_data = cab[,!num_index]

#For continuous variables let's use Histograms to represent the data
hist_fareamount = ggplot(data = numeric_data,aes(x =numeric_data[,1])) + 
  ggtitle("Fare Amount") + geom_histogram(bins = 30)
hist_passengercount = ggplot(data = numeric_data, aes(x = numeric_data[,2])) +
  ggtitle("Distribution of Passengers") + geom_histogram(bins = 30)
hist_distancetravelled = ggplot(data = numeric_data, aes(x = numeric_data[,3])) +
  ggtitle("Distance") + geom_histogram(bins = 30)


#To arrange all the plots of numerical variables distribution in one page
gridExtra::grid.arrange(hist_fareamount,hist_passengercount,hist_distancetravelled,ncol = 2)

#Distribution of factor(categorical) data using bar plot
bar_year = ggplot(data = cat_data, aes(x = cat_data[,1])) + geom_bar() + 
  ggtitle("Year") + theme_bw()
bar_month = ggplot(data = cat_data, aes(x = cat_data[,2])) + geom_bar() + 
  ggtitle("Month") + theme_bw()
bar_date = ggplot(data = cat_data, aes(x = cat_data[,3])) + geom_bar() + 
  ggtitle("Date") + theme_bw()
bar_hour = ggplot(data = cat_data, aes(x = cat_data[,4])) + geom_bar() + 
  ggtitle("Hour") + theme_bw()

#Plotting all the above plots in one page
gridExtra::grid.arrange(bar_year,bar_month,ncol = 2)
gridExtra::grid.arrange(bar_date,bar_hour,ncol = 2)

######################################################## OUTLIER ANALYSIS ###########################################################

#Get the data with only numeric columns
num_index = sapply(cab, is.numeric)
numeric_data = cab[,num_index]
View(numeric_data)

#Get the data with only factor columns
cat_data = cab[,!num_index]
View(cat_data)

#Check for outliers using boxplots
for(i in 1:ncol(numeric_data)) {
  assign(paste0("boxplot",i), ggplot(data = cab, aes_string(y = numeric_data[,i])) +
           stat_boxplot(geom = "errorbar", width = 0.75) +
           geom_boxplot(outlier.colour = "red", fill = "blue", outlier.size = 1) +
           labs(y = colnames(numeric_data[i])) +
           ggtitle(paste("Boxplot: ",colnames(numeric_data[i]))))
}


gridExtra::grid.arrange(boxplot1,boxplot2,boxplot3,ncol=2)

#Imputing NAs into Outliers

#Check the number of missing values
for(i in colnames(numeric_data)){
  val = cab[,i][cab[,i] %in% boxplot.stats(cab[,i])$out]
  print(paste(i,length(val)))
  cab[,i][cab[,i] %in% val] = NA
}

#Imputing NAs with values using KNN Imputation method
cab = knnImputation(cab,k=3,meth = "median", distData = NULL)
#To ensure no NA values are left in our dataset
sum(is.na(cab))
View(cab)


#########################################FEATURE SELECTION#################################################
#Checking for multicollinearity for continuous variables using correlation plot
#We have observed that none of the variables are high correlated to each other

cor = cor(numeric_data)
corrgram(cor,type = 'cor',lower.panel = panel.pie,diag.panel = panel.density,upper.panel = panel.conf,text.panel = panel.txt)

#cab_train is now the cleaned data with no missing values, no outliers and no multicolinearity in continuous variables

#Chisquared test of independence
factor_idex = sapply(cab, is.factor)
factor_data = cab[,factor_idex]
View(factor_data)

for (i in 1:length(colnames(factor_data))){
  print(names(factor_data[i]))
  print(chisq.test(table(cab$fare_amount,factor_data[,i])))}


########################################################FEATURE SCALING#########################################################
#As the numeric data is not uniformly distributed we will scale the data by Normalisation

norm_var = c("distance_travelled","passenger_count")
for (i in norm_var){
  cab[,i] = (cab[,i] - min(cab[,i]))/
    (max(cab[,i] - min(cab[,i])))
}

#To check the data after all the preprocessing techniques
print(head(cab))
View(cab)

##########################################################DEVELOPMENT OF THE MODEL###################################################
#Divide the data into test and train
#Train data
set.seed(1)
train_index = sample(1:nrow(cab),0.8*nrow(cab)) #By random Sampling
train_data = cab[train_index,]
dim(train_data) #12852 7
test_data = cab[-train_index,]
dim(test_data) #3213 7

#####################################################MACHINE LEARNING MODELS#########################################################

##################LINEAR REGRESSION###############
#RMSE : 2.242702
#MAPE : 1.584502
#R-squared:  0.7173
#Adjusted R-squared:  0.7157

linear_m = lm(fare_amount~.,data = train_data) #here Y variable is fare_amount
summary(linear_m)

#Residual standard error: 0.09911 on 12779 degrees of freedom
#Multiple R-squared:  0.7173,	Adjusted R-squared:  0.7157 
#F-statistic: 450.3 on 72 and 12779 DF,  p-value: < 2.2e-16

#Predict for new test cases
predict_linear = predict(linear_m,test_data[-1])
RMSE(predict_linear,test_data$fare_amount)
#Creating a function to find MAPE
MAPE = function(actual,predicted){
  mean(abs(actual - predicted))
}
MAPE(test_data$fare_amount,predict_linear)

#Plot a graph for actual vs predicted values
plot(test_data$fare_amount,type="l",lty=2,col="green")
lines(predict_linear,col="blue")

##################DECISION TREE######################

#Build decision tree model using rpart
#RMSE : 2.470647
#MAPE : 1.808818
#R-Square : 0.441103635089046
#AdjRsquare : 0.440859985929528

decision_m = rpart(fare_amount~.,data = train_data, method = "anova")
predict_dt = predict(decision_m,test_data[,2:7])
RMSE(predict_dt,test_data$fare_amount)
MAPE(test_data$fare_amount,predict_dt)
plot(test_data$fare_amount,type="l",lty=2,col="green")
lines(predict_linear,col="blue")

#R-squared
rsq = function(act, pred){
  (1-(sum((act-pred)**2)/sum((mean(act)-pred)**2)))
}

#Adjusted R-squared
adjrsq = function(rsq,n,k){
  (1-((1-rsq)*(n-1)/(n-k-1)))
}

rsquared = rsq(test_data$fare_amount,predict_dt)
adjustedrsquare = adjrsq(rsquared,nrow(cab),ncol(cab))
print(paste("R-squared - ",rsquared))
print(paste("Adjusted R-squared - ",adjustedrsquare))

#################RANDOM FOREST#######################
#RMSE : 2.470647
#MAPE : 1.67049
#R-Square : 0.441103635089046
#Adj R Square : 0.440859985929528

random_m = randomForest(fare_amount~.,data = train_data, ntree = 200)
predict_random = predict(random_m,test_data[-1])
RMSE(predict_dt,test_data$fare_amount)
MAPE(test_data$fare_amount,predict_random)
rsquared_rf = rsq(test_data$fare_amount,predict_random)
adjustedrsquare_rf = adjrsq(rsquared_rf,nrow(cab),ncol(cab))
print(paste("R-squared - ",rsquared))
print(paste("Adjusted R-squared - ",adjustedrsquare))
plot(test_data$fare_amount,type="l",lty=2,col="green")
lines(predict_linear,col="blue")

#Mean of squared residuals: 5.116293
#% Var explained: 69.65

########################################TEST OUTPUT###################################
#To view the given test dataset
View(cab_test)
str(cab_test)
dim(cab_test) #9914 6
#To convert timestamp to year,month,date and hour respectively
cab_test$pickup_datetime = strptime(cab_test$pickup_datetime,"%Y-%m-%d %H:%M:%S", "UTC")
cab_test$Year = format(cab_test$pickup_datetime,"%Y")
cab_test$Month = format(cab_test$pickup_datetime,"%m")
cab_test$Date = format(cab_test$pickup_datetime,"%d")
cab_test$Hour = format(cab_test$pickup_datetime,"%H")
dim(cab_test) #(9914 10)
View(cab_test)

#Let us find the distance using latitudes and longitudes of pickup and drop by haversine formula

haversine <- function(long1, lat1, long2, lat2)
{
  
  # convert degrees to radians 
  radian <- function(deg)
  {
    rad = (deg*22)/(7*180)
    return (rad)
  }
  
  long1 = radian(long1)
  lat1 = radian(lat1)
  long2 = radian(long2)
  lat2 = radian(lat2)
  
  
  # haversine formula 
  dlon = long2 - long1 
  dlat = lat2 - lat1 
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a)) 
  r = 6371 # Radius of earth in kilometers. Use 3956 for miles
  return (c * r)
}

#To create a variable distance_travelled and fill the values of that variable
cab_test$distance_travelled = haversine(cab_test$pickup_longitude,cab_test$pickup_latitude,cab_test$dropoff_longitude,cab_test$dropoff_latitude)
cab_test$distance_travelled = as.numeric(cab_test$distance_travelled) #To convert the new variable to numeric type
dim(cab_test)#9914 11

#Now let us remove the latitude and longitude variables and pickup_datetime as new variables have been derived from them
cab_test = cab_test[,-c(1,2,3,4,5)]
dim(cab_test) #16065 6
View(cab_test)

#Divide cont and cat vars
num_var_test = c('passenger_count','distance_travelled')
cat_var_test = c('Year','Month','Date','Hour')

#Transforming categorical variables to factor type
for (k in cat_var_test){
  cab_test[,k] = factor(cab_test[,k],labels = 1:length(levels(factor(cab_test[,k]))))
}



#Normalisation of distance_travelled
norm_var = c("distance_travelled","passenger_count")
for (i in norm_var){
  cab_test[,i] = (cab_test[,i] - min(cab_test[,i]))/
    (max(cab_test[,i] - min(cab_test[,i])))
}

#To check the data after all the preprocessing techniques
print(head(cab_test))
View(cab_test)

#To predict fare_amount by using LR model
fareamount_test = predict(linear_m,cab_test)
cab_test$fare_amount = fareamount_test
View(cab_test)

#Write output
write.csv(cab_test,"R_Test_Output_AK.csv",row.names = FALSE)



