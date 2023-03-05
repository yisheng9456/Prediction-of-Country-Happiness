library(dplyr)
happiness_raw <- read.csv("world_happiness_report.csv")
## Only select the columns that the models need
happiness_raw <- subset(happiness_raw, select = -c(Positive.affect, Negative.affect))
## To check the datasets has missing value
any(is.na(happiness_raw)) 
##To check the datasets has duplicated
anyDuplicated(happiness_raw)

## calculate number of rows has null value
nrow(happiness_raw)
## remove the na values
happiness_raw <-na.omit(happiness_raw)
nrow(happiness_raw)
any(is.na(happiness_raw)) 
## To check the outliers
summary(happiness_raw)

boxplot(happiness_raw$Life.Ladder, main = "Life Ladder")
boxplot(happiness_raw$Log.GDP.per.capita, main = "GDP Per Capital")
## We may see some outliers at below columns. We could not treat them as outliers as the the values are measured based on each countries.
## The value will differs between the countries are well developed, developing and under developed 
## Hence, we will see some of the value will be out of the box plot

boxplot(happiness_raw$Social.support, main = "Social Support")
boxplot(happiness_raw$Healthy.life.expectancy.at.birth, main = "Healthy life expectancy at birth")
boxplot(happiness_raw$Freedom.to.make.life.choices, main = "Freedom to make life choices")
boxplot(happiness_raw$Generosity, main = "Generosity")
boxplot(happiness_raw$ Perceptions.of.corruption, main = "Perceptions of corruption")
##total omitted rows is 237. It occupied the total data is 12.2%
happiness_raw <-mutate(happiness_raw,Rank = if_else(Life.Ladder >7,"High",if_else(Life.Ladder <3.5,"Low","Medium")))

#--------------------------------------------------BORDER LINE (EDA)--------------------------------------------------
# EXPLORE

# load relevant library
library(ggplot2)
library(reshape2)
library(ggpubr)
library(CatEncoders)
library(smotefamily)

# Explore the structure of the dataframe
str(happiness_raw)

# Explore the max and min happiness in the dataset
max_happiness_summary <- happiness_raw %>% group_by(Country.name) %>% summarise (max_happiness=max(Life.Ladder)) %>% arrange(desc(max_happiness))
min_happiness_summary <- happiness_raw %>% group_by(Country.name) %>% summarise (min_happiness=min(Life.Ladder)) %>% arrange(min_happiness)
Top_6_highest <- head(max_happiness_summary)
Top_6_lowest <- head(min_happiness_summary)

ggplot(Top_6_highest, aes(x=reorder(Country.name, max_happiness), y=max_happiness, fill=Country.name)) +
  geom_bar(stat="identity", width=0.5) +
  coord_cartesian(ylim=c(7,8)) +
  labs(y= "Happiness", x = "Country Name") +
  ggtitle("Graph of Happiness vs Country Name (Highest 6)")

ggplot(Top_6_lowest, aes(x=reorder(Country.name, -min_happiness), y=min_happiness, fill=Country.name)) +
  geom_bar(stat="identity", width=0.5) +
  coord_cartesian(ylim=c(2,3)) +
  labs(y= "Happiness", x = "Country Name") +
  ggtitle("Graph of Happiness vs Country Name (Lowest 6)")

#-----------------------------Q1 REGRESSION PROBLEM-----------------------------
# Drop Country.name, year and rank from dataframe
happiness_regression <- happiness_raw %>% select(-Country.name,-year,-Rank)

# Create Correlation Matrix
corr_mat <- round(cor(happiness_regression, method = "pearson"),3)
corr_mat

# Melt Correlation Matrix
melted_corr_mat <- melt(corr_mat)
head(melted_corr_mat)

#Plot Correlation Heat Map
ggplot(melted_corr_mat, aes(Var1, Var2, fill=value)) + geom_tile() + 
  ggtitle("Correlation Matrix") + xlab("Features") + ylab("Features") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  geom_text(aes(label = value)) +
  scale_fill_gradient(low = "azure1", high = "cornflowerblue")

# Based on Correlation matrix, Log.GDP.per.capita, Social.support, Healthy.life.expectancy.at.birth, Freedom.to.make.life.choices have strong correlation with Life.Ladder
# Combine and plot the scatter plot graph to study the trend of Life.Ladder against the aforementioned features
sp1 <- ggplot(happiness_regression, aes(Log.GDP.per.capita, Life.Ladder)) + 
  geom_point(alpha=0.1) + geom_smooth(method="lm", colour = "blue") + ggtitle("Scatter Plot against Log.GDP.per.capita")

sp2 <- ggplot(happiness_regression, aes(Social.support, Life.Ladder)) + 
  geom_point(alpha = 0.1) + geom_smooth(method="lm", colour = "red") + ggtitle("Scatter Plot against Social.support")

sp3 <- ggplot(happiness_regression, aes(Healthy.life.expectancy.at.birth, Life.Ladder)) + 
  geom_point(alpha=0.1) + geom_smooth(method="lm", colour = "green") + ggtitle("Scatter Plot against Healthy.life.expectancy.at.birth")

sp4 <- ggplot(happiness_regression, aes(Freedom.to.make.life.choices, Life.Ladder)) + 
  geom_point(alpha = 0.1) + geom_smooth(method="lm", colour = "orange") + ggtitle("Scatter Plot against Freedom.to.make.life.choices")

# Scatter Plot with best fit of LM against 4 features with strong correlations (>0.5)
sp_comb <- ggarrange(sp1, sp2, sp3, sp4,ncol = 2, nrow = 2)
print(sp_comb)

# As observed from the combined scatter plot, the higher the value of features, the higher the Life.Ladder


#-----------------------------Q2 CLASSIFICATION PROBLEM-----------------------------
# Count the number of rows for each Rank class
nrow(filter(happiness_raw, Rank == "Low"))
nrow(filter(happiness_raw, Rank == "Medium"))
nrow(filter(happiness_raw, Rank == "High"))
# The number of rows for Low, Medium and High are 54, 1449 & 209 respectively.

# Factor the Rank of Original Data for Plotting
happiness_class <- happiness_raw 
happiness_class$Rank <- factor(happiness_class$Rank,
                         levels = c("Low", "Medium", "High"))

# Plot Bar Chart to visualize the number of class
ggplot(happiness_class, aes(Rank, fill=Rank)) + geom_bar(colour='black') 
# From the bar chart, it was noted that the data is strongly imbalanced. 
# Both "Low" and "High" ranks are the minority class in this dataset.
# SMOTE will be done in later step.

# Plot Density Plot to see the distribution of a numeric variables
# Plot Box Plot to see the distributions of statistical data 

# Plot against Log.GDP.per.capita
dp1 <- ggplot(happiness_class, aes(x = Log.GDP.per.capita, fill=Rank)) +
  geom_density(alpha=0.25) + facet_wrap(~Rank) + ggtitle("Density Plot against Log.GDP.per.capita")
bp1 <- ggplot(happiness_class, aes(x = Rank, y= Log.GDP.per.capita , fill=Rank)) +
  geom_boxplot() + ggtitle("Box Plot against Log.GDP.per.capita")
dpbp1 <- ggarrange(dp1, bp1, ncol = 1, nrow = 2)
print(dpbp1)

# Plot against Social.support
dp2 <- ggplot(happiness_class, aes(x = Social.support, fill=Rank)) +
  geom_density(alpha=0.25) + facet_wrap(~Rank) + ggtitle("Density Plot against Social.support")
bp2 <- ggplot(happiness_class, aes(x = Rank, y= Social.support , fill=Rank)) +
  geom_boxplot() + ggtitle("Box Plot against Social.support")
dpbp2 <- ggarrange(dp2, bp2, ncol = 1, nrow = 2)
print(dpbp2)

# Plot against Healthy.life.expectancy.at.birth
dp3 <- ggplot(happiness_class, aes(x = Healthy.life.expectancy.at.birth, fill=Rank)) +
  geom_density(alpha=0.25) + facet_wrap(~Rank) + ggtitle("Density Plot against Healthy.life.expectancy.at.birth")
bp3 <- ggplot(happiness_class, aes(x = Rank, y= Healthy.life.expectancy.at.birth , fill=Rank)) +
  geom_boxplot() + ggtitle("Box Plot against Healthy.life.expectancy.at.birth")
dpbp3 <- ggarrange(dp3, bp3, ncol = 1, nrow = 2)
print(dpbp3)

# Plot against Freedom.to.make.life.choices
dp4 <- ggplot(happiness_class, aes(x = Freedom.to.make.life.choices, fill=Rank)) +
  geom_density(alpha=0.25) + facet_wrap(~Rank) + ggtitle("Density Plot against Freedom.to.make.life.choices")
bp4 <- ggplot(happiness_class, aes(x = Rank, y= Freedom.to.make.life.choices , fill=Rank)) +
  geom_boxplot() + ggtitle("Box Plot against Freedom.to.make.life.choices")
dpbp4 <- ggarrange(dp4, bp4, ncol = 1, nrow = 2)
print(dpbp4)

# Plot against Generosity
dp5 <- ggplot(happiness_class, aes(x = Generosity, fill=Rank)) +
  geom_density(alpha=0.25) + facet_wrap(~Rank) + ggtitle("Density Plot against Generosity")
bp5 <- ggplot(happiness_class, aes(x = Rank, y= Generosity , fill=Rank)) +
  geom_boxplot() + ggtitle("Box Plot against Generosity")
dpbp5 <- ggarrange(dp5, bp5, ncol = 1, nrow = 2)
print(dpbp5)

# Plot against Perceptions.of.corruption
dp6 <- ggplot(happiness_class, aes(x = Perceptions.of.corruption, fill=Rank)) +
  geom_density(alpha=0.25) + facet_wrap(~Rank) + ggtitle("Density Plot against Perceptions.of.corruption")
bp6 <- ggplot(happiness_class, aes(x = Rank, y= Perceptions.of.corruption , fill=Rank)) +
  geom_boxplot() + ggtitle("Box Plot against Perceptions.of.corruption")
dpbp6 <- ggarrange(dp6, bp6, ncol = 1, nrow = 2)
print(dpbp6)

# CATEGORIES ENCODER
# Factor the Rank of Original Data
happiness_Factored <- happiness_raw
happiness_Factored$Rank <- factor(happiness_Factored$Rank,
                               levels = c("Low", "Medium", "High"))
table(happiness_Factored$Rank)
prop.table(table(happiness_Factored$Rank))

#define original categorical labels
labs = LabelEncoder.fit(happiness_Factored$Rank)

#convert labels to numeric values
happiness_Factored$Rank = transform(labs, factor(happiness_Factored$Rank, levels = c("Low", "Medium", "High")))

#“Low” has become 1, “Medium” has become 2, “High” has become 3.

# MULTIVARIATE ANOVA TEST
# The null hypothesis (H0) of the ANOVA is no difference in means.
# The alternative hypothesis (Ha) is that the means are different from one another.
# Confidence Interval = 95% (Default)
# The p value of the variable is low (p < 0.025), then the variables are statistically significant, and vice cersa
# reference: https://www.scribbr.com/statistics/anova-in-r/#:~:text=Revised%20on%20November%2017%2C%202022,level%20of%20the%20independent%20variable.

aov1 = aov(data= happiness_Factored, Rank  ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + 
             Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption)
summary(aov1)

# The result of ANOVA test shows that all variables are statistically significant except Healthy.life.expectancy.at.birth

#SMOTE for IMBALANCED DATA
#Reference https://s3.amazonaws.com/assets.datacamp.com/production/course_8916/slides/chapter3.pdf
#Reference https://rpubs.com/ZardoZ/SMOTE_FRAUD_DETECTION
#Low and High are the minority class in this dataset

#SMOTE for "Low" rank
low_smoted <- SMOTE(happiness_Factored[,-1], happiness_Factored$Rank, K = 5, dup_size = 25)
low_smoted.data <- low_smoted$data
table(low_smoted.data$Rank)
prop.table(table(low_smoted.data$Rank))

#SMOTE for "High" rank
high_smoted <- SMOTE(low_smoted.data[,1:9], low_smoted.data$Rank, K = 5, dup_size = 6)
high_smoted.data <- high_smoted$data
table(high_smoted.data$Rank)
prop.table(table(high_smoted.data$Rank))

# The data has been balanced with SMOTE method.

#Dataframe with Balanced Data for Machine Learning Modelling
happiness_Balanced <- high_smoted.data

