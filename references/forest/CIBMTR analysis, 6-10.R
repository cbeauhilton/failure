setwd("~/Desktop")

library(ggRandomForests)
library(ggplot2)
library(randomForestSRC)
library(survival)
library(survminer)
library(pec)
library(dplyr)


d <- read.csv("cibmtr_mut.csv")
v <- read.csv("valid.csv")

#drop Columns from the database
drop <- c("dfs", "intxrel", "rel", "trm", "dcrid", "dccn", "ncgp", "grpcod")
d = d[,!(names(d) %in% drop)]
v = v[,!(names(v) %in% drop)]

#select variables from the database
f <- select(d, "age", "mut_tp53", "wbcpr", "donorgp", "kps", "ipssrpr",
            "leuk2", "cytogener", "dnrage", "ancpr", "ancpr", "new3_condint", "blblpr",
            "yeartxgp", "mkstat", "hbpr", "platepr", "newblbmpr", "mut_cux1",
            "gvhdgp", "mut_ppm1d", "MutNum", "graftypecat")

#summary data
summary(d)




#plotting survival using ggsurvplot
SurvObj <- with(d, Surv(os, status == 1))
SurvObj
fit <- survfit(Surv(intxrel, rel) ~ d$MutNum_All_notp53, data = d)
fit

fit <- coxph(Surv(intxrel, rel) ~ d$MutNum_All_notp53, d)
summary(fit)

ggsurvplot(
    fit, 
    data = d, 
    size = 1,                 # change line size
    #palette = c("#E7B800", "#2E9FDF"),# custom color palettes
    #conf.int = TRUE,          # Add confidence interval
    pval = TRUE,              # Add p-value
    pval.size = 4,
    grid = NULL,
    break.x.by = 12,
    xlab = c("Time (Months)"),
    pval.coord = c(0.1, 0.1),
    surv.median.line = NULL,    #add median survival line
    risk.table = FALSE,        # Add risk table
    risk.table.col = FALSE,# Risk table color by groups
    legend.labs = c("MutNum = 0", "MutNum = 1", "MutNum = 2", "MutNum = 3", "MutNum = 4", "MutNum =/> 5"),    # Change legend labels
    risk.table.height = 0.25 # Useful to change when you have multiple groups
        
  )
#random survival forest
rfs <- rfsrc(Surv(intxsurv, dead) ~ ., data = d,na.action = "na.impute", importance = TRUE)

#Divide the data into training and validation cohorts
split=0.8
trainIndex <- createDataPartition(d, p=split, list=FALSE)
data_train <- d[ trainIndex,]
data_test <- d[-trainIndex,]

#testdata
prediction <- predict(rfs, newdata = v, na.action = "na.impute")


#plot the survival of each patinet
ggRFsrc <- plot(gg_rfsrc(rfs), alpha = 0.2) +
  theme(legend.position = "none") +
  labs(y = "Survival Probability", x = "Time (years)") +
  coord_cartesian(ylim = c(-0.01, 1.01))

show(ggRFsrc)


#ploting the prediction
ggRFsrc <- plot(gg_rfsrc(rfs), alpha = 0.2) +
  theme(legend.position = "none") +
  labs(y = "Survival Probability", x = "Time (years)") +
  coord_cartesian(ylim = c(-0.01, 1.01))

#VIMP function
vimp(rfs)
plot(gg_vimp(rfs)) + theme(legend.position = c(0.8, 0.2)) + labs(fill = "VIMP > 0")

#minimal depth
gg_md <- gg_minimal_depth(rfs, lbls = st.labs)
print(gg_md)
plot(gg_md)

#variable selection with VIMP and minimal depth
plot(gg_minimal_vimp(rfs))

#Variable depedence
gg_v <- gg_variable(rfs, time = c(2,3), time.labels = c("1 Year", "2 Years"))
xvar = "kps"

plot(gg_v, xvar)
  labs(y = "Survival", x = "rfs") +
  theme(legend.position = "none") +
  scale_color_discrete(name="Legend") +
  coord_cartesian(ylim = c(-0.01, 1.01))

#varaible dependence

  
#varaible interactions rsf
inter <- find.interaction(rfs)
inter <- find.interaction(rfs, method = "vimp", outcome.target = "mut_tp53")
plot(inter)

#variable interactions ggrandomforest
ggint <- gg_interaction(rfs)
xvar = c("mut_tp53") 
xvar = c("disstat") 
xvar = c("ipssrpr") 
xvar = c("new3_condint") 
xvar = c("hbpr") 
xvar = c("ancpr") 
xvar = c("age") 
xvar = c("wbcpr") 
xvar = c("cytogene") 
xvar = c("dnrage") 
xvar = c("blblpr") 
xvar = c("mut_rastk") 
xvar = c("leuk2") 
xvar = c("mut_tet2") 
xvar = c("newblbmpr") 
xvar = c("MutNum") 
xvar = c("mut_tet2")
xvar = c("mut_ppm1d") 
xvar = c("platepr") 
xvar = c("mut_cux1")
xvar = c("yeartxgp") 
plot(ggint, xvar)

#conditional dependence plots
age_cuts <-quantile_pts(ggvar$mut_tp53, groups = 6, intervals = TRUE)






aziz <- coxph(Surv(os, status) ~ cibmtr_score, d)

survConcordance(Surv(os, status) ~predict (aziz), d)
attach(d)
surv <- Surv(os, status)
sum.surv <- summary(coxph(surv ~ ipssrpr))
c_index <- sum.surv$concordance
print(c_index)

#naive Bayes
d <- select(d, "plts", "cytoipssr", "hb", "WHO", "bmbper", "wbc",
            "TP53", "age", "alc", "amc", "anc", 
            "PBB_per", "etiology", 'NPM1', "EZH2", "IDH2", "TET2")
library(naivebayes)
nb <- naive_bayes(TP53 ~ ., data = d)
d$TP53  <- as.factor(d$TP53)
ranf <- randomForest(TP53 ~ ., data = d, na.action = "na.omit")
pre <- predict(nb, d)

library(pROC)
roc_obj <- roc(nb, pre)
auc(roc_obj)

print(str(rfs$forest$xvar))
D=rfs$forest$xvar
I=which(sapply(D,is.factor))
for (i in I)  d[[i]]=factor(d[[i]],levels=levels(D[[i]]))
str(d)
aziz6 <- predictSurvProb(rfs, newdata = d, times = 6)

#Gradient boosted tree
library(gbm)
gbm <- gbm(day100 ~ ., data = d, distribution = "gaussian", shrinkage = 0.01)
test.error<-with(v,apply( (predmatrix-medv)^2,2,mean))
