graphics.off();rm(list=ls())#clear plots and environment
library(Hmisc)
library(tidyverse)
d=read_csv("aziz/forest.csv")
head(d)
text <- bind_cols(t1=c(as.character(d$Variable)) )
text
graphics.off()
quartz(width=7,height=6)
library(forestplot)

forestplot(text, 
           mean=d$HR,
           lower=d$Lower,
           upper=d$Upper,
           title="Hazard Ratio",
           xlab="Hazard Ratio (HR)",
           col=fpColors(box="black", lines="black", zero = "gray50"),
           zero=1,
           boxsize=0.3,
           lineheight = unit(4,"mm"),
           lwd.ci=1.3, ci.vertices=TRUE, ci.vertices.height = 0.2,
           txt_gp=fpTxtGp(label=gpar(cex=.8),
                          ticks=gpar(cex=1),
                          xlab=gpar(cex = 1),
                          title=gpar(cex = 1))
)  #end forest plot
