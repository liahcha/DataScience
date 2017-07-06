## QUIZ DAY1 ##
# HMEQ
hist(hmeq$MORTDUE)

hmeq$logVALUE=log(hmeq$MORTDUE)
hist(hmeq$logVALUE)

hmeq$sqrtALUE=sqrt(hmeq$MORTDUE)
hist(hmeq$sqrtVALUE)

plot(~VALUE+MORTDUE, data=hmeq, cex=0.1)

# CARSALE
carsale = read.table('C:/DMdata/carsale.csv', sep=',', header=T)
head(carsale,3)
colnames(carsale) <- c('SALES_CNT', 'AD_TIME', 'SALES_PPL_CNT', 'LOC')
LR_carsale<-lm(SALES_CNT ~ AD_TIME+SALES_PPL_CNT+LOC+LOC:AD_TIME+LOC:SALES_PPL_CNT, 
             data=carsale)
summary(LR_carsale)
final_LR <- step(LR_carsale)
summary(final_LR)