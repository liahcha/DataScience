setwd("C:/DMdata")
buytest = read.table('buytest.txt',sep='\t',header=T)
head(buytest,10)

quiz.buytest <- subset(buytest, select=c('AGE','BUY18','MARRIED','OWNHOME'))
head(quiz.buytest,10)

LR = glm(RESPOND~AGE+BUY18+MARRIED+OWNHOME,family=binomial(),data=buytest)
summary(LR)
exp(coef(LR))

a <- exp(-1.235529-(0.038387*25)+(0.456774*3))/(1+exp(-1.235529-(0.038387*25)+(0.456774*3)))
b <- exp((LR$coefficients[[1]])+(LR$coefficients[[2]]*25)+(LR$coefficients[[3]]*3))/(1+exp((LR$coefficients[[1]])+(LR$coefficients[[2]]*25)+(LR$coefficients[[3]]*3)))

out4 <- exp(-0.03)/(1+(exp(-0.03)))
out5 <- exp(-0.1)/(1+(exp(-0.1)))

in6.1 <- (exp(-0.03)/(1+(exp(-0.03))) * 0.1) + (exp(-0.1)/(1+(exp(-0.1))) * 0.3)
in6.2 <- (out4 * 0.1) + (out5 * 0.3)

out6 <- exp(in6.2)/(1+(exp(in6.2)))