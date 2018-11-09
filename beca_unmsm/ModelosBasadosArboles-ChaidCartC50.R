rm(list=ls())
##########################################################################
################## -- San Marcos Data Science Community -- ###############
##########################################################################
######## Autores: Jose Cardenas - Andre Chavez  ########################## 
##########################################################################

# ######### 1) LIBRERIAS A UTILIZAR ################# 

library(ggvis)
library(party)
library(pROC)
library(e1071)
library(caret)
library(ROCR)
library(C50)
library(mlr)
library(lattice)
library(gmodels)
library(gplots)
library(rminer)
library(polycor)
library(class)
library(rpart)

######### 2) EXTRAYENDO LA DATA ################# 

train<-read.csv("train.csv",na.strings = c(""," ",NA)) # leer la data de entrenamiento

names(train) # visualizar los nombres de la data
head(train)  # visualizar los 6 primeros registros
str(train)   # ver la estructura de la data

######### 3) EXPLORACION DE LA DATA ################# 

# tablas resumen
summary(train) # tabla comun de obtener
summarizeColumns(train) # tabla mas completa

resumen=data.frame(summarizeColumns(train))


######### 4) IMPUTACION DE LA DATA ################# 

# revisar valores perdidos

perdidos=data.frame(resumen$name,resumen$na,resumen$type); colnames(perdidos)=c("name","na","type")
perdidos

# recodificando Dependents
train$Dependents=ifelse(train$Dependents=="3+",3,
                        ifelse(train$Dependents=="0",0,
                               ifelse(train$Dependents=="1",1,
                                      ifelse(train$Dependents=="2",2,
                                             train$Dependents))))
train$Dependents=as.factor(train$Dependents)

# convirtiendo en factor Credit_History
train$Credit_History <- as.factor(train$Credit_History)

# recodificando Loan_Status
train$Loan_Status=ifelse(train$Loan_Status=="N",0,1)
train$Loan_Status=as.factor(train$Loan_Status)

# partcionando la data en numericos y factores

numericos <- sapply(train, is.numeric) # variables cuantitativas
factores <- sapply(train, is.factor)  # variables cualitativas

train_numericos <-  train[ , numericos]
train_factores <- train[ , factores]

# APLICAR LA FUNCION LAPPLY PARA DISTINTAS COLUMNAS CONVERTIR A FORMATO NUMERICO
n1=min(dim(train_factores))
train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.numeric)
train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.factor)

# Para train 

train=cbind(train_numericos,train_factores[,-1])

## Imputacion Parametrica

#Podemos imputar los valores perdidos por la media o la moda

# data train
train_parametrica <- impute(train, classes = list(factor = imputeMode(), 
                                                  integer = imputeMode(),
                                                  numeric = imputeMean()),
                            dummy.classes = c("integer","factor"), dummy.type = "numeric")
train_parametrica=train_parametrica$data[,1:min(dim(train))]

summary(train_parametrica)


######### 5) PARTICION MUESTRAL ######################## 

set.seed(123)
training.samples <- train_parametrica$Loan_Status %>% 
        createDataPartition(p = 0.8, list = FALSE)
train.data  <- train_parametrica[training.samples, ]
test.data <- train_parametrica[-training.samples, ]

######### 6) MODELADO DE LA INFORMACION ################# 

################## Arbol CHAID ##########################


arbolchaid<-ctree(Loan_Status~.,data = train.data, 
               controls=ctree_control(mincriterion=0.95))

##probabilidades
proba4=sapply(predict(arbolchaid, newdata=test.data,type="prob"),'[[',2)

# curva ROC	
AUC4 <- roc(test.data$Loan_Status, proba4) 
auc_modelo4=AUC4$auc

# Gini
gini4 <- 2*(AUC4$auc) -1

# Calcular los valores predichos
PRED <-predict(arbolchaid, newdata=test.data,type="response")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,test.data$Loan_Status,positive = "1")

# sensibilidad
Sensitivity4=as.numeric(tabla$byClass[1])

# Precision
Accuracy4=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error4=mean(PRED!=test.data$Loan_Status)

# indicadores
auc_modelo4
gini4
Accuracy4
error4
Sensitivity4

################## Arbol CART ##########################

arbolcart <- rpart(Loan_Status~.,data = train.data,method="class",cp=0, minbucket=0)
xerr <- arbolcart$cptable[,"xerror"] ## error de la validacion cruzada
minxerr <- which.min(xerr)
mincp <- arbolcart$cptable[minxerr, "CP"]

arbolcart.poda <- prune(arbolcart,cp=mincp)

##probabilidades
proba5=predict(arbolcart.poda, newdata=test.data,type="prob")[,2]

# curva ROC
AUC5 <- roc(test.data$Loan_Status, proba5) 
auc_modelo5=AUC5$auc

# Gini
gini5 <- 2*(AUC5$auc) -1

# Calcular los valores predichos
PRED <-predict(arbolcart.poda, newdata=test.data,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,test.data$Loan_Status,positive = "1")

# sensibilidad
Sensitivity5=as.numeric(tabla$byClass[1])

# Precision
Accuracy5=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error5=mean(PRED!=test.data$Loan_Status)

# indicadores
auc_modelo5
gini5
Accuracy5
error5
Sensitivity5


################## Arbol C50 ############################


arbolc50 <- C5.0(Loan_Status~.,data = train.data,trials = 55,rules= TRUE,tree=FALSE,winnow=FALSE)

##probabilidades
proba6=predict(arbolc50, newdata=test.data,type="prob")[,2]

# curva ROC
AUC6 <- roc(test.data$Loan_Status, proba6) 
auc_modelo6=AUC6$auc

# Gini
gini6 <- 2*(AUC6$auc) -1

# Calcular los valores predichos
PRED <-predict(arbolc50, newdata=test.data,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,test.data$Loan_Status,positive = "1")

# sensibilidad
Sensitivity6=as.numeric(tabla$byClass[1])

# Precision
Accuracy6=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error6=mean(PRED!=test.data$Loan_Status)

# indicadores
auc_modelo6
gini6
Accuracy6
error6
Sensitivity6


## --Tabla De Resultados ####

AUC=rbind(auc_modelo4,
          auc_modelo5,
          auc_modelo6)

GINI=rbind(gini4,
           gini5,
           gini6)

Accuracy=rbind(Accuracy4,
            Accuracy5,
            Accuracy6)

ERROR= rbind(error4,
             error5,
             error6)

SENSIBILIDAD=rbind(Sensitivity4,
                   Sensitivity5,
                   Sensitivity6)

resultado=data.frame(AUC,GINI,Accuracy,ERROR,SENSIBILIDAD)
rownames(resultado)=c('Arbol_CHAID',
                      'Arbol_CART ',
                      'Arbol_c50')

resultado=round(resultado,2)
resultado

######### 7) COMPARACION DE RESULTADOS ################## 

# Ordenamos por el indicador que deseamos.
Resultado_ordenado <- resultado[order(-Accuracy),] 
Resultado_ordenado
