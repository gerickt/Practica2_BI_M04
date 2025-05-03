# ————————————————
# Instrucciones de Ejecución:
# 1) Reiniciar la sesión de R o limpia el entorno (Session > Restart R en RStudio).
# 2) Cargar y ejecuta el script completo de principio a fin.
# 3) Limpiar cualquier objeto de la memoria.
# ————————————————
# Cargar librerías necesarias
library(dplyr)
library(caret)
library(randomForest)
library(pROC)
library(rpart)
library(rpart.plot)
library(tidyr)
library(ggplot2)

# Cargar datos y preprocesamiento
datos <- read.csv("~/MIADAS/m04/R_Examples/WA_Fn-UseC_-Telco-Customer-Churn.csv", stringsAsFactors = TRUE)

datos_clean <- datos %>%
  select(-customerID) %>%
  mutate(
    TotalCharges = as.numeric(as.character(TotalCharges))
  ) %>%
  na.omit() %>%
  # Asegurar que todas las variables categóricas sean factor
  mutate_if(is.character, as.factor)

# Asegurar Churn como factor "No"/"Yes"
datos_clean$Churn <- factor(datos_clean$Churn, levels = c("No", "Yes"))

# División de datos
set.seed(123)
indices <- createDataPartition(datos_clean$Churn, p = 0.7, list = FALSE)[,1]
train <- datos_clean[indices, ]
test  <- datos_clean[-indices, ]

# Función para umbral óptimo (Youden)
obtener_umbral <- function(roc_obj) {
  as.numeric(coords(roc_obj, "best", best.method = "youden", ret = "threshold"))
}

# 1) RANDOM FOREST
set.seed(123)
modelo_rf <- randomForest(Churn ~ ., data = train, ntree = 500, importance = TRUE)
prob_rf   <- predict(modelo_rf, test, type = "prob")[, "Yes"]
roc_rf    <- roc(response = test$Churn, predictor = prob_rf, levels = c("No","Yes"))
umbral_rf <- obtener_umbral(roc_rf)
pred_rf   <- factor(ifelse(prob_rf > umbral_rf, "Yes", "No"), levels = c("No","Yes"))
cm_rf     <- confusionMatrix(pred_rf, test$Churn, positive = "Yes")

# 2) REGRESIÓN LOGÍSTICA con validación cruzada
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
modelo_logit <- train(
  Churn ~ ., data = train,
  method = "glm", family = binomial(),
  metric = "ROC", trControl = ctrl
)
prob_logit   <- predict(modelo_logit, test, type = "prob")[, "Yes"]
roc_logit    <- roc(response = test$Churn, predictor = prob_logit, levels = c("No","Yes"))
umbral_logit <- obtener_umbral(roc_logit)
pred_logit   <- factor(ifelse(prob_logit > umbral_logit, "Yes", "No"), levels = c("No","Yes"))
cm_logit     <- confusionMatrix(pred_logit, test$Churn, positive = "Yes")

# 3) ÁRBOL DE DECISIÓN
modelo_tree <- rpart(
  Churn ~ ., data = train,
  method = "class",
  control = rpart.control(minsplit = 20, cp = 0.01)
)
prob_tree   <- predict(modelo_tree, test, type = "prob")[, "Yes"]
roc_tree    <- roc(response = test$Churn, predictor = prob_tree, levels = c("No","Yes"))
umbral_tree <- obtener_umbral(roc_tree)
pred_tree   <- factor(ifelse(prob_tree > umbral_tree, "Yes", "No"), levels = c("No","Yes"))
cm_tree     <- confusionMatrix(pred_tree, test$Churn, positive = "Yes")

# Comparación de métricas
tabla_resultados <- data.frame(
  Modelo    = c("Random Forest", "Regresión Logística", "Árbol de Decisión"),
  Accuracy  = c(cm_rf$overall["Accuracy"], cm_logit$overall["Accuracy"], cm_tree$overall["Accuracy"]),
  Precision = c(cm_rf$byClass["Pos Pred Value"], cm_logit$byClass["Pos Pred Value"], cm_tree$byClass["Pos Pred Value"]),
  Recall    = c(cm_rf$byClass["Sensitivity"], cm_logit$byClass["Sensitivity"], cm_tree$byClass["Sensitivity"]),
  F1        = c(cm_rf$byClass["F1"], cm_logit$byClass["F1"], cm_tree$byClass["F1"]),
  AUC       = c(auc(roc_rf), auc(roc_logit), auc(roc_tree))
)
print(tabla_resultados)

# Gráfico comparativo
tabla_resultados_long <- tabla_resultados %>%
  pivot_longer(-Modelo, names_to = "Métrica", values_to = "Valor")
ggplot(tabla_resultados_long, aes(x = Modelo, y = Valor, fill = Modelo)) +
  geom_col(position = position_dodge()) +
  geom_text(aes(label = round(Valor, 3)), vjust = -0.5, size = 3) +
  facet_wrap(~Métrica, scales = "free_y") +
  labs(title = "Comparación de Métricas", y = "Valor") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot de matrices de confusión
plot_conf_matrix <- function(cm, title) {
  df <- as.data.frame(cm$table)
  ggplot(df, aes(Prediction, Reference, fill = Freq)) +
    geom_tile() + geom_text(aes(label = Freq), color = "white", size = 6) +
    labs(title = title) + theme_minimal()
}
plot_conf_matrix(cm_rf, "Random Forest")
plot_conf_matrix(cm_logit, "Regresión Logística")
plot_conf_matrix(cm_tree, "Árbol de Decisión")

# Curvas ROC
ggroc(list(RF = roc_rf, Logit = roc_logit, Tree = roc_tree)) +
  labs(title = "Curvas ROC Comparativas")

# Importancia y visuales adicionales
varImpPlot(modelo_rf, main = "Importancia - RF")
rpart.plot(modelo_tree, type = 3, extra = 104, fallen.leaves = TRUE,
           main = "Árbol de Decisión")

# Coeficientes - Regresión Logística
coef_df <- as.data.frame(summary(modelo_logit$finalModel)$coefficients)
coef_df$Variable <- rownames(coef_df)

ggplot(coef_df[-1, ], aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_col() + coord_flip() +
  labs(title = "Coeficientes - Regresión Logística", x = "Variable", y = "Estimación") +
  theme_minimal()