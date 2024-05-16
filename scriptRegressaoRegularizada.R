## PACOTES ##

library(MASS)       # stepwise (aic)
library(glmulti)    # all regression
library(car)        # multicolinearidade
library(tidyverse)  # manipulacao de dados
library(tidymodels) # modelos de machine learning
library(plsmod)     # uso de pls no tidymodels
library(glmtoolbox) # testes para regressao logistica
library(ecostats)   # grafico de envelope do modelo
library(gtsummary)  # visualizacao dos resultados da regressao
library(janitor)    # limpeza de dados
library(tictoc)     # cronometrar tempo de execucao



##### CARREGANDO OS DADOS #####

df<- readxl::read_xlsx("dadosVinicius.xlsx") %>% data.frame()



##### MANIPULANDO OS DADOS #####

df<- df %>% clean_names()
names(df)<- gsub("_1", "", names(df))

#df %>% glimpse()  # breve visualizacao

#df<- na.omit(df)
df<- df %>% filter(!is.na(grave))
df<- df %>% dplyr::select(-id)
df$grave<- df$grave %>% factor()


##### SPLIT #####

set.seed(0)
split<- initial_split(df, strata=grave)

df.train<- training(split)
df.test<- testing(split)

#folds<- vfold_cv(df.train, v=10, strata=grave)
folds<- vfold_cv(df.train, v=2, repeats=5, strata=grave)




##### PRÉ-PROCESSAMENTO #####

receita<- recipe(grave ~ . , data = df.train) %>%
  step_zv(all_predictors()) %>% 
  step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  #step_normalize(all_numeric_predictors()) %>% 
  #step_impute_knn(all_predictors()) %>%
  step_naomit()




##### MODELOS #####

model_pls<- parsnip::pls(num_comp = tune()) %>%
  set_engine("mixOmics") %>%
  set_mode("classification")

model_las<- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

model_rid<- logistic_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

model_net<- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")





##### WORKFLOW #####

wf_pls<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_pls)

wf_las<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_las)

wf_rid<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_rid)

wf_net<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_net)




##### HIPERPARAMETERS TUNING - BAYESIAN SEARCH #####

tic()
tune_pls<- tune_bayes(wf_pls,
                      resamples = folds,
                      initial = 5,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      #metrics = metric_set(mn_log_loss),
                      param_info = parameters(num_comp(range(1, 20)))
)
toc()
# 151.08 sec elapsed


tic()
tune_las<- tune_bayes(wf_las,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      #metrics = metric_set(mn_log_loss),
                      param_info = parameters(penalty(range=c(-10,-2)))
)
toc()
# 59.35 sec elapsed


tic()
tune_rid<- tune_bayes(wf_rid,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      #metrics = metric_set(mn_log_loss),
                      param_info = parameters(penalty(range=c(-10,5)))
)
toc()
# 66.18 sec elapsed



tic()
tune_net<- tune_bayes(wf_net,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      #metrics = metric_set(mn_log_loss),
                      param_info = parameters(penalty(range=c(-10,5)),
                                              mixture(range=c(0,1)))
)
toc()
# 75.98 sec elapsed




## VISUALIZANDO OS MELHORES MODELOS (BEST RMSE)

show_best(tune_pls,n=3)
show_best(tune_las,n=3)
show_best(tune_rid,n=3)
show_best(tune_net,n=3)

#best_num_comp<- show_best(tune_pls,n=1)[1] %>% as.numeric()
#best_pen_las<- show_best(tune_las,n=1)[1] %>% as.numeric()
#best_pen_rid<- show_best(tune_rid,n=1)[1] %>% as.numeric()
#best_pen_net<- show_best(tune_net,n=1)[1] %>% as.numeric()

#best_num_comp
#best_pen_las
#best_pen_rid
#best_pen_net



## MODELOS TREINADOS APÓS TUNAR OS HIPERPARAMETROS

wf_pls_trained<- wf_pls %>% finalize_workflow(select_best(tune_pls)) %>% fit(df.train)
wf_las_trained<- wf_las %>% finalize_workflow(select_best(tune_las)) %>% fit(df.train)
wf_rid_trained<- wf_rid %>% finalize_workflow(select_best(tune_rid)) %>% fit(df.train)
wf_net_trained<- wf_net %>% finalize_workflow(select_best(tune_net)) %>% fit(df.train)




### VALIDATION ###

# PREDIZENDO DADOS TESTE

pred_pls<- predict(wf_pls_trained, df.test, type="prob")
pred_las<- predict(wf_las_trained, df.test, type="prob")
pred_rid<- predict(wf_rid_trained, df.test, type="prob")
pred_net<- predict(wf_net_trained, df.test, type="prob")



df.prob<- data.frame(df.test$grave,
                     pred_pls[,2],
                     pred_las[,2],
                     pred_rid[,2],
                     pred_net[,2])

colnames(df.prob)<- c("y",
                      "pls",
                      "las",
                      "rid",
                      "net")

head(df.prob)

cut<- summary(df.train$grave)[2]/nrow(df.train)

df.class<- df.prob %>% 
  mutate(pls=ifelse(pls<cut,0,1)) %>% 
  mutate(las=ifelse(las<cut,0,1)) %>% 
  mutate(rid=ifelse(rid<cut,0,1)) %>% 
  mutate(net=ifelse(net<cut,0,1)) %>%
  mutate(across(!y, as.factor))

df.class %>% head()    # VISUALIZANDO CLASSES





#####  VERIFICANDO MEDIDAS DE CLASSIFICAÇÃO  #####

medidas<- cbind(summary(conf_mat(df.class, y, pls))[,-2],
                summary(conf_mat(df.class, y, las))[,3],
                summary(conf_mat(df.class, y, rid))[,3],
                summary(conf_mat(df.class, y, net))[,3])                     

colnames(medidas)<- c("medida",
                      "pls",
                      "las",
                      "rid",
                      "net")

medidas








## COEFICIENTES DOS MODELOS TREINADOS

coef_pls_trained<- wf_pls_trained %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  filter(component==best_num_comp) %>% 
  dplyr::select(value)

coef_las_trained<- wf_las_trained %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  dplyr::select(estimate)

coef_rid_trained<- wf_rid_trained %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  dplyr::select(estimate)







##### VERIFICANDO MULTICOLINEARIDADE #####

# remover ou transformar a variavel com valor maior que 10

glm(Tipo ~ ., family=binomial, data=df) %>% vif()




### PREPARANDO MODELO DE REGRESSAO ###

modelo_base<- glm(Tipo ~ .^2, family=binomial, data=df)
modelo_base %>% summary()



### SELECAO DE VARIAVEIS - STEPWISE (AIC) ###

modelo_step<- stepAIC(modelo_base, direction="both")
modelo_step %>% summary()
modelo_step %>% tbl_regression(exponentiate = TRUE,
                               estimate_fun = function(x) style_ratio(x, digits = 4))




##### TODOS MODELOS DE REGRESSAO #####

modelo_all_reg<- glmulti(Tipo ~ .,
                         data = df,
                         crit = aic,         # aic, aicc, bic, bicc
                         level = 2,          # 1 sem interacoes, 2 com
                         method = "h",       # "d", ou "h", ou "g"
                         family = binomial,
                         fitfunction = glm,  # tipo de modelo (lm, glm, etc)
                         report = FALSE,
                         plotty = FALSE
)

modelo_all_reg<- modelo_all_reg@objects[[1]]
modelo_all_reg %>% summary()
modelo_all_reg %>% tbl_regression(exponentiate = TRUE,
                                  estimate_fun = function(x) style_ratio(x, digits = 4))





### TESTE HOSMER-LEMESHOW DE QUALIDADE DO AJUSTE ###

hltest(modelo_step,verbose=FALSE)$p.value

# é um teste qui-quadrado de aderência
# H0: não existe diferença entre observados e esperados
# H1: há diferença
# desejamos um valor MAIOR que 0.05




### GRÁFICOS DO MODELO ###

n<- nrow(df)    		                # número de observações
k<- length(modelo_step$coef)        # k=p+1 (número de coeficientes)

corte.hii<- 2*k/n		                # corte para elementos da diagonal de H
corte.cook<- qf(0.5,k,n-k)      	  # corte para Distância de Cook

hii<- hatvalues(modelo_step) 		    # valores da diagonal da matriz H
dcook<- cooks.distance(modelo_step)	# distância de Cook

obs<- 1:n

df.fit<- data.frame(obs,hii,dcook)


# GRÁFICO - ALAVANCAGEM

df.fit %>% ggplot(aes(x=obs,y=hii,ymin=0,ymax=hii)) + 
  geom_point() + 
  geom_linerange() + 
  geom_hline(yintercept = corte.hii, color="red", linetype="dashed") + 
  xlab("Observação") + 
  ylab("Alavancagem") + 
  theme_bw()



# GRÁFICO - DISTÂNCIA DE COOK

df.fit %>% ggplot(aes(x=obs,y=dcook,ymin=0,ymax=dcook)) + 
  geom_point() + 
  geom_linerange() +
  geom_hline(yintercept = corte.cook, color="red", linetype="dashed") + 
  xlab("Observação") + 
  ylab("Distância de Cook") + 
  theme_bw()



# ENVELOPE

env<- plotenvelope(modelo_step,
                   which=2,
                   n.sim=10000,
                   conf.level=0.95,
                   plot.it=FALSE) 

env[[2]]$p.value    # H0: modelo correto vs. H1: modelo incorreto (desejamos um valor MAIOR que 0.05)

df.env<- data.frame(obs, env[[2]]$x, env[[2]]$y, env[[2]]$lo, env[[2]]$hi)
colnames(df.env)<- c("obs", "x", "y", "lo", "hi")


df.env %>% ggplot(aes(x=x,y=y)) + 
  geom_point() + 
  geom_line(aes(x=x,y=lo), linetype="dashed") +
  geom_line(aes(x=x,y=hi), linetype="dashed") +
  xlab("Resíduos Simulados") + 
  ylab("Resíduos Observados") + 
  theme_bw()
