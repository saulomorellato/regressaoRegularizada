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
split<- initial_split(df, strata=grave, prop=0.8)

df.train<- training(split)
df.test<- testing(split)

#folds<- vfold_cv(df.train, v=10, strata=grave)
folds<- vfold_cv(df.train, v=2, repeats=5, strata=grave)




##### PRÉ-PROCESSAMENTO #####

receita<- recipe(grave ~ . , data = df.train) %>%
  #step_zv(all_predictors()) %>% 
  step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  #step_normalize(all_numeric_predictors()) %>% 
  #step_impute_knn(all_predictors()) %>%
  step_naomit()




##### MODELOS #####

model_pls<- parsnip::pls(num_comp = tune()) %>%
  set_engine("mixOmics") %>%
  set_mode("classification")

model_las<- logistic_reg(penalty = tune(),
                         mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

model_rid<- logistic_reg(penalty = tune(),
                         mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

model_net<- logistic_reg(penalty = tune(),
                         mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")


cores<- parallel::detectCores()

model_rfo<- rand_forest(mtry = tune(),
                        trees = 10000,
                        min_n = tune()) %>%
  set_engine("ranger", num.threads = cores, importance = "impurity") %>%
  set_mode("classification")

model_xgb<- boost_tree(mtry = tune(),
                       trees = 10000,
                       min_n = tune(),
                       loss_reduction = tune(),
                       learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


model_mlp <- mlp(hidden_units = tune(),
                 penalty = tune()) %>% 
  set_engine("brulee", importance = "permutation") %>% 
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

wf_rfo<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_rfo)

wf_xgb<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_xgb)

wf_mlp<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_mlp)




##### HIPERPARAMETERS TUNING - BAYESIAN SEARCH #####

tic()
tune_pls<- tune_bayes(wf_pls,
                      resamples = folds,
                      initial = 5,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(num_comp(range(1, 20)))
)
toc()
# 91.91 sec elapsed


tic()
tune_las<- tune_bayes(wf_las,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(penalty(range=c(-10,5)))
)
toc()
# 61.17 sec elapsed


tic()
tune_rid<- tune_bayes(wf_rid,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(penalty(range=c(-10,5)))
)
toc()
# 64.85 sec elapsed



tic()
tune_net<- tune_bayes(wf_net,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(penalty(range=c(-10,5)),
                                              mixture(range=c(0,1)))
)
toc()
# 72.12 sec elapsed


## RFO - RANDOM FOREST

tic()
tune_rfo<- tune_bayes(wf_rfo,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(mtry(range=c(1,1500)),
                                              #trees(range=c(50,10000)),
                                              min_n(range=c(1,50)))
)
toc()
# 586.63 sec elapsed (~ 10 min)


## XGB - XGBOOSTING

doParallel::registerDoParallel()

tic()
tune_xgb<- tune_bayes(wf_xgb,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(mtry(range=c(1,1500)),
                                              #trees(range=c(50,10000)),
                                              min_n(range=c(1,50)),
                                              loss_reduction(range=c(-10,5)),
                                              learn_rate(range=c(-10,0)))
)
toc()
# 1264.11 sec elapsed (~ 20 min)



##### HIPERPARAMETERS TUNING - GRID SEARCH #####

grid_hidden_units <- tribble(
  ~hidden_units,
  c(32, 32),
  c(32, 32, 32),
  c(64, 64),
  c(64, 64, 64),
  c(128, 128),
  c(128, 128, 128)
)
grid_penalty <- tibble(penalty = c(0.0001, 0.001, 0.01, 0.1))
grid_mlp <- grid_hidden_units %>% crossing(grid_penalty)
#grid_mlp

doParallel::registerDoParallel()

tic()
tune_mlp<- tune_grid(wf_mlp,
                     resamples = folds,
                     grid = grid_mlp,
                     control = control_grid(save_pred=TRUE,
                                            save_workflow=TRUE),
                     metrics = metric_set(roc_auc)
)
toc()
# 1056.56 sec elapsed (~ 18 min)



## VISUALIZANDO OS MELHORES MODELOS (BEST RMSE)

show_best(tune_pls,n=3)
show_best(tune_las,n=3)
show_best(tune_rid,n=3)
show_best(tune_net,n=3)
show_best(tune_rfo,n=3)
show_best(tune_xgb,n=3)
show_best(tune_mlp,n=3)




## MODELOS TREINADOS APOS TUNAR OS HIPERPARAMETROS

wf_pls_trained<- wf_pls %>% finalize_workflow(select_best(tune_pls)) %>% fit(df.train)
wf_las_trained<- wf_las %>% finalize_workflow(select_best(tune_las)) %>% fit(df.train)
wf_rid_trained<- wf_rid %>% finalize_workflow(select_best(tune_rid)) %>% fit(df.train)
wf_net_trained<- wf_net %>% finalize_workflow(select_best(tune_net)) %>% fit(df.train)
wf_rfo_trained<- wf_rfo %>% finalize_workflow(select_best(tune_rfo)) %>% fit(df.train)
wf_xgb_trained<- wf_xgb %>% finalize_workflow(select_best(tune_xgb)) %>% fit(df.train)
#wf_mlp_trained<- wf_mlp %>% finalize_workflow(select_best(tune_mlp)) %>% fit(df.train)
mlp_best<- tune_mlp %>% select_best(metric = "roc_auc")
mlp_best_list<- mlp_best %>% as.list()
mlp_best_list$hidden_units <- mlp_best_list$hidden_units %>% unlist()
wf_mlp_trained<- wf_mlp %>% finalize_workflow(mlp_best_list) %>% fit(df.train)




## SALVANDO OS MODELOS

saveRDS(wf_pls_trained,"wf_pls_trained.rds")
saveRDS(wf_las_trained,"wf_las_trained.rds")
saveRDS(wf_rid_trained,"wf_rid_trained.rds")
saveRDS(wf_net_trained,"wf_net_trained.rds")
saveRDS(wf_rfo_trained,"wf_rfo_trained.rds")
saveRDS(wf_xgb_trained,"wf_xgb_trained.rds")
saveRDS(wf_mlp_trained,"wf_mlp_trained.rds")



## CARREGANDO OS MODELOS SALVOS

# wf_pls_trained<- readRDS("wf_pls_trained.rds")
# wf_las_trained<- readRDS("wf_las_trained.rds")
# wf_rid_trained<- readRDS("wf_rid_trained.rds")
# wf_net_trained<- readRDS("wf_net_trained.rds")
# wf_rfo_trained<- readRDS("wf_rfo_trained.rds")
# wf_xgb_trained<- readRDS("wf_xgb_trained.rds")
# wf_mlp_trained<- readRDS("wf_mlp_trained.rds")



### VALIDATION ###

# PREDIZENDO DADOS TESTE

pred_pls<- predict(wf_pls_trained, df.test, type="prob")
pred_las<- predict(wf_las_trained, df.test, type="prob")
pred_rid<- predict(wf_rid_trained, df.test, type="prob")
pred_net<- predict(wf_net_trained, df.test, type="prob")
pred_rfo<- predict(wf_rfo_trained, df.test, type="prob")
pred_xgb<- predict(wf_xgb_trained, df.test, type="prob")
pred_mlp<- predict(wf_mlp_trained, df.test, type="prob")

df.prob<- data.frame(df.test$grave,
                     pred_pls[,2],
                     pred_las[,2],
                     pred_rid[,2],
                     pred_net[,2],
                     pred_rfo[,2],
                     pred_xgb[,2],
                     pred_mlp[,2])

colnames(df.prob)<- c("y",
                      "pls",
                      "las",
                      "rid",
                      "net",
                      "rfo",
                      "xgb",
                      "mlp")

head(df.prob)

cut<- summary(df.train$grave)[2]/nrow(df.train)

df.class<- df.prob %>% 
  mutate(pls=ifelse(pls<cut,0,1)) %>% 
  mutate(las=ifelse(las<cut,0,1)) %>% 
  mutate(rid=ifelse(rid<cut,0,1)) %>% 
  mutate(net=ifelse(net<cut,0,1)) %>%
  mutate(rfo=ifelse(rfo<cut,0,1)) %>% 
  mutate(xgb=ifelse(xgb<cut,0,1)) %>% 
  mutate(mlp=ifelse(mlp<cut,0,1)) %>%
  mutate(across(!y, as.factor))

df.class %>% head()    # VISUALIZANDO CLASSES





#####  VERIFICANDO MEDIDAS DE CLASSIFICAÇÃO  #####

medidas<- cbind(summary(conf_mat(df.class, y, pls))[,-2],
                summary(conf_mat(df.class, y, las))[,3],
                summary(conf_mat(df.class, y, rid))[,3],
                summary(conf_mat(df.class, y, net))[,3],
                summary(conf_mat(df.class, y, rfo))[,3],
                #summary(conf_mat(df.class, y, xgb))[,3],
                summary(conf_mat(df.class, y, mlp))[,3])                     

colnames(medidas)<- c("medida",
                      "pls",
                      "las",
                      "rid",
                      "net",
                      "rfo",
                      #"xgb",
                      "mlp")

medidas




## COEFICIENTES DOS MODELOS TREINADOS

coef_pls_trained<- wf_pls_trained %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  #filter(component==20) %>% 
  filter(component==as.numeric(show_best(tune_pls,n=1)[1])) %>% 
  filter(term != "Y") %>% 
  dplyr::select(term, value)

coef_las_trained<- wf_las_trained %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  filter(term != "(Intercept)") %>% 
  dplyr::select(term, estimate)

coef_rid_trained<- wf_rid_trained %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  filter(term != "(Intercept)") %>% 
  dplyr::select(term, estimate)

coef_net_trained<- wf_net_trained %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  filter(term != "(Intercept)") %>% 
  dplyr::select(term, estimate)

coef_rfo_trained<- wf_rfo_trained %>% 
  extract_fit_parsnip() %>% 
  #vip::vip(num_feature=trunc(sqrt(ncol(df.train)))) %>% 
  vip::vip(num_feature=2040)
coef_rfo_trained<- coef_rfo_trained$data

coef_xgb_trained<- wf_xgb_trained %>% 
  extract_fit_parsnip() %>% 
  #vip::vip(num_feature=trunc(sqrt(ncol(df.train)))) %>% 
  vip::vip(num_feature=2040)
coef_xgb_trained<- coef_xgb_trained$data

coef_mlp_trained<- wf_mlp_trained %>% 
  extract_fit_parsnip() %>% 
  #vip::vip(num_feature=trunc(sqrt(ncol(df.train)))) %>% 
  #vip::vip(num_feature=2040)
  baguette::nnet_imp_garson()
coef_mlp_trained<- coef_mlp_trained$data





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
