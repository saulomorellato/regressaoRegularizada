## PACOTES ##

library(MASS)       # stepwise (aic)
library(glmulti)    # all regression
library(car)        # multicolinearidade
library(tidyverse)  # manipulacao de dados
library(tidymodels) # modelos de machine learning
library(glmtoolbox) # testes para regressao logistica
library(ecostats)   # grafico de envelope do modelo
library(gtsummary)  # visualizacao dos resultados da regressao



##### CARREGANDO OS DADOS #####

df<- readxl::read_xlsx("dadosVinicius.xlsx", header=TRUE) %>% data.frame()




##### MANIPULANDO OS DADOS #####

df %>% glimpse()  # breve visualizacao

df$Tipo<- df$Tipo %>% factor(labels=c("benigno","maligno"))
df$Sexo<- df$Sexo %>% factor(labels=c("Masculino","Feminino"))

df<- df %>%
  mutate(HL=ifelse(HL<3, "Baixa", "Alta")) %>% 
  mutate(FF=ifelse(FF<3, "Baixa", "Alta"))

df$HL<- df$HL %>% factor(levels=c("Baixa","Alta"))
df$FF<- df$FF %>% factor(levels=c("Baixa","Alta"))

df %>% glimpse()  # breve visualizacao




##### VISUALIZANDO OS DADOS #####

# Tipo x Sexo - versao 1

df %>% ggplot(aes(x=Sexo, fill=Tipo)) +
  geom_bar() +
  ylab("quantidade") +
  theme_bw()



# Tipo x Sexo - versao 2

df %>% ggplot(aes(x=Sexo, fill=Tipo)) +
  geom_bar(position="dodge") +
  ylab("quantidade") +
  theme_bw()



# Tipo x Sexo - versao 3

df %>% ggplot(aes(x=Sexo, fill=Tipo)) +
  geom_bar(position="fill") +
  ylab("proporção") +
  theme_bw()



# Tipo x HL

df %>% ggplot(aes(x=HL, fill=Tipo)) +
  geom_bar(position="fill") +
  ylab("proporção") +
  theme_bw()



### Tipo x FF

df %>% ggplot(aes(x=FF, fill=Tipo)) +
  geom_bar(position="fill") +
  ylab("proporção") +
  theme_bw()



### Tipo x Idade

df %>% ggplot(aes(y=Idade, x=Tipo, fill=Tipo)) +
  geom_boxplot() +
  theme_bw()




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
