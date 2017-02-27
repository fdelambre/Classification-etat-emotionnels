library(tidyverse)
library(hms)
library(ggplot2)
library(viridis)
library(GGally)

# Import des données
# utiliser read_delim pour choisir exactement ce qu'on veut
# read_csv : données US (sep col = ",", n sep decimal = ".")
# read_csv2 : données EU (sep col = ";", n sep decimal = ",")
data <- read_delim(
  "../data/AB_SurE_Entête.csv",
  delim = ";",
  col_types = cols(
    Respiration = col_double(),
    GSR = col_double(),
    date = col_time("%d/%m/%Y %H:%M:%OS"),
    Temperature = col_double(),
    CFM = col_double()),
  locale = locale(decimal_mark = ".")
)

View(data)

# Création de deux nouvelles colonnes pour les secondes et les minutes
data <- mutate(data, secondes = (date-date[1]))
data <- mutate(data, minutes = secondes/60)

View(data)

# Visualisation
ggplot(data) + geom_line(mapping = aes(x = minutes, y = Respiration))
ggplot(data) + geom_line(mapping = aes(x = minutes, y = GSR))
ggplot(data) + geom_line(mapping = aes(x = minutes, y = Temperature))
ggplot(data) + geom_line(mapping = aes(x = minutes, y = CFM))

# Statistiques de base
summary(data)

# Extraction des variables
var <-  select(data, Respiration, GSR, Temperature, CFM)
t <- select(data, secondes)

# Extraction (les indices commencent à 1)
# colonnes (variables) :
# - data$nom_col et data[["nom_col"]] par nom
# - data[[i]] par indice
# - data[c(debut:fin)] pour plusieurs colonnes

# lignes (observations) :
# - data[i,] une ligne
# - data[c(debut:fin),] plusieurs lignes

# class() permet d'avoir le type

acp <- princomp(var, cor = TRUE, scores = TRUE)
# Les vp sont les carrés des standards deviations calculées dans acp
val_propres <- acp$sdev^2
barplot(val_propres)

#View(var)
# Matrice de scatterplots
t_num <- as.numeric(t[[1]])
ggpairs(var[1:2000,])
# Un graphique simple avec transparence et coloration en fonction du temps
ggplot(var[1:300,], mapping = aes(x=Respiration[1:300], y=GSR[1:300], color = t_num[1:300])) + geom_point(alpha = 0.01)
