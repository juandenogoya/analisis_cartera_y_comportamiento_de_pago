# analisis_cartera_y_comportamiento_de_pago
19/10/2023: Realizaron cambios en las variables, se redujo la dimensionalidad, se opto por otro enfoque.
Se realizao el OHE sobre la variable dependiente (categorica), pero luego de esto se definio trabajar sobre 1 sola de las clases, haciendo binaria la variable.
Con este cambio se realizaron las transformaciones y el split, para luego definir el modelo SVM y realizar el Train, Validation, Test.
Por ultimo se realizaron las predicciones, se compararon resultados y aplicaron las metricas para los modelos con sus conclusiones.


26/10/2023:
Para esta entrega y teniendo en cuenta los resultados de las transformaciones anteriores y del desempeño del modelo, realice las siguientes acciones.
# Respecto a la obtención de los datos a analizar, cambie el enfoque de la mineria, mejorando los script de SQL, este cambio me permitio:
  * Información mas completa.
  * Menor cantidad de transformaciones.
  * No fue necesario realizar un concatenado de varios archivos, sino que me limite a trabajar con un solo CSV, que concentraba 1 año de registros.
  * No existian duplicados y la información estaba validada en el DWH, lo que hizo más sencillo el EDA.
# Respecto a las transformaciones, elimine algunas variables que no representaban información importante en el analisis.
# Elimine Variables que estan relacionadas, tales como Importe de Cuota y Cantidades de Cuotas Pagadas y Atrasadas.


02/11/2023:
En esta entrega se realizo El Cross Validation.
Tambien se avanzo en la utilizacion de las librerias de Grid Search y de Random Search, con estas conclusiones:
# Respecto a Grid Search: No se pudo ejecutar, el entrenamiento del modelo Random Forest, estuvo ejecutandose por casi 120 minutos sin resultados.
# Respecto a Random Search:
 * Se obtuvieron resultados superiores al Cross Validation Score, en un 2%.
 * El rendimiento fue superior al Grid, en el sentido que el tiempo demandado fue menor a 10 minutos.
 * Versus el Modelo Random Forest Tradiciona, con hiperparametros manuales, tuvo un mejor resultado, en 2%.


 10/12/2023:
 En esta ultima actualizacion:
 * Se aplicaron sobre el DS, tecnicas de Agrupación para poder identificar insight desde diferentes analisis.
 * Se realizaron graficos utilizando las diferentes librerias de Matplotlib y Seaborn.
 * Aplicaron tecnias de Split, para luego comenzar con las transformaciones.
 * Se trebajo con el Modelo de Random Forest.
 * Se realizo Cross Validation sobre el Random Forest.
 * Se aplicaron tecnicas de Random Search respecto de los Hiperparametros.
 * Se aplicaron tecnicas de Bayesian Search sobre hiperparametros, para esto se aplicaron codificación de Label Encoder sobre las Variable Dependiente que es categorica.

