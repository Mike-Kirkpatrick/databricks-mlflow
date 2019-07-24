# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Tracking Experiment Runs

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook illustrates how MLflow can be used to track experiments for a binary classification problem.
# MAGIC This notebook has the following sections:
# MAGIC - Requirements
# MAGIC - The Dataset
# MAGIC - Create PySpark ML Pipelines
# MAGIC - Define Experiment Parameters
# MAGIC - Run The Experiments
# MAGIC - View Experiment Results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC 
# MAGIC All you need to take advantage of MLflow is a Databricks cluster with `Databricks Runtime 5.5 ML` or greater.
# MAGIC These runtimes come with MLflow installed and track MLlib enabled.
# MAGIC This notebook uses python but other languages are available.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC 
# MAGIC The Adult dataset we are going to use is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult).
# MAGIC These data are derived from census data, and consists of information about 48842 individuals and their annual income.
# MAGIC We will use this information to predict if an individual earns **<=50K or >50k** a year.

# COMMAND ----------

# MAGIC %md
# MAGIC We will read in the Adult dataset from databricks-datasets.
# MAGIC We'll read in the data in SQL using the CSV data source for Spark and rename the columns appropriately.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS adult;
# MAGIC 
# MAGIC CREATE TABLE adult (
# MAGIC   age DOUBLE,
# MAGIC   workclass STRING,
# MAGIC   fnlwgt DOUBLE,
# MAGIC   education STRING,
# MAGIC   education_num DOUBLE,
# MAGIC   marital_status STRING,
# MAGIC   occupation STRING,
# MAGIC   relationship STRING,
# MAGIC   race STRING,
# MAGIC   sex STRING,
# MAGIC   capital_gain DOUBLE,
# MAGIC   capital_loss DOUBLE,
# MAGIC   hours_per_week DOUBLE,
# MAGIC   native_country STRING,
# MAGIC   income STRING)
# MAGIC USING CSV
# MAGIC OPTIONS (path "/databricks-datasets/adult/adult.data", header "true");

# COMMAND ----------

dataset = spark.table("adult")

# We subset the dataset to include only features that we're interested in
label = ["income"]
numericColumns = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
categoricalColumns = ["workclass", "education", "relationship", "race", "sex"]
dataset = dataset.select(label + numericColumns + categoricalColumns)

display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC We split the dataset into training and testing (70% and 30%, respectively).
# MAGIC We set the seed for reproducibility.

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create PySpark ML Pipelines
# MAGIC 
# MAGIC PySpark Pipelines are used to encapsulate the steps necessary to build ML models.
# MAGIC Each step is stored in a "stage".
# MAGIC Once all the stages have been created, they are put into a [Pipeline].
# MAGIC The Pipeline is then executed and a ML model is the result.
# MAGIC The stages of our Pipeline are:
# MAGIC - Data Prep Stages
# MAGIC   - One-hot encode categorical features
# MAGIC   - Index our class label
# MAGIC   - Assemble our categorical and continuous features
# MAGIC - Create Model Pipelines
# MAGIC   - Logistic Regression
# MAGIC   - Decision Tree
# MAGIC   - Naive Bayes
# MAGIC   - Random Forest
# MAGIC 
# MAGIC [Pipeline]: http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Prep Stages

# COMMAND ----------

# MAGIC %md
# MAGIC Here we convert the categorical variables in the dataset into numeric variables via one-hot encoding.
# MAGIC We use a combination of [StringIndexer] and [OneHotEncoderEstimator] to convert the categorical variables.
# MAGIC The `OneHotEncoderEstimator` will return a [SparseVector].
# MAGIC 
# MAGIC [StringIndexer]: http://spark.apache.org/docs/latest/ml-features.html#stringindexer
# MAGIC [OneHotEncoderEstimator]: https://spark.apache.org/docs/latest/ml-features.html#onehotencoderestimator
# MAGIC [SparseVector]: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.linalg.SparseVector

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer

encodeCategoricalFeatures = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    encodeCategoricalFeatures += [stringIndexer, encoder]

# COMMAND ----------

# MAGIC %md
# MAGIC We convert our label into label indices using the `StringIndexer`.

# COMMAND ----------

label_stringIdx = StringIndexer(inputCol="income", outputCol="label")

# COMMAND ----------

# MAGIC %md
# MAGIC We use a `VectorAssembler` to combine all the feature columns into a single vector column.
# MAGIC This includes both the numeric columns and the one-hot encoded (previously categorical) binary vector columns in our dataset.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericColumns
assembleFeatures = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC For simplicity, w put all our data prep stages into one list.
# MAGIC This will make it simpler for us when we extened our data prep stages to our model stages.

# COMMAND ----------

dataPrepStages = encodeCategoricalFeatures + [label_stringIdx] + [assembleFeatures]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Model Pipelines
# MAGIC 
# MAGIC We are going to train 4 different algorithms:
# MAGIC - [Logistic Regression]
# MAGIC - [Decision Tree]
# MAGIC - [Naive Bayes]
# MAGIC - [Random Forest]
# MAGIC 
# MAGIC 
# MAGIC [Logistic Regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression
# MAGIC [Decision Tree]: https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier
# MAGIC [Naive Bayes]: https://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes
# MAGIC [Random Forest]: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier

# COMMAND ----------

# MAGIC %md
# MAGIC We first create a Logistic Regression pipeline.
# MAGIC To do this, we instantiate a Logistic Regression model.
# MAGIC We then append our model to our data prep stages, thus making the model the fourth and final stage.
# MAGIC Finally, we create the Logistic Regression pipeline from our stages.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="label", featuresCol="features")
lrStages= dataPrepStages + [lr]

lrPipeline = Pipeline(stages=lrStages)

# COMMAND ----------

# MAGIC %md
# MAGIC We do the same for a Decision Tree model.

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dtStages= dataPrepStages + [dt]

dtPipeline = Pipeline(stages=dtStages)

# COMMAND ----------

# MAGIC %md
# MAGIC We do the same for a Naive Bayes model.

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(labelCol="label", featuresCol="features")
nbStages= dataPrepStages + [nb]

nbPipeline = Pipeline(stages=nbStages)

# COMMAND ----------

# MAGIC %md
# MAGIC We do the same for a Random Forest model.

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features")
rfStages= dataPrepStages + [rf]

rfPipeline = Pipeline(stages=rfStages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Experiment Parameters

# COMMAND ----------

# MAGIC %md
# MAGIC Some parameters are going to be used across all algorithms.
# MAGIC We set the evaluator (default for `BinaryClassificationEvaluator` is AUC) and the number of folds for our cross-validation (k=10).
# MAGIC Again, these parameters are going to be applied to all the algorithms we train.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

k=3

# COMMAND ----------

# MAGIC %md
# MAGIC Conversely, some parameters are algorithm specific.
# MAGIC These are the hyper-parameters for each algorithm.
# MAGIC For each algorithm, we create a parameter grid and then create the cross-validation object.
# MAGIC The cross-validation object contains the algorithm pipeline, the parameter grid, the evaluator and the number of folds.
# MAGIC MLflow will track all of these parameters.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Logistic Regression
lrParamGrid = (ParamGridBuilder()
              .addGrid(lr.regParam, [0.01, 0.5, 2.0])
              .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
              .addGrid(lr.maxIter, [1, 5, 10])
              .build())

lrCv = CrossValidator(estimator=lrPipeline, estimatorParamMaps=lrParamGrid, evaluator=evaluator, numFolds=k)

# Decision Tree
dtParamGrid = (ParamGridBuilder()
              .addGrid(dt.maxDepth, [1, 2, 6, 10])
              .addGrid(dt.maxBins, [20, 40, 80])
              .build())

dtCv = CrossValidator(estimator=dtPipeline, estimatorParamMaps=dtParamGrid, evaluator=evaluator, numFolds=k)


# Naive Bayes
nbParamGrid = (ParamGridBuilder()
              .addGrid(nb.smoothing, [0, 0.5, 1, 2, 5, 10, 20])
              .build())

nbCv = CrossValidator(estimator=nbPipeline, estimatorParamMaps=nbParamGrid, evaluator=evaluator, numFolds=k)


# Random Forest
rfParamGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())
rfCv = CrossValidator(estimator=rfPipeline, estimatorParamMaps=rfParamGrid, evaluator=evaluator, numFolds=k)

# COMMAND ----------

# MAGIC %md
# MAGIC Additionally, we write a function to create ROC Curve plots that we will store as MLflow artifacts later.

# COMMAND ----------

def plot_roc(predictions, algorithm):
  import matplotlib.pyplot as plt
  from sklearn.metrics import roc_curve
  import numpy as np

  plotFile = "ROC Curve - " + algorithm + ".png"

  label = predictions.select("label").collect()
  preds = predictions.select("rawPrediction").collect()
  pred = []
  for p in preds:
    pred.append(p[0][1])
  pred = np.asfarray(pred)
  pred = np.reshape(pred, (-1, 1))
  label = np.asfarray(label)
  merge = np.concatenate((label, pred), axis=1)
  merge = merge[np.isfinite(merge).all(axis=1)] #remove nan's and inf's

  FPR, TPR, thresholds = roc_curve(merge[:,0], merge[:,1])

  fig, ax = plt.subplots()
  ax.scatter(FPR, TPR)
  plt.xlabel("FPR")
  plt.ylabel("TPR")
  plt.title("ROC Curve - " + algorithm)
  display(fig)
  image = fig

  fig.savefig(plotFile)
  plt.close(fig)
  
  return plotFile

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run The Experiments

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we train our models.
# MAGIC Cross validation will find the best model from the hyper-parameter combinations for each algorithm.
# MAGIC We make sure to apply our best model to the test data and log the corresponding metric (AUC) so that we can decide which algorithm, with which hyper-parameter combination, is best.
# MAGIC 
# MAGIC [MLflow] will track quite a bit for us.
# MAGIC Later, we can query MLflow based on all of these things
# MAGIC - I created an arbitrary "version" tag.
# MAGIC - When the model is fit , MLflow will track the results of each k-fold.
# MAGIC - I save the best model.
# MAGIC - I log some parameters for the best model so that, at a glance, I can see which hyperparameters resulted in the best performance on the test data.
# MAGIC - I also log the AUC for the best model so, at a glance, I can see the performance metric for the best model.
# MAGIC - Finally, I create a ROC plot and log it as an artifact in MLflow. Anything can be logged as an artifact in MLflow.
# MAGIC 
# MAGIC [MLflow]: https://docs.databricks.com/applications/mlflow/tracking.html

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic Regression Experiment

# COMMAND ----------

import mlflow
import mlflow.spark

# Logistic Regression
with mlflow.start_run():
  
  algorithm = "Logistic Regression"

  # Log some tags
  mlflow.set_tag("version","0.0.1")
  
  # Fit the model and make predictions
  lrCvModel = lrCv.fit(trainingData)
  lrPredictions = lrCvModel.transform(testData)
  
  # Save the best model
  lrModel = lrCvModel.bestModel
  mlflow.spark.save_model(lrModel, "model/" + algorithm)
  
  # Log some parameters
  mlflow.log_param("Algorithm", algorithm)
  mlflow.log_param("regParam", lrCvModel.bestModel.stages[-1]._java_obj.getRegParam())
  mlflow.log_param("elasticNetParam", lrCvModel.bestModel.stages[-1]._java_obj.getElasticNetParam())
  mlflow.log_param("maxIter", lrCvModel.bestModel.stages[-1]._java_obj.getMaxIter())
  
  # Log some metrics
  mlflow.log_metric("auc", evaluator.evaluate(lrPredictions))
  
  # Log ROC plot
  plotFile = plot_roc(predictions = lrPredictions, algorithm = algorithm)
  mlflow.log_artifact(plotFile)

# COMMAND ----------

# MAGIC %md
# MAGIC Decision Tree Experiment

# COMMAND ----------

# Decision Tree
with mlflow.start_run():
  
  algorithm = "Decision Tree"

  # Log some tags
  mlflow.set_tag("version","0.0.1")
  
  # Fit the model and make predictions
  dtCvModel = dtCv.fit(trainingData)
  dtPredictions = dtCvModel.transform(testData)
  
  # Save the best model
  dtModel = dtCvModel.bestModel
  mlflow.spark.save_model(dtModel, "model/" + algorithm)
  
  # Log some parameters
  mlflow.log_param("Algorithm", algorithm)
  mlflow.log_param("Max Depth", dtCvModel.bestModel.stages[-1]._java_obj.getMaxDepth())
  mlflow.log_param("Max Bins", dtCvModel.bestModel.stages[-1]._java_obj.getMaxBins())
  
  # Log some metrics
  mlflow.log_metric("auc", evaluator.evaluate(dtPredictions))
  
  # Log ROC plot
  plotFile = plot_roc(predictions = dtPredictions, algorithm = algorithm)
  mlflow.log_artifact(plotFile)

# COMMAND ----------

# MAGIC %md
# MAGIC Naive Bayes Experiment

# COMMAND ----------

# Naive Bayes
with mlflow.start_run():
  
  algorithm = "Naive Bayes"

  # Log some tags
  mlflow.set_tag("version","0.0.1")
  
  # Fit the model and make predictions
  nbCvModel = nbCv.fit(trainingData)
  nbPredictions = nbCvModel.transform(testData)
  
  # Save the best model
  nbModel = nbCvModel.bestModel
  mlflow.spark.save_model(nbModel, "model/" + algorithm)
  
  # Log some parameters
  mlflow.log_param("Algorithm", algorithm)
  mlflow.log_param("Smoothing", nbCvModel.bestModel.stages[-1]._java_obj.getSmoothing())
  
  # Log some metrics
  mlflow.log_metric("auc", evaluator.evaluate(nbPredictions))
  
  # Log ROC plot
  plotFile = plot_roc(predictions = nbPredictions, algorithm = algorithm)
  mlflow.log_artifact(plotFile)

# COMMAND ----------

# MAGIC %md
# MAGIC Random Forest Experiment

# COMMAND ----------

# Random Forest
with mlflow.start_run():
  
  algorithm = "Random Forest"

  # Log some tags
  mlflow.set_tag("version","0.0.1")
  
  # Fit the model and make predictions
  rfCvModel = rfCv.fit(trainingData)
  rfPredictions = rfCvModel.transform(testData)
  
  # Save the best model
  rfModel = rfCvModel.bestModel
  mlflow.spark.save_model(rfModel, "model/" + algorithm)
  
  # Log some parameters
  mlflow.log_param("Algorithm", algorithm)
  mlflow.log_param("Max Depth", rfCvModel.bestModel.stages[-1]._java_obj.getMaxDepth())
  mlflow.log_param("Max Bins", rfCvModel.bestModel.stages[-1]._java_obj.getMaxBins())
  mlflow.log_param("Num Trees", rfCvModel.bestModel.stages[-1]._java_obj.getNumTrees())
  
  # Log some metrics
  mlflow.log_metric("auc", evaluator.evaluate(rfPredictions))
  
  # Log ROC plot
  plotFile = plot_roc(predictions = rfPredictions, algorithm = algorithm)
  mlflow.log_artifact(plotFile)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Experiment Results

# COMMAND ----------

# MAGIC %md
# MAGIC Our experiments have finished, so now we click the `Runs` tab in the top-right corner of Databricks and view the experiment results.

# COMMAND ----------

#display(dbutils.fs.ls("dbfs:/databricks/mlflow/3981305023940101/c1059dbd33dd4c23b52a9aa885bc734e/artifacts"))
#display(dbutils.fs.ls("dbfs:/databricks/mlflow/3981305023940101"))
display(dbutils.fs.ls("dbfs:/databricks/mlflow/3981305023940101/model"))
#display(dbutils.fs.ls("*model*"))
#display(dbutils.fs.ls("dbfs:"))

# COMMAND ----------

import mlflow
import mlflow.spark

#mlflow.set_experiment_id(4186070636236695)
4186070636236695
4186070636236696
mlflow.set_experiment('/Users/mkirkpatrick@capaxglobal.com/MLflow Demo/Training Experiments')

# COMMAND ----------

import mlflow
import mlflow.spark


# Naive Bayes
with mlflow.start_run():
  
  algorithm = "Naive Bayes"

  # Log some tags
  #mlflow.set_tag("version","0.0.1")
  
  # Fit the model and make predictions
  nbCvModel = nbCv.fit(trainingData)
  nbPredictions = nbCvModel.transform(testData)
  
  # Save the best model
  nbModel = nbCvModel.bestModel
  mlflow.spark.save_model(nbModel, "model/" + algorithm)
  
  # Log some parameters
  #mlflow.log_param("Algorithm", algorithm)
  #mlflow.log_param("Smoothing", nbCvModel.bestModel.stages[-1]._java_obj.getSmoothing())
  
  # Log some metrics
  mlflow.log_metric("auc", evaluator.evaluate(nbPredictions))
  
  # Log ROC plot
  #plotFile = plot_roc(predictions = nbPredictions, algorithm = algorithm)
  #mlflow.log_artifact(plotFile)

# COMMAND ----------

model = mlflow.spark.load_model('/Users/mkirkpatrick@capaxglobal.com/MLflow Demo/Training Experiments/model/Naive Bayes/')

# COMMAND ----------

model = mlflow.spark.load_model('model/Naive Bayes/')

# COMMAND ----------

#display(dbutils.fs.ls("/tmp/mlflow/"))
#display(dbutils.fs.ls("/tmp/mlflow/015e6f32-75a0-4ef6-8f08-24ba3559513b"))
#display(dbutils.fs.ls("/tmp/mlflow/943ef997-6610-480d-b6ae-b4fd196813dc/"))
display(dbutils.fs.ls("/tmp/mlflow/d271a00b-fbd2-4c5e-8f0f-b0d5c718b73f/metadata/part-00000/"))

# COMMAND ----------

nbPredictions = nbCvModel.transform(testData)