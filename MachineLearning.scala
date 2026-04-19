import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.regression.{RandomForestRegressor, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator

/*
 * ============================================================
 * IT462 - Big Data Systems
 * Phase 5 - Machine Learning with Spark
 * Group 2 - Section 72889
 *
 * Problem: Predict total bike demand at each start station
 *          per hour, based on temporal and station features.
 * Models:  Random Forest Regressor, Gradient Boosted Trees
 * ============================================================
 */

object MachineLearning {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Machine Learning - Demand Prediction")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // ============================================================
    // STEP 1: Load the Preprocessed Data
    // ============================================================
    println("\n[INFO] Loading preprocessed dataset...")
    val tripsDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/processed/transformedData.csv")

    println(s"[INFO] Total records loaded: ${tripsDF.count()}")
    tripsDF.printSchema()

    // ============================================================
    // STEP 2: Target Variable Preparation (Aggregation)
    // ============================================================
    // Fine-grained aggregation preserves end_station_name_idx
    // for computing avg_incoming_trips in Step 4.
    // Station-hour aggregation is the actual modeling granularity:
    // total trips per start station per hour per date.
    // 'date' is excluded from features — the model learns temporal
    // patterns from hour and day_of_week, not specific calendar dates.
    println("\n[INFO] Aggregating data (fine-grained) for avg_incoming_trips computation...")
    val fineGrainedDF = tripsDF.groupBy(
      col("date"),
      col("hour"),
      col("start_station_name_idx"),
      col("end_station_name_idx"),
      col("day_of_week"),
      col("is_weekend")
    ).agg(
      sum(expr("classic_trips + electric_trips")).alias("total_demand")
    )

    println("\n[INFO] Aggregating data at station-hour level for modeling...")
    val demandDF = tripsDF.groupBy(
      col("date"),
      col("hour"),
      col("start_station_name_idx"),
      col("day_of_week"),
      col("is_weekend")
    ).agg(
      sum(expr("classic_trips + electric_trips")).alias("total_demand")
    )

    println(s"[INFO] Records after station-hour aggregation: ${demandDF.count()}")

    println("\n[INFO] Target variable (total_demand) distribution:")
    demandDF.select("total_demand").summary("min", "25%", "50%", "75%", "max", "mean").show()

    // ============================================================
    // STEP 3: Train / Test Split (BEFORE any feature fitting)
    // ============================================================
    // Split first to prevent data leakage in all subsequent steps.
    println("\n[INFO] Splitting data into 70% train / 30% test (seed=42)...")
    val Array(trainRaw, testRaw) = demandDF.randomSplit(Array(0.7, 0.3), seed = 42)

    trainRaw.cache()
    testRaw.cache()

    println(s"[INFO] Training set size : ${trainRaw.count()}")
    println(s"[INFO] Test set size     : ${testRaw.count()}")

    // ============================================================
    // STEP 4: Feature Engineering — avg_incoming_trips
    // ============================================================
    // Captures how many bikes historically arrive at each station.
    // Computed from training data only to prevent leakage,
    // then joined to both train and test sets.
    println("\n[INFO] Computing avg_incoming_trips from training data only...")

    val trainStationDates = trainRaw.select("date", "start_station_name_idx")

    val trainFineGrained = fineGrainedDF.join(
      trainStationDates,
      fineGrainedDF("date") === trainStationDates("date") &&
      fineGrainedDF("start_station_name_idx") === trainStationDates("start_station_name_idx"),
      "inner"
    ).drop(trainStationDates("date"))
     .drop(trainStationDates("start_station_name_idx"))

    val avgIncoming = trainFineGrained
      .groupBy("end_station_name_idx")
      .agg(avg("total_demand").alias("avg_incoming_trips"))

    val trainWithIncoming = trainRaw
      .join(avgIncoming,
        trainRaw("start_station_name_idx") === avgIncoming("end_station_name_idx"),
        "left")
      .drop(avgIncoming("end_station_name_idx"))
      .na.fill(0.0, Seq("avg_incoming_trips"))

    val testWithIncoming = testRaw
      .join(avgIncoming,
        testRaw("start_station_name_idx") === avgIncoming("end_station_name_idx"),
        "left")
      .drop(avgIncoming("end_station_name_idx"))
      .na.fill(0.0, Seq("avg_incoming_trips"))

    // ============================================================
    // STEP 5: Categorical Encoding — OneHotEncoder on day_of_week
    // ============================================================
    // Prevents the model from treating day numbers as ordered values.
    // OHE not applied to start_station_name_idx (~2200 stations)
    // to avoid dimensional explosion — tree-based models handle
    // index encoding without ordinal bias.
    println("\n[INFO] Fitting OneHotEncoder on training data...")
    val encoder = new OneHotEncoder()
      .setInputCols(Array("day_of_week"))
      .setOutputCols(Array("day_of_week_ohe"))
      .setDropLast(false)

    val encoderModel = encoder.fit(trainWithIncoming)
    val trainEncoded  = encoderModel.transform(trainWithIncoming)
    val testEncoded   = encoderModel.transform(testWithIncoming)

    // ============================================================
    // STEP 6: Feature Integration — VectorAssembler
    // ============================================================
    // No scaling applied — tree-based models are invariant to feature scale.
    println("\n[INFO] Assembling feature vector...")
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "hour",
        "start_station_name_idx",
        "day_of_week_ohe",
        "is_weekend",
        "avg_incoming_trips"
      ))
      .setOutputCol("features")

    val trainFinal = assembler.transform(trainEncoded).select("features", "total_demand")
    val testFinal  = assembler.transform(testEncoded).select("features", "total_demand")

    println("\n[INFO] Sample of final training data (features + label):")
    trainFinal.show(10, truncate = false)

    // ============================================================
    // STEP 7: Baseline Model — Mean Prediction
    // ============================================================
    println("\n[INFO] Computing baseline (mean prediction)...")
    val trainMean = trainFinal.agg(avg("total_demand")).first().getDouble(0)
    println(f"[INFO] Training mean total_demand = $trainMean%.4f")

    val baselineDF = testFinal.withColumn("baseline_prediction", lit(trainMean))

    val baseEvalRMSE = new RegressionEvaluator().setLabelCol("total_demand").setPredictionCol("baseline_prediction").setMetricName("rmse")
    val baseEvalMAE  = new RegressionEvaluator().setLabelCol("total_demand").setPredictionCol("baseline_prediction").setMetricName("mae")
    val baseEvalR2   = new RegressionEvaluator().setLabelCol("total_demand").setPredictionCol("baseline_prediction").setMetricName("r2")

    val baselineRMSE = baseEvalRMSE.evaluate(baselineDF)
    val baselineMAE  = baseEvalMAE.evaluate(baselineDF)
    val baselineR2   = baseEvalR2.evaluate(baselineDF)

    println("\n============================================================")
    println("Baseline Model Results (Always predict training mean)")
    println("============================================================")
    println(f"  RMSE : $baselineRMSE%.4f")
    println(f"  MAE  : $baselineMAE%.4f")
    println(f"  R2   : $baselineR2%.4f")

    // Evaluators — defined once, reused for all models
    val evalRMSE = new RegressionEvaluator().setLabelCol("total_demand").setPredictionCol("prediction").setMetricName("rmse")
    val evalMAE  = new RegressionEvaluator().setLabelCol("total_demand").setPredictionCol("prediction").setMetricName("mae")
    val evalR2   = new RegressionEvaluator().setLabelCol("total_demand").setPredictionCol("prediction").setMetricName("r2")

    // ============================================================
    // STEP 8: Random Forest Regressor — Training
    // ============================================================
    // numTrees=100, maxDepth=15 for stronger pattern learning.
    println("\n[INFO] Training Random Forest Regressor...")
    val rf = new RandomForestRegressor()
      .setLabelCol("total_demand")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(15)
      .setSeed(42)

    val rfModel = rf.fit(trainFinal)
    println("[INFO] Random Forest training complete.")

    // Training metrics — RF
    val rfTrainPredictions = rfModel.transform(trainFinal)
    val rfTrainRMSE = evalRMSE.evaluate(rfTrainPredictions)
    val rfTrainMAE  = evalMAE.evaluate(rfTrainPredictions)
    val rfTrainR2   = evalR2.evaluate(rfTrainPredictions)

    println("\n============================================================")
    println("Random Forest — Training Set Results")
    println("============================================================")
    println(f"  RMSE : $rfTrainRMSE%.4f")
    println(f"  MAE  : $rfTrainMAE%.4f")
    println(f"  R2   : $rfTrainR2%.4f")

    // ============================================================
    // STEP 9: Evaluation — Random Forest
    // ============================================================
    val rfPredictions = rfModel.transform(testFinal)

    val rfRMSE = evalRMSE.evaluate(rfPredictions)
    val rfMAE  = evalMAE.evaluate(rfPredictions)
    val rfR2   = evalR2.evaluate(rfPredictions)

    println("\n============================================================")
    println("Random Forest Regressor — Test Set Results")
    println("============================================================")
    println(f"  RMSE : $rfRMSE%.4f")
    println(f"  MAE  : $rfMAE%.4f")
    println(f"  R2   : $rfR2%.4f")

    // ============================================================
    // STEP 10: Feature Importance — Random Forest
    // ============================================================
    // day_of_week_ohe produces 7 binary columns (one per day).
    println("\n============================================================")
    println("Feature Importances — Random Forest")
    println("============================================================")

    val featureNames = Array(
      "hour",
      "start_station_name_idx",
      "day_of_week_Sun", "day_of_week_Mon", "day_of_week_Tue",
      "day_of_week_Wed", "day_of_week_Thu", "day_of_week_Fri", "day_of_week_Sat",
      "is_weekend",
      "avg_incoming_trips"
    )

    rfModel.featureImportances.toArray
      .zip(featureNames)
      .sortBy(-_._1)
      .foreach { case (importance, name) =>
        println(f"  $name%-30s -> $importance%.4f")
      }

    // ============================================================
    // STEP 11: Gradient Boosted Trees Regressor — Training
    // ============================================================
    // GBT builds trees sequentially, each correcting the previous one's errors.
    // maxIter=50 boosting rounds, maxDepth=10 kept shallow to avoid overfitting.
    println("\n[INFO] Training Gradient Boosted Trees Regressor...")
    val gbt = new GBTRegressor()
      .setLabelCol("total_demand")
      .setFeaturesCol("features")
      .setMaxIter(50)
      .setMaxDepth(10)
      .setSeed(42)

    val gbtModel = gbt.fit(trainFinal)
    println("[INFO] GBT training complete.")

    // Training metrics — GBT
    val gbtTrainPredictions = gbtModel.transform(trainFinal)
    val gbtTrainRMSE = evalRMSE.evaluate(gbtTrainPredictions)
    val gbtTrainMAE  = evalMAE.evaluate(gbtTrainPredictions)
    val gbtTrainR2   = evalR2.evaluate(gbtTrainPredictions)

    println("\n============================================================")
    println("Gradient Boosted Trees — Training Set Results")
    println("============================================================")
    println(f"  RMSE : $gbtTrainRMSE%.4f")
    println(f"  MAE  : $gbtTrainMAE%.4f")
    println(f"  R2   : $gbtTrainR2%.4f")

    // ============================================================
    // STEP 12: Evaluation — GBT
    // ============================================================
    val gbtPredictions = gbtModel.transform(testFinal)

    val gbtRMSE = evalRMSE.evaluate(gbtPredictions)
    val gbtMAE  = evalMAE.evaluate(gbtPredictions)
    val gbtR2   = evalR2.evaluate(gbtPredictions)

    println("\n============================================================")
    println("Gradient Boosted Trees — Test Set Results")
    println("============================================================")
    println(f"  RMSE : $gbtRMSE%.4f")
    println(f"  MAE  : $gbtMAE%.4f")
    println(f"  R2   : $gbtR2%.4f")

    // ============================================================
    // STEP 13: Feature Importance — GBT
    // ============================================================
    println("\n============================================================")
    println("Feature Importances — Gradient Boosted Trees")
    println("============================================================")

    gbtModel.featureImportances.toArray
      .zip(featureNames)
      .sortBy(-_._1)
      .foreach { case (importance, name) =>
        println(f"  $name%-30s -> $importance%.4f")
      }

    // ============================================================
    // STEP 14: Full Comparison — RF vs GBT vs Baseline
    // ============================================================
    println("\n============================================================")
    println("Full Comparison: RF vs GBT vs Baseline")
    println("============================================================")
    println(f"  Metric  | Random Forest | GBT        | Baseline")
    println(f"  --------|---------------|------------|----------")
    println(f"  RMSE    | $rfRMSE%-13.4f | $gbtRMSE%-10.4f | $baselineRMSE%.4f")
    println(f"  MAE     | $rfMAE%-13.4f | $gbtMAE%-10.4f | $baselineMAE%.4f")
    println(f"  R2      | $rfR2%-13.4f | $gbtR2%-10.4f | $baselineR2%.4f")

    // ============================================================
    // STEP 15: Sample Predictions — Best Model
    // ============================================================
    val bestPredictions = if (gbtR2 > rfR2) gbtPredictions else rfPredictions
    val bestModelName   = if (gbtR2 > rfR2) "GBT" else "Random Forest"

    println(s"\n============================================================")
    println(s"Sample Predictions vs Actual Values — $bestModelName (Best Model)")
    println( "============================================================")
    bestPredictions.select("total_demand", "prediction").show(15, truncate = false)

    // ============================================================
    // STEP 16: Save Results
    // ============================================================
    println("\n[INFO] Saving ML metrics to results/ml_metrics.txt...")
    new java.io.File("results").mkdirs()

    val metricsContent =
      s"""Random Forest Regressor - Training Set Metrics
         |==========================================
         |RMSE : $rfTrainRMSE
         |MAE  : $rfTrainMAE
         |R2   : $rfTrainR2
         |
         |Random Forest Regressor - Test Set Metrics
         |==========================================
         |RMSE : $rfRMSE
         |MAE  : $rfMAE
         |R2   : $rfR2
         |
         |Gradient Boosted Trees - Training Set Metrics
         |==========================================
         |RMSE : $gbtTrainRMSE
         |MAE  : $gbtTrainMAE
         |R2   : $gbtTrainR2
         |
         |Gradient Boosted Trees - Test Set Metrics
         |==========================================
         |RMSE : $gbtRMSE
         |MAE  : $gbtMAE
         |R2   : $gbtR2
         |
         |Baseline (Mean Prediction) - Test Set Metrics
         |==========================================
         |RMSE : $baselineRMSE
         |MAE  : $baselineMAE
         |R2   : $baselineR2
         |
         |Best Model: $bestModelName
         |""".stripMargin

    val pw = new java.io.PrintWriter(new java.io.File("results/ml_metrics.txt"))
    pw.write(metricsContent)
    pw.close()

    println("[INFO] Metrics saved to results/ml_metrics.txt")

    spark.stop()
    println("\n[INFO] Spark session stopped. Phase 5 complete.")
  }
}