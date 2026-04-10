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
 * Model:   Random Forest Regressor (Spark MLlib)
 * ============================================================
 */

object MachineLearning_commented {

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
    // Goal: predict total trips originating from a start station in a given hour.
    // We focus on the start station only — operators need to know WHERE bikes
    // will be needed before trips begin, not the full route.
    //
    // Two-step aggregation:
    // First, we aggregate at the fine-grained level (including end_station_name_idx)
    // to preserve end-station information for the avg_incoming_trips feature later.
    // Then, we re-aggregate at the correct modeling granularity:
    // (date, hour, start_station_name_idx) — this is the level at which operators
    // make redistribution decisions.
    //
    // Keeping end_station and member_casual in the final groupBy would produce
    // total_demand = 1 for almost every row (too granular), leaving the model
    // nothing meaningful to learn from.
    //
    // 'date' is kept for correct temporal grouping but excluded from model features —
    // the model learns from hour and day_of_week patterns, not specific calendar dates,
    // ensuring generalization to future unseen dates.
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

    // Show target variable distribution to confirm meaningful variance
    println("\n[INFO] Target variable (total_demand) distribution:")
    demandDF.select("total_demand").summary("min", "25%", "50%", "75%", "max", "mean").show()

    // ============================================================
    // STEP 3: Train / Test Split  (BEFORE any feature fitting)
    // ============================================================
    // Fixed seed = 42 ensures reproducibility across runs.
    // Split is performed BEFORE any encoding or feature computation
    // to strictly prevent data leakage — test data must not influence
    // any step that learns statistics from the data.
    println("\n[INFO] Splitting data into 70% train / 30% test (seed=42)...")
    val Array(trainRaw, testRaw) = demandDF.randomSplit(Array(0.7, 0.3), seed = 42)

    trainRaw.cache()
    testRaw.cache()

    println(s"[INFO] Training set size : ${trainRaw.count()}")
    println(s"[INFO] Test set size     : ${testRaw.count()}")

    // ============================================================
    // STEP 4: Feature Engineering — avg_incoming_trips
    // ============================================================
    // Some stations naturally accumulate bikes because many trips end there.
    // This feature captures the historical incoming demand pattern per station,
    // which correlates with how active and popular a station is overall.
    //
    // Computed from training rows in fineGrainedDF ONLY, then joined to both
    // train and test sets. Using the full dataset here would leak test information
    // into the training feature, violating the no-leakage requirement.
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
    // day_of_week values (1-7) have no real numeric ordering for the model.
    // OneHotEncoder prevents the model from assuming Monday(1) < Friday(5).
    // Fitted on training data only, then applied to both sets.
    //
    // start_station_name_idx is used directly as a numeric index from Phase 2.
    // OHE is not applied to stations because ~2200 unique stations would cause
    // dimensional explosion. Random Forest handles index-based encoding without
    // introducing harmful ordinal bias for tree-based splits.
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
    // Spark ML requires all input features combined into a single vector column.
    // No feature scaling is applied — Random Forest is a tree-based model
    // and is invariant to feature magnitude and scale. Scaling would add
    // complexity with no benefit for this model type.
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
    // A naive baseline that always predicts the training mean.
    // The Random Forest must outperform this to demonstrate real learning.
    // R2 of the baseline will be ~0, so any positive R2 from RF shows improvement.
    println("\n[INFO] Computing baseline (mean prediction)...")
    val trainMean = trainFinal
      .agg(avg("total_demand"))
      .first()
      .getDouble(0)

    println(f"[INFO] Training mean total_demand = $trainMean%.4f")

    val baselineDF = testFinal.withColumn("baseline_prediction", lit(trainMean))

    val baseEvalRMSE = new RegressionEvaluator()
      .setLabelCol("total_demand")
      .setPredictionCol("baseline_prediction")
      .setMetricName("rmse")

    val baseEvalMAE = new RegressionEvaluator()
      .setLabelCol("total_demand")
      .setPredictionCol("baseline_prediction")
      .setMetricName("mae")

    val baseEvalR2 = new RegressionEvaluator()
      .setLabelCol("total_demand")
      .setPredictionCol("baseline_prediction")
      .setMetricName("r2")

    val baselineRMSE = baseEvalRMSE.evaluate(baselineDF)
    val baselineMAE  = baseEvalMAE.evaluate(baselineDF)
    val baselineR2   = baseEvalR2.evaluate(baselineDF)

    println("\n============================================================")
    println("Baseline Model Results (Always predict training mean)")
    println("============================================================")
    println(f"  RMSE : $baselineRMSE%.4f")
    println(f"  MAE  : $baselineMAE%.4f")
    println(f"  R2   : $baselineR2%.4f")

    // ============================================================
    // STEP 8: Random Forest Regressor — Training
    // ============================================================
    // Random Forest is chosen because:
    //   - Demand vs features relationship is non-linear
    //   - Handles high-cardinality station index without OHE explosion
    //   - Robust to outliers common in real-world trip data
    //   - Provides interpretable feature importance scores
    // numTrees=100 and maxDepth=15 for stronger learning vs default settings.
    // seed=42 ensures reproducible results.
    println("\n[INFO] Training Random Forest Regressor...")
    val rf = new RandomForestRegressor()
      .setLabelCol("total_demand")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(15)
      .setSeed(42)

    val rfModel = rf.fit(trainFinal)
    println("[INFO] Random Forest training complete.")

    // ============================================================
    // STEP 9: Evaluation — Random Forest
    // ============================================================
    val rfPredictions = rfModel.transform(testFinal)

    val evalRMSE = new RegressionEvaluator()
      .setLabelCol("total_demand")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val evalMAE = new RegressionEvaluator()
      .setLabelCol("total_demand")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    val evalR2 = new RegressionEvaluator()
      .setLabelCol("total_demand")
      .setPredictionCol("prediction")
      .setMetricName("r2")

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
    // Feature names aligned with VectorAssembler input order.
    // day_of_week_ohe produces 7 binary columns (one per day, setDropLast=false).
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
    // GBT builds trees sequentially, each correcting the errors of the previous.
    // This often achieves higher accuracy than Random Forest on tabular data
    // because it focuses learning on hard-to-predict samples.
    // maxIter=50 controls the number of boosting rounds.
    // maxDepth=10 keeps individual trees shallow to avoid overfitting.
    // seed=42 ensures reproducibility.
    println("\n[INFO] Training Gradient Boosted Trees Regressor...")
    val gbt = new GBTRegressor()
      .setLabelCol("total_demand")
      .setFeaturesCol("features")
      .setMaxIter(50)
      .setMaxDepth(10)
      .setSeed(42)

    val gbtModel = gbt.fit(trainFinal)
    println("[INFO] GBT training complete.")

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
      s"""Random Forest Regressor - Test Set Metrics
         |==========================================
         |RMSE : $rfRMSE
         |MAE  : $rfMAE
         |R2   : $rfR2
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