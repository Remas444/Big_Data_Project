import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler}

/*
 * ============================================================
 * IT462 - Big Data Systems
 * Phase 5 - Machine Learning with Spark
 * Group 2
 *
 * Part 1: Feature Engineering and Preparation 
 * (Target Variable definition, Categorical Encoding, Vector Assembling)
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
    println("Loading preprocessed dataset...")
    val tripsDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/processed/transformedData.csv")

    // ============================================================
    // STEP 2: Target Variable Preparation (Aggregation)
    // ============================================================
    // As defined in the Problem Formulation, we need to predict the total demand at a specific start station in a given hour.
    // We group by temporal features and station, then sum the trips.
    println("Aggregating data to compute the target variable 'total_demand'...")
    val demandDF = tripsDF.groupBy(
      col("date"),
      col("hour"),
      col("start_station_name_idx"),
      col("day_of_week"),
      col("is_weekend")
    ).agg(
      sum(expr("classic_trips + electric_trips")).alias("total_demand")
    )

    // ============================================================
    // STEP 3: Categorical Encoding (OneHotEncoder)
    // ============================================================
    // 'start_station_name_idx' is already indexed via StringIndexer in Phase 2.
    // We apply OneHotEncoder to 'day_of_week' to prevent the model from assuming mathematical relationships between the days.
    println("Applying OneHotEncoder to 'day_of_week'...")
    val encoder = new OneHotEncoder()
      .setInputCols(Array("day_of_week"))
      .setOutputCols(Array("day_of_week_ohe"))
      .setDropLast(false)

    val encodedDF = encoder.fit(demandDF).transform(demandDF)

    // ============================================================
    // STEP 4: Feature Integration (VectorAssembler)
    // ============================================================
    // Combine all selected features into a single vector column named 'features'.
    // NOTE: No scaling is needed here. Random Forest is not affected by the size or scale of the numbers.
    println("Assembling features into a single vector...")
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "start_station_name_idx", "day_of_week_ohe", "is_weekend"))
      .setOutputCol("features")

    val finalFeaturesDF = assembler.transform(encodedDF)

    // ============================================================
    // STEP 5: Final Dataset Preparation for Model Training
    // ============================================================
    // Keep only the 'features' input vector and the 'total_demand' target label.
    val mlReadyDF = finalFeaturesDF.select("features", "total_demand")

    println("\n============================================================")
    println("Snapshot of Feature Engineered Data (Ready for ML Split & Train)")
    println("============================================================")
    mlReadyDF.show(10, truncate = false)
    
  }
}
