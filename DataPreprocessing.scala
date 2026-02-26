import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.hadoop.fs.{FileSystem, Path}

object DataPreprocessing {

  // -----------------------------------
  // Helper: read CSV with fixed schema
  // -----------------------------------
  def readCsv(spark: SparkSession, path: String, schema: StructType): DataFrame =
    spark.read
      .option("header", "true")
      .option("mode", "PERMISSIVE")
      .schema(schema)
      .csv(path)

  // -----------------------------------
  // Helper: write DF to a single CSV file
  // -----------------------------------
  def writeSingleCsv(df: DataFrame, tmpDir: String, finalFile: String)(implicit spark: SparkSession): Unit = {
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    df.coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv(tmpDir)

    val tmpPath = new Path(tmpDir)
    val partFile = fs.listStatus(tmpPath)
      .map(_.getPath)
      .find(p => p.getName.startsWith("part-") && p.getName.endsWith(".csv"))
      .getOrElse(throw new RuntimeException(s"No part-*.csv found in $tmpDir"))

    val finalPath = new Path(finalFile)
    if (fs.exists(finalPath)) fs.delete(finalPath, true)

    val ok = fs.rename(partFile, finalPath)
    if (!ok) throw new RuntimeException(s"Failed to rename $partFile -> $finalFile")

    fs.delete(tmpPath, true)
  }

  // -----------------------------------
  // CLEANING LOGIC (DO NOT CHANGE)
  // + counts per step
  // -----------------------------------
  def cleanData(df: DataFrame): DataFrame = {

    val criticalCols = Seq(
      "ride_id", "started_at", "ended_at",
      "start_station_name", "end_station_name",
      "member_casual"
    )
    val step1 = df.na.drop(criticalCols)
    println(s"[COUNT] Clean-Step1 (drop null critical cols): ${step1.count()}")

    val step2 = step1.dropDuplicates("ride_id")
    println(s"[COUNT] Clean-Step2 (dropDuplicates ride_id): ${step2.count()}")

    val step3 = step2
      .withColumn("started_ts", to_timestamp(col("started_at")))
      .withColumn("ended_ts", to_timestamp(col("ended_at")))
    println(s"[COUNT] Clean-Step3 (added timestamps cols): ${step3.count()}")

    val step4 = step3.na.drop(Seq("started_ts", "ended_ts"))
    println(s"[COUNT] Clean-Step4 (drop null started_ts/ended_ts): ${step4.count()}")

    val step5 = step4.withColumn(
      "trip_duration_min",
      (unix_timestamp(col("ended_ts")) - unix_timestamp(col("started_ts"))) / 60.0
    )
    println(s"[COUNT] Clean-Step5 (added duration col): ${step5.count()}")

    val step6 = step5.filter(col("trip_duration_min") > 0 && col("trip_duration_min") < 1440)
    println(s"[COUNT] Clean-Step6 (filter duration 0..1440): ${step6.count()}")

    val cleanDF = step6
      .withColumn("start_station_name", trim(col("start_station_name")))
      .withColumn("end_station_name", trim(col("end_station_name")))
      .withColumn("member_casual", lower(trim(col("member_casual"))))

    println(s"[COUNT] Clean-Final cleanedDF: ${cleanDF.count()}")
    cleanDF
  }

  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession.builder()
      .appName("Phase2-DataPreprocessing (Integrate + Clean + Reduce)")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", "200")

    // ---------------------------
    // Schema (consistent)
    // ---------------------------
    val schema = StructType(Seq(
      StructField("ride_id", StringType, true),
      StructField("rideable_type", StringType, true),
      StructField("started_at", StringType, true),
      StructField("ended_at", StringType, true),
      StructField("start_station_name", StringType, true),
      StructField("start_station_id", StringType, true),
      StructField("end_station_name", StringType, true),
      StructField("end_station_id", StringType, true),
      StructField("start_lat", StringType, true),
      StructField("start_lng", StringType, true),
      StructField("end_lat", StringType, true),
      StructField("end_lng", StringType, true),
      StructField("member_casual", StringType, true)
    ))

    // =========================================================
    // (A) INTEGRATION STEP
    // =========================================================
    val octParts = Seq(
      readCsv(spark, "data/raw/202510-citibike-tripdata_1.csv", schema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_2.csv", schema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_3.csv", schema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_4.csv", schema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_5.csv", schema)
    )

    val novParts = Seq(
      readCsv(spark, "data/raw/202511-citibike-tripdata_1.csv", schema),
      readCsv(spark, "data/raw/202511-citibike-tripdata_2.csv", schema),
      readCsv(spark, "data/raw/202511-citibike-tripdata_3.csv", schema),
      readCsv(spark, "data/raw/202511-citibike-tripdata_4.csv", schema)
    )

    val decParts = Seq(
      readCsv(spark, "data/raw/202512-citibike-tripdata_1.csv", schema),
      readCsv(spark, "data/raw/202512-citibike-tripdata_2.csv", schema),
      readCsv(spark, "data/raw/202512-citibike-tripdata_3.csv", schema)
    )

    val oct = octParts.reduce(_ unionByName _)
    val nov = novParts.reduce(_ unionByName _)
    val dec = decParts.reduce(_ unionByName _)
    val integrated = Seq(oct, nov, dec).reduce(_ unionByName _)

    oct.cache(); nov.cache(); dec.cache(); integrated.cache()

    val octCount = oct.count()
    val novCount = nov.count()
    val decCount = dec.count()
    val expectedTotal = octCount + novCount + decCount
    val actualTotal = integrated.count()

    println(s"[COUNT] Integrate-Oct rows: $octCount")
    println(s"[COUNT] Integrate-Nov rows: $novCount")
    println(s"[COUNT] Integrate-Dec rows: $decCount")
    println(s"[COUNT] Integrate-Expected total: $expectedTotal")
    println(s"[COUNT] Integrate-Actual total:   $actualTotal")

    require(expectedTotal == actualTotal, "Integration totals mismatch! Check missing files / union logic.")

    writeSingleCsv(
      integrated,
      tmpDir = "data/processed/integratedData_tmp",
      finalFile = "data/processed/integratedData.csv"
    )
    println("✅ Saved: data/processed/integratedData.csv")

    // =========================================================
    // (B) CLEANING STEP (read integratedData.csv)
    // =========================================================
    val integratedDF = spark.read
      .option("header", "true")
      .option("mode", "PERMISSIVE")
      .schema(schema)
      .csv("data/processed/integratedData.csv")

    println(s"[COUNT] Clean-Input (integratedData rows): ${integratedDF.count()}")

    val cleanedDF = cleanData(integratedDF)

    writeSingleCsv(
      cleanedDF,
      tmpDir = "data/processed/cleanedData_tmp",
      finalFile = "data/processed/cleanedData.csv"
    )
    println("✅ Saved: data/processed/cleanedData.csv")

    // =========================================================
    // (C) REDUCTION STEP (MATCHED TO reductionStep.scala)
    // =========================================================
    val c0 = cleanedDF.count()
    println(s"[COUNT] Reduce-Input (cleanedData rows): $c0")

    // تعديل الفلترة هنا لتطابق الملف المنفصل (النطاق التاريخي الصريح)
    val filtered = cleanedDF
      .filter(col("started_ts").isNotNull)
      .filter(col("started_ts") >= lit("2025-10-01") && col("started_ts") < lit("2026-01-01"))

    val c0b = filtered.count()
    println(s"[COUNT] Reduce-(0.1) After filtering Oct/Nov/Dec 2025: $c0b (removed ${c0 - c0b})")

    val withStrata = filtered
      .withColumn("year", year(col("started_ts")))
      .withColumn("month", month(col("started_ts")))
      .withColumn("year_month", concat_ws("-", col("year"), lpad(col("month").cast("string"), 2, "0")))

    val strata = withStrata.select("year_month").distinct().collect()
    val fractions: Map[String, Double] = strata.map(r => r.getAs[String]("year_month") -> 0.30).toMap

    val sampled = withStrata.stat.sampleBy("year_month", fractions, 42L)
    val c1 = sampled.count()
    println(s"[COUNT] Reduce-(1) After Sampling 30% stratified: $c1")
    println("[INFO]  Reduce-(1) Sample distribution by year_month:")
    sampled.groupBy("year_month").count().orderBy("year_month").show(50, truncate = false)

    val selected = sampled.select(
      col("rideable_type"),
      col("start_station_name"),
      col("end_station_name"),
      col("member_casual"),
      col("started_ts"),
      col("trip_duration_min")
    )
    println(s"[COUNT] Reduce-(2) After Feature Selection rows: ${selected.count()}")

    val reduced = selected
      .withColumn("hour_bucket", date_trunc("hour", col("started_ts")))
      .groupBy("hour_bucket", "start_station_name", "end_station_name", "member_casual")
      .agg(
        sum(when(col("rideable_type") === "classic_bike", 1).otherwise(0)).alias("classic_trips"),
        sum(when(col("rideable_type") === "electric_bike", 1).otherwise(0)).alias("electric_trips"),
        avg(col("trip_duration_min")).alias("avg_duration_min")
      )

    println(s"[COUNT] Reduce-(3) After Aggregation (rows/groups): ${reduced.count()}")

    writeSingleCsv(
      reduced,
      tmpDir = "data/processed/reducedData_tmp",
      finalFile = "data/processed/reducedData.csv"
    )
    println("✅ Saved: data/processed/reducedData.csv")

    spark.stop()
  }
}