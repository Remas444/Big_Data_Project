import org.apache.spark.sql.{Column, DataFrame, SparkSession}
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

  def parseTimestamp(c: Column): Column = {
    coalesce(
      to_timestamp(c, "yyyy-MM-dd HH:mm:ss"),
      to_timestamp(c, "yyyy-MM-dd HH:mm:ss.SSS"),
      to_timestamp(c, "yyyy-MM-dd'T'HH:mm:ss.SSSX"),
      to_timestamp(c, "yyyy-MM-dd'T'HH:mm:ssX"),
      to_timestamp(c)
    )
  }

  def printNullReport(df: DataFrame, cols: Seq[String], label: String): Unit = {
    val exprs = cols.map { c =>
      sum(
        when(col(c).isNull, 1).otherwise(0)
      ).alias(c)
    }
    val row = df.select(exprs: _*).collect()(0)
    val total = df.count()

    println(s"[NULL-REPORT] $label (rows=$total)")
    cols.zipWithIndex.foreach { case (c, i) =>
      val n = row.getLong(i)
      val p = if (total == 0) 0.0 else (n.toDouble / total.toDouble) * 100.0
      println(f"  - $c: $n nulls ($p%.4f%%)")
    }
  }

  val criticalColsDrop = Seq(
    "ride_id",
    "start_station_name", "end_station_name",
    "started_at", "ended_at"
  )

  // (0) Basic null report on columns that must remain usable after cleaning
  val reportCols = Seq(
    "ride_id", "rideable_type",
    "started_at", "ended_at",
    "start_station_name", "end_station_name",
    "member_casual"
  ).filter(df.columns.contains)

  printNullReport(df, reportCols, "Clean-Step0 BEFORE any cleaning")

  val step1 = df.na.drop(criticalColsDrop)
  println(s"[COUNT] Clean-Step1 (drop null essential cols): ${step1.count()}")

  // Standardize text early (so mode calculation is consistent)
  val step1b = step1
    .withColumn("start_station_name", trim(col("start_station_name")))
    .withColumn("end_station_name", trim(col("end_station_name")))
    .withColumn("member_casual", lower(trim(col("member_casual"))))

  // (1.1) Impute member_casual nulls with mode (member/casual)
  val mcCol = "member_casual"
  val mcNullsBefore = step1b.filter(col(mcCol).isNull).count()
  println(s"[COUNT] Clean-Step1.1 (member_casual nulls BEFORE imputation): $mcNullsBefore")

  val mcModeOpt = step1b
    .filter(col(mcCol).isNotNull)
    .groupBy(col(mcCol))
    .count()
    .orderBy(desc("count"))
    .limit(1)
    .collect()
    .headOption
    .map(_.getString(0))

  val step1c = mcModeOpt match {
    case Some(modeVal) =>
      val tmp = step1b.na.fill(Map(mcCol -> modeVal))
      val mcNullsAfter = tmp.filter(col(mcCol).isNull).count()
      println(s"[COUNT] Clean-Step1.2 (member_casual nulls AFTER imputation): $mcNullsAfter (mode='$modeVal')")
      tmp
    case None =>
      println("[INFO] Clean-Step1.2 (member_casual mode not found; dataset may be empty after Step1)")
      step1b
  }

  val step2 = step1c.dropDuplicates("ride_id")
  println(s"[COUNT] Clean-Step2 (dropDuplicates ride_id): ${step2.count()}")

  // (3) Convert timestamps with explicit formats (avoid unexpected nulls after conversion)
  val step3 = step2
    .withColumn("started_at_raw", col("started_at"))
    .withColumn("ended_at_raw", col("ended_at"))
    .withColumn("started_at", parseTimestamp(col("started_at")))
    .withColumn("ended_at", parseTimestamp(col("ended_at")))

  println(s"[COUNT] Clean-Step3 (convert started_at/ended_at to timestamp): ${step3.count()}")

  val parseCheck = step3.select(
    sum(when(col("started_at").isNull, 1).otherwise(0)).alias("started_at_nulls_after_parse"),
    sum(when(col("ended_at").isNull, 1).otherwise(0)).alias("ended_at_nulls_after_parse")
  ).collect()(0)

  val startedNullsAfterParse = parseCheck.getLong(0)
  val endedNullsAfterParse = parseCheck.getLong(1)
  println(s"[COUNT] Clean-Step3.1 (started_at nulls AFTER parse): $startedNullsAfterParse")
  println(s"[COUNT] Clean-Step3.2 (ended_at nulls AFTER parse): $endedNullsAfterParse")

  val step4 = step3.na.drop(Seq("started_at", "ended_at"))
    .drop("started_at_raw", "ended_at_raw")

  println(s"[COUNT] Clean-Step4 (drop null started_at/ended_at after parsing): ${step4.count()}")

  val step5 = step4.withColumn(
    "trip_duration_min",
    (unix_timestamp(col("ended_at")) - unix_timestamp(col("started_at"))) / 60.0
  )
  println(s"[COUNT] Clean-Step5 (added duration col): ${step5.count()}")

  val step6 = step5.filter(col("trip_duration_min") > 0 && col("trip_duration_min") < 1440)
  println(s"[COUNT] Clean-Step6 (filter duration 0..1440): ${step6.count()}")

  val cleanDF = step6
    .withColumn("start_station_name", trim(col("start_station_name")))
    .withColumn("end_station_name", trim(col("end_station_name")))
    .withColumn("member_casual", lower(trim(col("member_casual"))))

  printNullReport(cleanDF, reportCols.filter(cleanDF.columns.contains), "Clean-Final AFTER cleaning")

  println(s"[COUNT] Clean-Final cleanedDF: ${cleanDF.count()}")
  cleanDF
}

  // -----------------------------------
  // Helper: cast schema after reading cleanedData.csv (timestamps/double)
  // IMPORTANT: cleanedData.csv gets saved with started_ts/ended_ts/trip_duration_min as STRING in CSV
  // so we must cast back before reduction.
  // -----------------------------------
  def castCleanedFromCsv(df: DataFrame): DataFrame = {

  def parseTimestamp(c: Column): Column = {
    coalesce(
      to_timestamp(c, "yyyy-MM-dd HH:mm:ss"),
      to_timestamp(c, "yyyy-MM-dd HH:mm:ss.SSS"),
      to_timestamp(c, "yyyy-MM-dd'T'HH:mm:ss.SSSX"),
      to_timestamp(c, "yyyy-MM-dd'T'HH:mm:ssX"),
      to_timestamp(c)
    )
  }

  df
    .withColumn("started_at", parseTimestamp(col("started_at")))
    .withColumn("ended_at", parseTimestamp(col("ended_at")))
    .withColumn("trip_duration_min", col("trip_duration_min").cast("double"))
    .withColumn("start_station_name", trim(col("start_station_name")))
    .withColumn("end_station_name", trim(col("end_station_name")))
    .withColumn("member_casual", lower(trim(col("member_casual"))))
}
  
  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession.builder()
      .appName("Phase2-DataPreprocessing (Integrate -> Clean -> Reduce using CSV between steps)")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", "200")

    // ---------------------------
    // Raw schema (input files)
    // ---------------------------
    val rawSchema = StructType(Seq(
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
    // (A) INTEGRATION STEP -> writes integratedData.csv
    // =========================================================
    val octParts = Seq(
      readCsv(spark, "data/raw/202510-citibike-tripdata_1.csv", rawSchema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_2.csv", rawSchema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_3.csv", rawSchema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_4.csv", rawSchema),
      readCsv(spark, "data/raw/202510-citibike-tripdata_5.csv", rawSchema)
    )

    val novParts = Seq(
      readCsv(spark, "data/raw/202511-citibike-tripdata_1.csv", rawSchema),
      readCsv(spark, "data/raw/202511-citibike-tripdata_2.csv", rawSchema),
      readCsv(spark, "data/raw/202511-citibike-tripdata_3.csv", rawSchema),
      readCsv(spark, "data/raw/202511-citibike-tripdata_4.csv", rawSchema)
    )

    val decParts = Seq(
      readCsv(spark, "data/raw/202512-citibike-tripdata_1.csv", rawSchema),
      readCsv(spark, "data/raw/202512-citibike-tripdata_2.csv", rawSchema),
      readCsv(spark, "data/raw/202512-citibike-tripdata_3.csv", rawSchema)
    )

    val oct = octParts.reduce(_ unionByName _)
    val nov = novParts.reduce(_ unionByName _)
    val dec = decParts.reduce(_ unionByName _)
    val integrated = Seq(oct, nov, dec).reduce(_ unionByName _)

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
    // (B) CLEANING STEP -> reads integratedData.csv -> writes cleanedData.csv
    // =========================================================
    val integratedFromFile = spark.read
      .option("header", "true")
      .option("mode", "PERMISSIVE")
      .schema(rawSchema)
      .csv("data/processed/integratedData.csv")

    println(s"[COUNT] Clean-Input (integratedData.csv rows): ${integratedFromFile.count()}")

    val cleanedDF = cleanData(integratedFromFile)

    writeSingleCsv(
      cleanedDF,
      tmpDir = "data/processed/cleanedData_tmp",
      finalFile = "data/processed/cleanedData.csv"
    )
    println("✅ Saved: data/processed/cleanedData.csv")

    // =========================================================
    // (C) REDUCTION STEP -> reads cleanedData.csv (NOT memory) -> writes reducedData.csv
    // =========================================================
    val cleanedFromFileRaw = spark.read
      .option("header", "true")
      .option("mode", "PERMISSIVE")
      .csv("data/processed/cleanedData.csv")

    println(s"[COUNT] Reduce-Read (cleanedData.csv rows): ${cleanedFromFileRaw.count()}")
    println("[INFO] cleanedData.csv schema (as read):")
    cleanedFromFileRaw.printSchema()

    // Cast back to correct types (important!)
    val cleanedFromFile = castCleanedFromCsv(cleanedFromFileRaw)
    println("[INFO] cleanedData.csv schema (after casts):")
    cleanedFromFile.printSchema()

    val c0 = cleanedFromFile.count()
    println(s"[COUNT] Reduce-Input (cleanedData.csv rows after cast): $c0")

    val filtered = cleanedFromFile
  .filter(col("started_at").isNotNull)
  .filter(col("started_at") >= lit("2025-10-01") && col("started_at") < lit("2026-01-01"))

    val c0b = filtered.count()
    println(s"[COUNT] Reduce-(0.1) After filtering Oct/Nov/Dec 2025: $c0b (removed ${c0 - c0b})")

    val withStrata = filtered
      .withColumn("year", year(col("started_at")))
      .withColumn("month", month(col("started_at")))
      .withColumn("year_month", concat_ws("-", col("year"), lpad(col("month").cast("string"), 2, "0")))

    println("[INFO] Reduce-(0.2) Distribution by year_month BEFORE sampling:")
    withStrata.groupBy("year_month").count().orderBy("year_month").show(50, truncate = false)

    val strata = withStrata.select("year_month").distinct().collect()
    val fractions: Map[String, Double] =
      strata.map(r => r.getAs[String]("year_month") -> 0.30).toMap

    val sampled = withStrata.stat.sampleBy("year_month", fractions, 42L)
    val c1 = sampled.count()
    println(s"[COUNT] Reduce-(1) After Sampling 30% stratified: $c1")
    println("[INFO]  Reduce-(1.1) Sample distribution by year_month AFTER sampling:")
    sampled.groupBy("year_month").count().orderBy("year_month").show(50, truncate = false)

    val selected = sampled.select(
      col("rideable_type"),
      col("start_station_name"),
      col("end_station_name"),
      col("member_casual"),
      col("started_at"),
      col("trip_duration_min")
    )
    println(s"[COUNT] Reduce-(2) After Feature Selection rows: ${selected.count()}")
    println(s"[INFO]  Reduce-(2) Columns: ${selected.columns.mkString(", ")}")

    val reduced = selected
      .withColumn("hour_bucket", date_trunc("hour", col("started_at")))
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
