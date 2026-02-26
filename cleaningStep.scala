import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import java.nio.file.{Files, Paths, StandardCopyOption}
import java.io.File

object cleaningStep {

  def cleanData(df: DataFrame): DataFrame = {

    val criticalCols = Seq(
      "ride_id", "started_at", "ended_at",
      "start_station_name", "end_station_name",
      "member_casual"
    )
    val step1 = df.na.drop(criticalCols)
    println(s"[COUNT] Step1 (drop null critical cols): ${step1.count()}")

    val step2 = step1.dropDuplicates("ride_id")
    println(s"[COUNT] Step2 (dropDuplicates ride_id): ${step2.count()}")

    val step3 = step2
      .withColumn("started_at", to_timestamp(col("started_at")))
      .withColumn("ended_at", to_timestamp(col("ended_at")))
    println(s"[COUNT] Step3 (converted started_at/ended_at to timestamp): ${step3.count()}")

    val step4 = step3.na.drop(Seq("started_at", "ended_at"))
    println(s"[COUNT] Step4 (drop null started_at/ended_at): ${step4.count()}")

    val step5 = step4.withColumn(
      "trip_duration_min",
      (unix_timestamp(col("ended_at")) - unix_timestamp(col("started_at"))) / 60.0
    )
    println(s"[COUNT] Step5 (added duration col): ${step5.count()}")

    val step6 = step5.filter(col("trip_duration_min") > 0 && col("trip_duration_min") < 1440)
    println(s"[COUNT] Step6 (filter duration 0..1440): ${step6.count()}")

    val cleanDF = step6
      .withColumn("start_station_name", trim(col("start_station_name")))
      .withColumn("end_station_name", trim(col("end_station_name")))
      .withColumn("member_casual", lower(trim(col("member_casual"))))

    println(s"[COUNT] Final CleanDF: ${cleanDF.count()}")

    cleanDF
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Phase2-Cleaning")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", "200")

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

    val integratedPath = "data/processed/integrated.csv"

    val integrated = spark.read
      .option("header", "true")
      .option("mode", "PERMISSIVE")
      .schema(schema)
      .csv(integratedPath)

    println(s"[COUNT] Before cleaning (integrated rows): ${integrated.count()}")

    val cleanDF = cleanData(integrated)

    // ✅ SAVE OUTPUT as a single CSV file named cleanDF.csv
    val tempDir = "data/processed/cleanDF_csv_single"
    cleanDF
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(tempDir)

    val dir = new File(tempDir)
    val partFile = dir.listFiles().find(_.getName.startsWith("part-")).get
    val finalPath = Paths.get("data/processed/cleanDF.csv")
    Files.deleteIfExists(finalPath)

    Files.move(
      partFile.toPath,
      finalPath,
      StandardCopyOption.REPLACE_EXISTING
    )

    // Snapshot for report
    cleanDF.limit(20)
      .coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv("data/processed/clean_snapshot_20rows_csv")

    println("✅ Saved: data/processed/cleanDF.csv")
    println("✅ Saved snapshot: data/processed/clean_snapshot_20rows_csv")

    spark.stop()
  }
}
