import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.hadoop.fs.{FileSystem, Path}

object integrateStep {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Phase2-Integration")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // 1) Schema (consistent across all files)
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

    // 2) Reader helper
    def readCsv(path: String): DataFrame =
      spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(path)

    // 3) Read your split files (names must match data/raw exactly)
    val octParts = Seq(
      readCsv("data/raw/202510-citibike-tripdata_1.csv"),
      readCsv("data/raw/202510-citibike-tripdata_2.csv"),
      readCsv("data/raw/202510-citibike-tripdata_3.csv"),
      readCsv("data/raw/202510-citibike-tripdata_4.csv"),
      readCsv("data/raw/202510-citibike-tripdata_5.csv")
    )

    val novParts = Seq(
      readCsv("data/raw/202511-citibike-tripdata_1.csv"),
      readCsv("data/raw/202511-citibike-tripdata_2.csv"),
      readCsv("data/raw/202511-citibike-tripdata_3.csv"),
      readCsv("data/raw/202511-citibike-tripdata_4.csv")
    )

    val decParts = Seq(
      readCsv("data/raw/202512-citibike-tripdata_1.csv"),
      readCsv("data/raw/202512-citibike-tripdata_2.csv"),
      readCsv("data/raw/202512-citibike-tripdata_3.csv")
    )

    // 4) Union parts per month
    val oct = octParts.reduce(_ unionByName _)
    val nov = novParts.reduce(_ unionByName _)
    val dec = decParts.reduce(_ unionByName _)

    // 5) Union all months
    val integrated = Seq(oct, nov, dec).reduce(_ unionByName _)

    // ---------------------------
    // ✅ PROOF BLOCK (integration check)
    // ---------------------------
    oct.cache()
    nov.cache()
    dec.cache()
    integrated.cache()

    val octCount = oct.count()
    val novCount = nov.count()
    val decCount = dec.count()

    val expectedTotal = octCount + novCount + decCount
    val actualTotal = integrated.count()

    println(s"Oct rows: $octCount")
    println(s"Nov rows: $novCount")
    println(s"Dec rows: $decCount")
    println(s"Expected total (Oct+Nov+Dec): $expectedTotal")
    println(s"Actual integrated total:       $actualTotal")

    if (expectedTotal == actualTotal) {
      println("✅ Integration check PASSED: expected total equals integrated total.")
    } else {
      println("❌ Integration check FAILED: totals do not match!")
      println(s"Difference (actual - expected): ${actualTotal - expectedTotal}")
    }

    require(expectedTotal == actualTotal, "Integration totals mismatch! Check union logic / missing files.")

    // 6) Print schema + a small sample
    println("Final schema:")
    integrated.printSchema()

    println("Sample rows (10):")
    integrated.show(10, truncate = false)

    // ---------------------------
    // ✅ SAVE FULL DATASET AS ONE SINGLE CSV FILE
    // ---------------------------
    val tmpDir = "data/processed/integrated_single_csv_tmp"
    val finalFile = "data/processed/integrated.csv"

    // Write into one partition => one part file
    integrated
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(tmpDir)

    // Rename the single part file to integrated.csv
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val tmpPath = new Path(tmpDir)

    val partFile = fs.listStatus(tmpPath)
      .map(_.getPath)
      .find(p => p.getName.startsWith("part-") && p.getName.endsWith(".csv"))
      .getOrElse(throw new RuntimeException("No part CSV file found in temp output folder!"))

    val finalPath = new Path(finalFile)

    // If integrated.csv already exists, delete it
    if (fs.exists(finalPath)) fs.delete(finalPath, true)

    // Move part file -> integrated.csv
    val renamed = fs.rename(partFile, finalPath)
    if (!renamed) throw new RuntimeException("Failed to rename part file to integrated.csv")

    // Clean up temp folder
    fs.delete(tmpPath, true)

    println(s"✅ Saved single CSV file to: $finalFile")

    // 7) Optional: also save parquet for later speed (recommended)
    // integrated.write.mode("overwrite").parquet("data/processed/integrated_raw.parquet")

    // 8) Save small snapshot for report (20 rows, single file)
    integrated.limit(20)
      .coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv("data/processed/snapshot_20rows_csv")

    println("Saved 20-row snapshot to: data/processed/snapshot_20rows_csv")

    spark.stop()
  }
}