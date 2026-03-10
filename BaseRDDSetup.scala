import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

/*
 * ============================================================
 * IT462 - Big Data Systems
 * Phase 3 - RDD Operations
 * Group 2
 *
 * This file reads the transformed dataset from Phase 2
 * ============================================================
 */

object BaseRDDSetup {

  // ------------------------------------------------------------
  // Each record represents an aggregated demand observation
  // for a specific date, hour, start station, end station,
  // user type, and trip counts.
  // ------------------------------------------------------------
  case class TripRecord(
    date: String,
    hour: Int,
    startStation: String,
    startStationIdx: Int,
    endStation: String,
    endStationIdx: Int,
    userType: String,
    userTypeIdx: Int,
    classicTrips: Int,
    electricTrips: Int,
    avgDurationMin: Double,
    dayOfWeek: Int,
    isWeekend: Int
  )

  // ------------------------------------------------------------
  // Safe parsing helpers
  // ------------------------------------------------------------
  def toSafeString(v: Any): String =
    if (v == null) "" else v.toString.trim

  def toSafeInt(v: Any): Int =
    try {
      if (v == null) 0 else v.toString.trim.toDouble.toInt
    } catch {
      case _: Throwable => 0
    }

  def toSafeDouble(v: Any): Double =
    try {
      if (v == null) 0.0 else v.toString.trim.toDouble
    } catch {
      case _: Throwable => 0.0
    }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Phase3-RDDOperations")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // ============================================================
    // STEP 0 - Read transformed dataset produced in Phase 2
    // Cast date to string to avoid Java 17 DateType issues in RDD
    // ============================================================
    val transformedDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/processed/transformedData.csv")
      .withColumn("date", col("date").cast("string"))

    println("[INFO] Schema of transformedData.csv:")
    transformedDF.printSchema()

    // ============================================================
    // SHARED BASE RDD
    // ============================================================
    val rawRDD = transformedDF.rdd

    val baseRDD = rawRDD.map { row =>
      TripRecord(
        date            = toSafeString(row.getAs[Any]("date")),
        hour            = toSafeInt(row.getAs[Any]("hour")),
        startStation    = toSafeString(row.getAs[Any]("start_station_name")),
        startStationIdx = toSafeInt(row.getAs[Any]("start_station_name_idx")),
        endStation      = toSafeString(row.getAs[Any]("end_station_name")),
        endStationIdx   = toSafeInt(row.getAs[Any]("end_station_name_idx")),
        userType        = toSafeString(row.getAs[Any]("member_casual")),
        userTypeIdx     = toSafeInt(row.getAs[Any]("member_casual_idx")),
        classicTrips    = toSafeInt(row.getAs[Any]("classic_trips")),
        electricTrips   = toSafeInt(row.getAs[Any]("electric_trips")),
        avgDurationMin  = toSafeDouble(row.getAs[Any]("avg_duration_min")),
        dayOfWeek       = toSafeInt(row.getAs[Any]("day_of_week")),
        isWeekend       = toSafeInt(row.getAs[Any]("is_weekend"))
      )
    }

    // Simple action to confirm that baseRDD is ready
    val baseCount = baseRDD.count()
    println(s"[INFO] baseRDD records: $baseCount")

    println("[INFO] baseRDD is ready. Each student can now continue from baseRDD.")

    spark.stop()
  }
}