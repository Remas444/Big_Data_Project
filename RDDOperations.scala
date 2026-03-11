import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

/*
 * ============================================================
 * IT462 - Big Data Systems
 * Phase 3 - RDD Operations
 * Group 2
 *
 * This file reads the transformed dataset from Phase 2
 * and performs meaningful RDD transformations and actions
 * for Phase 3 analysis.
 * ============================================================
 */

object RDDOperations {

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

  // ------------------------------------------------------------
  // Helper labels for readable output
  // ------------------------------------------------------------
  def dayName(day: Int): String = day match {
    case 1 => "Sunday"
    case 2 => "Monday"
    case 3 => "Tuesday"
    case 4 => "Wednesday"
    case 5 => "Thursday"
    case 6 => "Friday"
    case 7 => "Saturday"
    case _ => s"Unknown($day)"
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Phase3-RDDOperations")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    println("============================================================")
    println("IT462 - Big Data Systems")
    println("Phase 3 - RDD Operations")
    println("============================================================")

    // ============================================================
    // STEP 0 - Read transformed dataset produced in Phase 2
    // Cast date to string to avoid Java 17 DateType issues in RDD
    // ============================================================
    val transformedDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/processed/transformedData.csv")
      .withColumn("date", col("date").cast("string"))

    println("\n[INFO] Schema of transformedData.csv:")
    transformedDF.printSchema()

    // ============================================================
    // Transformation 1
    // Convert DataFrame rows into rawRDD
    // ============================================================
    println("\n==================== Transformation 1 ====================")
    println("Convert transformed DataFrame into rawRDD")

    val rawRDD = transformedDF.rdd

    // ============================================================
    // Transformation 2
    // Build shared baseRDD as TripRecord objects
    // ============================================================
    println("\n==================== Transformation 2 ====================")
    println("Create baseRDD as TripRecord objects from transformedData.csv")

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

    // ============================================================
    // Action 1
    // Confirm baseRDD is ready
    // ============================================================
    println("\n======================= Action 1 =========================")
    println("Count total records in baseRDD")

    val baseCount = baseRDD.count()
    println(s"Action 1 Output -> baseRDD records = $baseCount")

    println("\n[INFO] baseRDD is ready. Continuing with Phase 3 analysis...")

    // ============================================================
    // TIME (HOUR) DEMAND ANALYSIS
    // ============================================================

    // ============================================================
    // Transformation 3
    // Map each record to (hour, totalTrips)
    // ============================================================
    println("\n==================== Transformation 3 ====================")
    println("Map each record to (hour, totalTrips) where totalTrips = classicTrips + electricTrips")

    val hourTripPairs = baseRDD.map { record =>
      val totalTrips = record.classicTrips + record.electricTrips
      (record.hour, totalTrips)
    }

    // ============================================================
    // Transformation 4
    // Aggregate total demand by hour
    // ============================================================
    println("\n==================== Transformation 4 ====================")
    println("Aggregate total demand for each hour using reduceByKey")

    val hourlyDemand = hourTripPairs.reduceByKey(_ + _)

    // ============================================================
    // Transformation 5
    // Sort hours by demand descending
    // ============================================================
    println("\n==================== Transformation 5 ====================")
    println("Sort hourly demand from highest to lowest")

    val hourlyDemandSorted = hourlyDemand.sortBy(
      { case (_, totalTrips) => totalTrips },
      ascending = false
    )

    // ============================================================
    // Action 2
    // Show top 10 peak hours and identify the peak hour
    // ============================================================
    println("\n======================= Action 2 =========================")
    println("Take top 10 peak hours by total demand and identify the peak hour")

    val top10Hours = hourlyDemandSorted.take(10)

    println("Action 2 Output -> Top 10 peak hours:")
    top10Hours.foreach { case (hour, totalTrips) =>
      println(f"Hour $hour%02d -> Total Trips = $totalTrips")
    }

    if (top10Hours.nonEmpty) {
      val peakHour = top10Hours(0)
      println(f"\nPeak Hour -> Hour ${peakHour._1}%02d with Total Trips = ${peakHour._2}")
    }

    // ============================================================
    // WEEKDAY VS WEEKEND HOURLY DEMAND
    // ============================================================

    // ============================================================
    // Transformation 6
    // Map each record to ((isWeekend, hour), totalTrips)
    // ============================================================
    println("\n==================== Transformation 6 ====================")
    println("Map each record to ((isWeekend, hour), totalTrips)")

    val weekendHourPairs = baseRDD.map { record =>
      val totalTrips = record.classicTrips + record.electricTrips
      ((record.isWeekend, record.hour), totalTrips)
    }

    // ============================================================
    // Transformation 7
    // Aggregate demand by weekday/weekend and hour
    // ============================================================
    println("\n==================== Transformation 7 ====================")
    println("Aggregate demand by weekday/weekend and hour using reduceByKey")

    val weekendHourDemand = weekendHourPairs.reduceByKey(_ + _)

    // ============================================================
    // Action 3
    // Show weekday and weekend top hours side by side
    // ============================================================
    println("\n======================= Action 3 =========================")
    println("Collect weekday and weekend hourly demand, then display the ranked patterns side by side")

    val weekendHourResults = weekendHourDemand.collect().toList

    val weekdaySorted = weekendHourResults
      .filter { case ((isWeekend, _), _) => isWeekend == 0 }
      .sortBy { case ((_, hour), totalTrips) => (-totalTrips, hour) }

    val weekendSorted = weekendHourResults
      .filter { case ((isWeekend, _), _) => isWeekend == 1 }
      .sortBy { case ((_, hour), totalTrips) => (-totalTrips, hour) }

    val maxRows = math.max(weekdaySorted.length, weekendSorted.length)

    println("Action 3 Output -> Weekend vs Weekday ranked hourly demand")
    println()
    println(f"${"Weekend"}%-35s | ${"Weekday"}%-35s")
    println("-" * 75)

    for (i <- 0 until maxRows) {
      val weekendText =
        if (i < weekendSorted.length) {
          val ((_, hour), totalTrips) = weekendSorted(i)
          f"Hour $hour%02d -> Total Trips = $totalTrips"
        } else ""

      val weekdayText =
        if (i < weekdaySorted.length) {
          val ((_, hour), totalTrips) = weekdaySorted(i)
          f"Hour $hour%02d -> Total Trips = $totalTrips"
        } else ""

      println(f"$weekendText%-35s | $weekdayText%-35s")
    }

    if (weekendSorted.nonEmpty) {
      val ((_, weekendPeakHour), weekendPeakTrips) = weekendSorted.head
      println(f"\nTop Weekend Hour -> Hour $weekendPeakHour%02d with Total Trips = $weekendPeakTrips")
    }

    if (weekdaySorted.nonEmpty) {
      val ((_, weekdayPeakHour), weekdayPeakTrips) = weekdaySorted.head
      println(f"Top Weekday Hour -> Hour $weekdayPeakHour%02d with Total Trips = $weekdayPeakTrips")
    }

     // ============================================================
    // DAY-OF-WEEK + HOUR DEMAND
    // ============================================================

    // ============================================================
    // Transformation 8
    // Map each record to ((dayOfWeek, hour), totalTrips)
    // ============================================================
    println("\n==================== Transformation 8 ====================")
    println("Map each record to ((dayOfWeek, hour), totalTrips)")

    val dayHourPairs = baseRDD.map { record =>
      val totalTrips = record.classicTrips + record.electricTrips
      ((record.dayOfWeek, record.hour), totalTrips)
    }

    // ============================================================
    // Transformation 9
    // Aggregate demand by day of week and hour
    // ============================================================
    println("\n==================== Transformation 9 ====================")
    println("Aggregate demand by day of week and hour using reduceByKey")

    val dayHourDemand = dayHourPairs.reduceByKey(_ + _)

    // ============================================================
    // Transformation 10
    // Sort by day, then by demand descending within each day
    // ============================================================
    println("\n==================== Transformation 10 ===================")
    println("Sort day-of-week and hour demand by day, then by demand descending within each day")

    val dayHourDemandSorted = dayHourDemand.sortBy {
      case ((day, hour), totalTrips) => (day, -totalTrips, hour)
    }

    // ============================================================
    // Action 4
    // Show top 5 ranked hours within each day
    // ============================================================
    println("\n======================= Action 4 =========================")
    println("Collect day-of-week and hour demand, then display the top 5 ranked hours within each day")

    val dayHourResults = dayHourDemandSorted.collect().toList
    val topHoursPerDay = 5

    println("Action 4 Output -> Top 5 ranked hours within each day:")
    println()

    val groupedByDay = dayHourResults.groupBy { case ((day, _), _) => day }

    (1 to 7).foreach { day =>
      println(s"${dayName(day)}:")
      groupedByDay.getOrElse(day, List()).take(topHoursPerDay).foreach {
        case ((_, hour), totalTrips) =>
          println(f"  Hour $hour%02d -> Total Trips = $totalTrips")
      }
      println()
    }

    // ============================================================
    // Transformation 11
    // Aggregate total demand by day of week
    // ============================================================
    println("\n==================== Transformation 11 ===================")
    println("Aggregate total demand for each day of the week")

    val dayTotalDemand = baseRDD.map { record =>
      val totalTrips = record.classicTrips + record.electricTrips
      (record.dayOfWeek, totalTrips)
    }.reduceByKey(_ + _)

    // ============================================================
    // Transformation 12
    // Sort days by total demand descending
    // ============================================================
    println("\n==================== Transformation 12 ===================")
    println("Sort days of the week from highest to lowest total demand")

    val dayTotalDemandSorted = dayTotalDemand.sortBy(
      { case (_, totalTrips) => totalTrips },
      ascending = false
    )

    // ============================================================
    // Action 5
    // Show top 3 highest-demand days and the peak hour in each
    // ============================================================
    println("\n======================= Action 5 =========================")
    println("Take the top 3 highest-demand days and show the peak hour in each day")

    val top3Days = dayTotalDemandSorted.take(3)

    println("Action 5 Output -> Top 3 highest-demand days and their peak hour:")
    top3Days.zipWithIndex.foreach { case ((day, totalTrips), index) =>
      val peakHourForDay = groupedByDay.getOrElse(day, List()).headOption

      peakHourForDay match {
        case Some(((_, hour), peakTrips)) =>
          println(
            f"${index + 1}) ${dayName(day)}%-10s -> Total Daily Trips = $totalTrips%-8d | Peak Hour = $hour%02d | Trips = $peakTrips"
          )
        case None =>
          println(
            f"${index + 1}) ${dayName(day)}%-10s -> Total Daily Trips = $totalTrips%-8d | Peak Hour = N/A"
          )
      }
    }

    spark.stop()
  }
}