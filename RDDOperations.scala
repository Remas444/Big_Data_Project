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

    // ============================================================
    // TIME (HOUR) DEMAND ANALYSIS
    // ============================================================

    // ============================================================
    // Transformation 1
    // Build hourly demand from baseRDD:
    // map -> reduceByKey -> sortBy
    // ============================================================
    println("\n==================== Transformation 1 ====================")
    println("Build hourly demand by mapping each record to (hour, totalTrips), aggregating by hour, and sorting by total demand")

    val hourlyDemandSorted = baseRDD
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        (record.hour, totalTrips)
      }
      .reduceByKey(_ + _)
      .sortBy({ case (_, totalTrips) => totalTrips }, ascending = false)

    // ============================================================
    // Action 1
    // Show top 10 peak hours and identify the peak hour
    // ============================================================
    println("\n======================= Action 1 =========================")
    println("Take top 10 peak hours by total demand and identify the peak hour")

    val top10Hours = hourlyDemandSorted.take(10)

    println("Action 1 Output -> Top 10 peak hours:")
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
    // Transformation 2
    // Build weekday/weekend hourly demand:
    // map -> reduceByKey
    // ============================================================
    println("\n==================== Transformation 2 ====================")
    println("Build weekday and weekend hourly demand by mapping records to ((isWeekend, hour), totalTrips) and aggregating them")

    val weekendHourDemand = baseRDD
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        ((record.isWeekend, record.hour), totalTrips)
      }
      .reduceByKey(_ + _)

    // ============================================================
    // Action 2
    // Show weekday and weekend top hours side by side
    // ============================================================
    println("\n======================= Action 2 =========================")
    println("Collect weekday and weekend hourly demand, then display the ranked patterns side by side")

    val weekendHourResults = weekendHourDemand.collect().toList

    val weekdaySorted = weekendHourResults
      .filter { case ((isWeekend, _), _) => isWeekend == 0 }
      .sortBy { case ((_, hour), totalTrips) => (-totalTrips, hour) }

    val weekendSorted = weekendHourResults
      .filter { case ((isWeekend, _), _) => isWeekend == 1 }
      .sortBy { case ((_, hour), totalTrips) => (-totalTrips, hour) }

    val maxRows = math.max(weekdaySorted.length, weekendSorted.length)

    println("Action 2 Output -> Weekend vs Weekday ranked hourly demand")
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
    // Transformation 3
    // Build day-of-week hourly demand:
    // map -> reduceByKey -> sortBy
    // ============================================================
    println("\n==================== Transformation 3 ====================")
    println("Build day-of-week and hour demand by mapping records, aggregating by day and hour, then sorting within each day")

    val dayHourDemandSorted = baseRDD
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        ((record.dayOfWeek, record.hour), totalTrips)
      }
      .reduceByKey(_ + _)
      .sortBy { case ((day, hour), totalTrips) => (day, -totalTrips, hour) }

    // ============================================================
    // Action 3
    // Show top 5 ranked hours within each day
    // ============================================================
    println("\n======================= Action 3 =========================")
    println("Collect day-of-week and hour demand, then display the top 5 ranked hours within each day")

    val dayHourResults = dayHourDemandSorted.collect().toList
    val topHoursPerDay = 5

    println("Action 3 Output -> Top 5 ranked hours within each day:")
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
    // Transformation 4
    // Build total demand by day of week:
    // map -> reduceByKey -> sortBy
    // ============================================================
    println("\n==================== Transformation 4 ====================")
    println("Build total demand for each day of the week by mapping records, aggregating by day, and sorting from highest to lowest")

    val dayTotalDemandSorted = baseRDD
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        (record.dayOfWeek, totalTrips)
      }
      .reduceByKey(_ + _)
      .sortBy({ case (_, totalTrips) => totalTrips }, ascending = false)

    // ============================================================
    // Action 4
    // Show top 3 highest-demand days and the peak hour in each
    // ============================================================
    println("\n======================= Action 4 =========================")
    println("Take the top 3 highest-demand days and show the peak hour in each day")

    val top3Days = dayTotalDemandSorted.take(3)

    println("Action 4 Output -> Top 3 highest-demand days and their peak hour:")
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

    // ============================================================
    // MEMBER VS CASUAL BEHAVIOR ANALYSIS
    // ============================================================

    // ============================================================
    // Transformation 5
    // Build rider-type demand summary:
    // map -> aggregateByKey -> map
    // ============================================================
    println("\n==================== Transformation 5 ====================")
    println("Build rider-type demand summary by mapping records, aggregating totals, and computing average duration")

    val riderTypeSummary = baseRDD
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        (
          record.userType.toLowerCase,
          (
            totalTrips,                          // total trips
            record.classicTrips,                // classic trips
            record.electricTrips,               // electric trips
            record.avgDurationMin * totalTrips, // weighted duration sum
            totalTrips                          // trip count for averaging
          )
        )
      }
      .aggregateByKey((0, 0, 0, 0.0, 0))(
        (acc, value) => (
          acc._1 + value._1,
          acc._2 + value._2,
          acc._3 + value._3,
          acc._4 + value._4,
          acc._5 + value._5
        ),
        (a, b) => (
          a._1 + b._1,
          a._2 + b._2,
          a._3 + b._3,
          a._4 + b._4,
          a._5 + b._5
        )
      )
      .map { case (userType, (totalTrips, classicTrips, electricTrips, weightedDurationSum, tripCount)) =>
        val avgDuration =
          if (tripCount == 0) 0.0 else weightedDurationSum / tripCount.toDouble

        (userType, (totalTrips, classicTrips, electricTrips, avgDuration))
      }

    // ============================================================
    // Action 5
    // Count rider segments and print demand summary
    // ============================================================
    println("\n======================= Action 5 =========================")
    println("Count rider segments and print total demand, bike-type demand, and average trip duration for each rider type")

    val riderSegmentCount = riderTypeSummary.count()
    println(s"Action 5 Output -> Number of rider segments found = $riderSegmentCount")

    riderTypeSummary.foreach {
      case (userType, (totalTrips, classicTrips, electricTrips, avgDuration)) =>
        println(
          f"$userType%-10s -> Total Trips = $totalTrips%-8d | Classic = $classicTrips%-8d | Electric = $electricTrips%-8d | Avg Duration = $avgDuration%.2f min"
        )
    }

    println("\nInterpretation:")
    println("- Higher total trips indicate which rider segment contributes more overall demand.")
    println("- Higher electric trips suggest stronger dependence on electric bikes.")
    println("- Longer average duration may indicate more leisure-oriented usage.")

    // ============================================================
    // Transformation 7
    // Build weekend vs weekday demand by rider type:
    // map -> aggregateByKey -> map -> sortByKey
    // ============================================================
    println("\n==================== Transformation 6 ====================")
    println("Build weekend and weekday demand by rider type by mapping trip totals, aggregating them, and sorting by weekend demand share")

    val weekendWeekdayByRider = baseRDD
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips

        val weekendTrips =
          if (record.isWeekend == 1) totalTrips else 0

        val weekdayTrips =
          if (record.isWeekend == 0) totalTrips else 0

        (record.userType.toLowerCase, (weekendTrips, weekdayTrips))
      }
      .aggregateByKey((0, 0))(
        (acc, value) => (acc._1 + value._1, acc._2 + value._2),
        (a, b) => (a._1 + b._1, a._2 + b._2)
      )
      .map { case (userType, (weekendTrips, weekdayTrips)) =>
        val total = weekendTrips + weekdayTrips
        val weekendShare =
          if (total == 0) 0.0 else weekendTrips.toDouble / total.toDouble

        (weekendShare, (userType, weekendTrips, weekdayTrips))
      }
      .sortByKey(ascending = false)

    // ============================================================
    // Action 7
    // Identify which rider type is more weekend-oriented
    // ============================================================
    println("\n======================= Action 6 =========================")
    println("Use take(1) to identify the rider type with the highest weekend demand share")

    val mostWeekendOriented = weekendWeekdayByRider.take(1)

    println("Action 7 Output -> Weekend vs weekday demand by rider type:")
    weekendWeekdayByRider.foreach {
      case (weekendShare, (userType, weekendTrips, weekdayTrips)) =>
        println(
          f"$userType%-10s -> Weekend Trips = $weekendTrips%-8d | Weekday Trips = $weekdayTrips%-8d | Weekend Share = ${weekendShare * 100}%.2f%%"
        )
    }

    if (mostWeekendOriented.nonEmpty) {
      val (weekendShare, (userType, _, _)) = mostWeekendOriented(0)
      println(
        f"\nMost Weekend-Oriented Rider Type -> $userType with Weekend Share = ${weekendShare * 100}%.2f%%"
      )
    }

    println("\nInterpretation:")
    println("- A higher weekend share suggests more leisure or flexible-use behavior.")
    println("- A lower weekend share suggests stronger weekday dependence, often linked to routine travel or commuting.")


    // ============================================================
    // Transformation 8
    // Build electric-bike preference by rider type:
    // map -> aggregateByKey -> map
    // ============================================================
    println("\n==================== Transformation 7 ====================")
    println("Build electric-bike preference by rider type by aggregating electric and total trips, then computing electric usage share")

    val electricPreferenceByRider = baseRDD
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        (record.userType.toLowerCase, (record.electricTrips, totalTrips))
      }
      .aggregateByKey((0, 0))(
        (acc, value) => (acc._1 + value._1, acc._2 + value._2),
        (a, b) => (a._1 + b._1, a._2 + b._2)
      )
      .map { case (userType, (electricTrips, totalTrips)) =>
        val electricShare =
          if (totalTrips == 0) 0.0 else electricTrips.toDouble / totalTrips.toDouble

        (userType, electricTrips, totalTrips, electricShare)
      }

    // ============================================================
    // Action 8
    // Rank rider types by electric-bike usage share
    // ============================================================
    println("\n======================= Action 7 =========================")
    println("Use takeOrdered to rank rider types by electric-bike usage share")

    val electricPreferenceRanking = electricPreferenceByRider
      .map { case (userType, electricTrips, totalTrips, electricShare) =>
        (-electricShare, userType, electricTrips, totalTrips)
      }
      .takeOrdered(2)

    println("Action 8 Output -> Rider types ranked by electric-bike usage share:")
    electricPreferenceRanking.zipWithIndex.foreach {
      case ((negativeShare, userType, electricTrips, totalTrips), index) =>
        val electricShare = -negativeShare * 100.0
        println(
          f"${index + 1}) $userType%-10s -> Electric Trips = $electricTrips%-8d | Total Trips = $totalTrips%-8d | Electric Share = $electricShare%.2f%%"
        )
    }

    println("\nInterpretation:")
    println("- Higher electric-bike share means that rider group relies more on electric bikes.")
    println("- This is useful for understanding resource preference and supporting bike-type allocation decisions.")

    spark.stop()
  }
}