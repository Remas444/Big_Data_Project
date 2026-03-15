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

   println("\nInterpretation:")
   println("- Higher trip counts in certain hours indicate periods of stronger bike demand.")
   println("- Lower counts suggest quieter hours with less activity.")

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
    // Show top 5 weekday and weekend hours side by side
    // ============================================================
    println("\n======================= Action 2 =========================")
    println("Collect weekday and weekend hourly demand, then display the top 5 ranked hours side by side")

    val weekendHourResults = weekendHourDemand.collect().toList

    val weekdaySorted = weekendHourResults
      .filter { case ((isWeekend, _), _) => isWeekend == 0 }
      .sortBy { case ((_, hour), totalTrips) => (-totalTrips, hour) }

    val weekendSorted = weekendHourResults
      .filter { case ((isWeekend, _), _) => isWeekend == 1 }
      .sortBy { case ((_, hour), totalTrips) => (-totalTrips, hour) }

    val topHoursPerType = 5
    val topWeekday = weekdaySorted.take(topHoursPerType)
    val topWeekend = weekendSorted.take(topHoursPerType)
    val maxRows = math.max(topWeekday.length, topWeekend.length)

    println("Action 2 Output -> Top 5 weekend vs weekday ranked hourly demand")
    println()
    println(f"${"Weekend"}%-35s | ${"Weekday"}%-35s")
    println("-" * 75)

    for (i <- 0 until maxRows) {
      val weekendText =
        if (i < topWeekend.length) {
          val ((_, hour), totalTrips) = topWeekend(i)
          f"Hour $hour%02d -> Total Trips = $totalTrips"
        } else ""

      val weekdayText =
        if (i < topWeekday.length) {
          val ((_, hour), totalTrips) = topWeekday(i)
          f"Hour $hour%02d -> Total Trips = $totalTrips"
        } else ""

      println(f"$weekendText%-35s | $weekdayText%-35s")
    }

    if (topWeekend.nonEmpty) {
      val ((_, weekendPeakHour), weekendPeakTrips) = topWeekend.head
      println(f"\nTop Weekend Hour -> Hour $weekendPeakHour%02d with Total Trips = $weekendPeakTrips")
    }

    if (topWeekday.nonEmpty) {
      val ((_, weekdayPeakHour), weekdayPeakTrips) = topWeekday.head
      println(f"Top Weekday Hour -> Hour $weekdayPeakHour%02d with Total Trips = $weekdayPeakTrips")
    }

    println("\nInterpretation:")
    println("- Higher weekday demand during specific hours may reflect routine daily patterns.")
    println("- Higher weekend demand around midday may indicate more flexible or leisure-related activity.")
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
    // Show top 5 ranked hours within each day in a compact table
    // ============================================================
    println("\n======================= Action 3 =========================")
    println("Collect day-of-week and hour demand, then display the top 5 ranked hours within each day")

    val dayHourResults = dayHourDemandSorted.collect().toList
    val topHoursPerDay = 5

    val groupedByDay: Map[Int, List[((Int, Int), Int)]] =
      dayHourResults.groupBy { case ((day, _), _) => day }

    println("Action 3 Output -> Top 5 ranked hours within each day:")
    println()

    println(f"${"Rank"}%-4s | ${"Sun"}%-12s | ${"Mon"}%-12s | ${"Tue"}%-12s | ${"Wed"}%-12s | ${"Thu"}%-12s | ${"Fri"}%-12s | ${"Sat"}%-12s")
    println("-" * 110)

    for (i <- 0 until topHoursPerDay) {
      val row = (1 to 7).map { day =>
        val list: List[((Int, Int), Int)] =
          groupedByDay.getOrElse(day, List.empty[((Int, Int), Int)])

        if (i < list.length) {
          val ((_, hour), totalTrips) = list(i)
          f"$hour%02d($totalTrips)"
        } else {
          ""
        }
      }

      println(
        f"${i + 1}%-4d | ${row(0)}%-12s | ${row(1)}%-12s | ${row(2)}%-12s | ${row(3)}%-12s | ${row(4)}%-12s | ${row(5)}%-12s | ${row(6)}%-12s"
      )
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
    println("\nInterpretation:")
    println("- Higher total trips on certain days indicate when bike demand is most concentrated during the week.")
    println("- Consistent peak hours across multiple days suggest stable daily usage patterns.")

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
    // Transformation 6
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
    // Action 6
    // Identify which rider type is more weekend-oriented
    // ============================================================
    println("\n======================= Action 6 =========================")
    println("Use take(1) to identify the rider type with the highest weekend demand share")

    val mostWeekendOriented = weekendWeekdayByRider.take(1)

    println("Action 6 Output -> Weekend vs weekday demand by rider type:")
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
    // Transformation 7
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
    // Action 7
    // Rank rider types by electric-bike usage share
    // ============================================================
    println("\n======================= Action 7 =========================")
    println("Use takeOrdered to rank rider types by electric-bike usage share")

    val electricPreferenceRanking = electricPreferenceByRider
      .map { case (userType, electricTrips, totalTrips, electricShare) =>
        (-electricShare, userType, electricTrips, totalTrips)
      }
      .takeOrdered(2)

    println("Action 7 Output -> Rider types ranked by electric-bike usage share:")
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

        // ============================================================
    // STATION USAGE ANALYSIS
    // ============================================================

    // ============================================================
    // Transformation 8
    // Build start-station demand:
    // filter -> map -> reduceByKey -> sortBy
    // ============================================================
    println("\n==================== Transformation 8 ====================")
    println("Build start-station demand by filtering valid station names, mapping records to (startStation, totalTrips), aggregating them, and sorting by total demand")

    val startStationDemandSorted = baseRDD
      .filter { record =>
        record.startStation != null &&
        record.startStation.nonEmpty &&
        record.startStation.toLowerCase != "unknown"
      }
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        (record.startStation, totalTrips)
      }
      .reduceByKey(_ + _)
      .sortBy({ case (_, totalTrips) => totalTrips }, ascending = false)

    // ============================================================
    // Action 8
    // Show top 10 most-used start stations
    // ============================================================
    println("\n======================= Action 8 =========================")
    println("Take the top 10 most-used start stations by total trip volume")

    val top10StartStations = startStationDemandSorted.take(10)

    println("Action 8 Output -> Top 10 most-used start stations:")
    top10StartStations.zipWithIndex.foreach {
      case ((station, totalTrips), index) =>
        println(f"${index + 1}) $station%-40s -> Total Trips = $totalTrips")
    }

    if (top10StartStations.nonEmpty) {
      val (station, totalTrips) = top10StartStations.head
      println(f"\nMost Used Start Station -> $station%-40s with Total Trips = $totalTrips")
    }

    println("\nInterpretation:")
    println("- Higher trip counts at certain start stations indicate stronger origin demand at those locations.")
    println("- These stations may represent major departure hubs that require greater bike availability.")


    // ============================================================
    // Transformation 9
    // Build end-station demand:
    // filter -> map -> reduceByKey -> sortBy
    // ============================================================
    println("\n==================== Transformation 9 ====================")
    println("Build end-station demand by filtering valid station names, mapping records to (endStation, totalTrips), aggregating them, and sorting by total demand")

    val endStationDemandSorted = baseRDD
      .filter { record =>
        record.endStation != null &&
        record.endStation.nonEmpty &&
        record.endStation.toLowerCase != "unknown"
      }
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        (record.endStation, totalTrips)
      }
      .reduceByKey(_ + _)
      .sortBy({ case (_, totalTrips) => totalTrips }, ascending = false)

    // ============================================================
    // Action 9
    // Show top 10 most-used end stations
    // ============================================================
    println("\n======================= Action 9 =========================")
    println("Take the top 10 most-used end stations by total trip volume")

    val top10EndStations = endStationDemandSorted.take(10)

    println("Action 9 Output -> Top 10 most-used end stations:")
    top10EndStations.zipWithIndex.foreach {
      case ((station, totalTrips), index) =>
        println(f"${index + 1}) $station%-40s -> Total Trips = $totalTrips")
    }

    if (top10EndStations.nonEmpty) {
      val (station, totalTrips) = top10EndStations.head
      println(f"\nMost Used End Station -> $station%-40s with Total Trips = $totalTrips")
    }

    println("\nInterpretation:")
    println("- Higher trip counts at certain end stations indicate stronger destination demand at those locations.")
    println("- These stations may require more frequent bike rebalancing and dock management.")


    // ============================================================
    // Transformation 10
    // Build route demand and station coverage:
    // filter -> map -> reduceByKey -> sortBy
    // and flatMap -> distinct
    // ============================================================
    println("\n==================== Transformation 10 ====================")
    println("Build route demand by mapping valid start-end station pairs, aggregating total trips per route, sorting by total demand, and extracting distinct station names")

    val routeDemandSorted = baseRDD
      .filter { record =>
        record.startStation != null &&
        record.endStation != null &&
        record.startStation.nonEmpty &&
        record.endStation.nonEmpty &&
        record.startStation.toLowerCase != "unknown" &&
        record.endStation.toLowerCase != "unknown"
      }
      .map { record =>
        val totalTrips = record.classicTrips + record.electricTrips
        ((record.startStation, record.endStation), totalTrips)
      }
      .reduceByKey(_ + _)
      .sortBy({ case (_, totalTrips) => totalTrips }, ascending = false)

    val distinctStations = baseRDD
      .flatMap { record => Seq(record.startStation, record.endStation) }
      .filter(station => station != null && station.nonEmpty && station.toLowerCase != "unknown")
      .distinct()

    // ============================================================
    // Action 10
    // Show top 10 most frequent routes and count distinct stations
    // ============================================================
    println("\n======================= Action 10 =========================")
    println("Take the top 10 most frequent routes and count the total number of distinct stations")

    val top10Routes = routeDemandSorted.take(10)
    val totalDistinctStations = distinctStations.count()

    println("Action 10 Output -> Top 10 most frequent routes:")
    top10Routes.zipWithIndex.foreach {
      case (((startStation, endStation), totalTrips), index) =>
        println(f"${index + 1}) $startStation%-30s -> $endStation%-30s | Total Trips = $totalTrips")
    }

    println(s"\nTotal Distinct Stations = $totalDistinctStations")

    if (top10Routes.nonEmpty) {
      val ((startStation, endStation), totalTrips) = top10Routes.head
      println(f"Most Frequent Route -> $startStation%-30s -> $endStation%-30s | Total Trips = $totalTrips")
    }

    println("\nInterpretation:")
    println("- Frequently repeated routes reveal strong mobility links between specific station pairs.")
    println("- The number of distinct stations reflects the overall spatial coverage of the bike-sharing system.")
    
    spark.stop()
  }
}
