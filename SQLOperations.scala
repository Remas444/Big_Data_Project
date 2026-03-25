import org.apache.spark.sql.SparkSession

object SQLOperations {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("SQL Operations")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // Step 1: Load the preprocessed dataset and convert it into a DataFrame
    val tripsDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/processed/transformedData.csv")

    // Step 2: Register the DataFrame as a temporary SQL view
    tripsDF.createOrReplaceTempView("trips")

    // Query 1: Stations with the largest imbalance between outgoing and incoming trips
    // Uses: GROUP BY, SUM aggregation, FULL OUTER JOIN, CTE, ORDER BY, LIMIT

    val stationImbalanceQuery =
      """
        |WITH outgoing AS (
        |  SELECT
        |    start_station_name AS station_name,
        |    SUM(classic_trips + electric_trips) AS outgoing_trips
        |  FROM trips
        |  GROUP BY start_station_name
        |),
        |incoming AS (
        |  SELECT
        |    end_station_name AS station_name,
        |    SUM(classic_trips + electric_trips) AS incoming_trips
        |  FROM trips
        |  GROUP BY end_station_name
        |)
        |SELECT
        |  COALESCE(o.station_name, i.station_name) AS station_name,
        |  COALESCE(o.outgoing_trips, 0) AS outgoing_trips,
        |  COALESCE(i.incoming_trips, 0) AS incoming_trips,
        |  COALESCE(o.outgoing_trips, 0) - COALESCE(i.incoming_trips, 0) AS net_difference,
        |  ABS(COALESCE(o.outgoing_trips, 0) - COALESCE(i.incoming_trips, 0)) AS absolute_imbalance
        |FROM outgoing o
        |FULL OUTER JOIN incoming i
        |  ON o.station_name = i.station_name
        |ORDER BY absolute_imbalance DESC
        |LIMIT 10
        |""".stripMargin

    val stationImbalanceResult = spark.sql(stationImbalanceQuery)

    println()
    println("============================================================")
    println("Stations with the Largest Imbalance Between Outgoing and Incoming Trips")
    println("============================================================")
    stationImbalanceResult.show(10, false)

    // Query 2: Top used stations with electric and classic bike percentages
    // Uses: GROUP BY, SUM aggregation, ORDER BY, LIMIT

    val bikePreferenceQuery =
      """
        |SELECT
        |  start_station_name,
        |  SUM(electric_trips) AS total_electric_trips,
        |  SUM(classic_trips) AS total_classic_trips,
        |  SUM(classic_trips + electric_trips) AS total_trips,
        |  ROUND(
        |    SUM(electric_trips) * 100.0 / SUM(classic_trips + electric_trips),
        |    2
        |  ) AS electric_share_percent,
        |  ROUND(
        |    SUM(classic_trips) * 100.0 / SUM(classic_trips + electric_trips),
        |    2
        |  ) AS classic_share_percent
        |FROM trips
        |GROUP BY start_station_name
        |HAVING SUM(classic_trips + electric_trips) > 0
        |ORDER BY total_trips DESC
        |LIMIT 10
        |""".stripMargin

    val bikePreferenceResult = spark.sql(bikePreferenceQuery)

    println()
    println("============================================================")
    println("Top Used Stations with Electric and Classic Bike Percentages")
    println("============================================================")
    bikePreferenceResult.show(10, false)

    // Query 3: Hours with the most stable and least stable demand across the week
    // Uses: CTE, GROUP BY, SUM aggregation, MIN, MAX, AVG, ORDER BY, LIMIT

    val stableHoursQuery =
      """
        |WITH hourly_day_demand AS (
        |  SELECT
        |    hour,
        |    day_of_week,
        |    SUM(classic_trips + electric_trips) AS total_demand
        |  FROM trips
        |  GROUP BY hour, day_of_week
        |)
        |SELECT
        |  hour,
        |  MIN(total_demand) AS min_daily_demand,
        |  MAX(total_demand) AS max_daily_demand,
        |  ROUND(AVG(total_demand), 2) AS avg_daily_demand,
        |  MAX(total_demand) - MIN(total_demand) AS demand_range
        |FROM hourly_day_demand
        |GROUP BY hour
        |ORDER BY demand_range ASC, avg_daily_demand DESC
        |LIMIT 10
        |""".stripMargin

    val stableHoursResult = spark.sql(stableHoursQuery)

    println()
    println("============================================================")
    println("Most Stable Demand Hours Across the Week")
    println("============================================================")
    stableHoursResult.show(10, false)


    val unstableHoursQuery =
      """
        |WITH hourly_day_demand AS (
        |  SELECT
        |    hour,
        |    day_of_week,
        |    SUM(classic_trips + electric_trips) AS total_demand
        |  FROM trips
        |  GROUP BY hour, day_of_week
        |)
        |SELECT
        |  hour,
        |  MIN(total_demand) AS min_daily_demand,
        |  MAX(total_demand) AS max_daily_demand,
        |  ROUND(AVG(total_demand), 2) AS avg_daily_demand,
        |  MAX(total_demand) - MIN(total_demand) AS demand_range
        |FROM hourly_day_demand
        |GROUP BY hour
        |ORDER BY demand_range DESC, avg_daily_demand DESC
        |LIMIT 10
        |""".stripMargin

    val unstableHoursResult = spark.sql(unstableHoursQuery)

    println()
    println("============================================================")
    println("Least Stable Demand Hours Across the Week")
    println("============================================================")
    unstableHoursResult.show(10, false)

    spark.stop()
  }
}
