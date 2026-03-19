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

    spark.stop()
  }
}