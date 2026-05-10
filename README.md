# Citi Bike Big Data Analysis

A Big Data project analyzing Citi Bike trip records (October–December 2025) using Apache Spark and Scala. The project covers demand analysis, RDD operations, SQL queries, and machine learning to uncover usage patterns across NYC bike-sharing stations.

---

## Prerequisites

Make sure the following are installed before running the project:

- Java 11 LTS
- Scala 2.12.18
- sbt (latest stable)
- VS Code with the Scala (Metals) extension

---

## Getting Started

**Step 1 – Install the Scala (Metals) extension in VS Code**

Open VS Code, go to Extensions, search for "Scala (Metals)" and install it.

**Step 2 – Verify sbt is installed**

Open a terminal and run:

```bash
sbt --version
```

If you get an error, install sbt from: https://www.scala-sbt.org/download/

**Step 3 – Create a new sbt project**

Follow this short video tutorial to create a new sbt project in VS Code. You can start from 1:56 since you already completed Steps 1 and 2:

https://youtu.be/fcl9dLmWhgo

Once done, the video will have created your project with this folder structure:

```
citibike-bigdata/
├── src/main/scala/
├── build.sbt
└── ...
```

**Step 4 – Download the project Scala files**

Go to the project GitHub repository and download the files:

https://github.com/Remas444/Big_Data_Project

Click the green "Code" button, select "Download ZIP", then extract it.

**Step 5 – Add the Scala files to your project**

From the extracted folder, drag and drop all `.scala` files into your project at:

```
citibike-bigdata/src/main/scala/
```

**Step 6 – Configure build.sbt**

Open `build.sbt` in the project root and replace its contents with the following — copy and paste it exactly, do not modify it:

```scala
ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "citibike-bigdata",
    version := "0.1.0-SNAPSHOT",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "3.5.1",
      "org.apache.spark" %% "spark-mllib" % "3.5.1",
      "org.scalameta" %% "munit" % "1.0.0" % Test
    ),
run / fork := true,

run / javaOptions ++= Seq(
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
  "--add-opens=java.base/java.io=ALL-UNNAMED",
  "--add-opens=java.base/java.net=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED"
)
  )
```

Note: The `javaOptions` flags are required to avoid module access errors when running Spark with Java 11. Do not remove them.

**Step 7 – Download the dataset**

Download the Citi Bike trip data from:

https://citibikenyc.com/system-data

Download the monthly files for October, November, and December 2025. Place all CSV files inside:

```
citibike-bigdata/data/raw/
```

Create the `data/raw/` folder manually if it does not exist.

**Step 8 – Compile the project**

In the VS Code terminal, run:

```bash
sbt compile
```

This will download all dependencies and confirm everything is set up correctly.

---

## Running the Scripts

Each phase of the project is a separate Scala file. To run a script, use:

```bash
sbt run
```

| File | Description |
|------|-------------|
| `DataPreprocessing.scala` | Cleans, integrates, reduces, and transforms the raw CSV data |
| `RDDOperations.scala` | Demand analysis by hour, day, station, and rider type using RDDs |
| `SQLOperations.scala` | Spark SQL queries for usage patterns and trip statistics |
| `MachineLearning.scala` | Hourly demand prediction using Spark MLlib |

---

## Project Structure

```
citibike-bigdata/
├── src/main/scala/
│   ├── DataPreprocessing.scala
│   ├── RDDOperations.scala
│   ├── SQLOperations.scala
│   └── MachineLearning.scala
├── data/
│   └── raw/
├── build.sbt
└── README.md
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| spark-sql | 3.5.1 | Spark SQL and DataFrames |
| spark-mllib | 3.5.1 | Machine learning |
| munit | 1.0.0 | Unit testing |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Metals not loading in VS Code | Run `sbt compile` once from the terminal to trigger the Metals import |
| `java` not found | Ensure Java 11 is installed and added to your system PATH |
| Out of memory errors | Run `sbt -J-Xmx4g run` to increase available memory |
| CSV files not found | Make sure the dataset files are placed inside `data/raw/` |
