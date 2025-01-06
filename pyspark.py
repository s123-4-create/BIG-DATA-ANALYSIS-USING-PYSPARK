from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("LargeDatasetAnalysis").getOrCreate()

# Load a large dataset into a DataFrame
df = spark.read.csv("path/to/large_dataset.csv", header=True, inferSchema=True)

# Clean and preprocess the data
df_cleaned = df.dropna()  # Removing rows with null values
df_cleaned = df_cleaned.withColumn("new_column", df_cleaned["existing_column"] * 2)  # Example transformation

# Generate summary statistics
df_cleaned.describe().show()

from pyspark.ml.feature import VectorAssembler

# Create new features and assemble them
assembler = VectorAssembler(inputCols=["col1", "col2"], outputCol="features")
df_features = assembler.transform(df_cleaned)

from pyspark.ml.regression import LinearRegression

# Split the data into training and test sets
train_data, test_data = df_features.randomSplit([0.8, 0.2])

# Train a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Evaluate the model's performance on the test set
test_results = lr_model.evaluate(test_data)
print(f"Root Mean Squared Error: {test_results.rootMeanSquaredError}")
