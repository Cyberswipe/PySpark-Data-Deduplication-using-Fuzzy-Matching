import logging
from fuzzywuzzy import fuzz
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, first, count
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import sum as sparkSum
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, dense_rank
from pyspark import SparkConf

def read_csv_spark(spark, input_file):
    '''
    Method to read CSV
    '''
    return spark.read.csv(input_file, header=True, inferSchema=True)

def load_data_flag(spark, file1, file2):
    '''
        Reads files to Dataframe and populates flag for source indication
    '''
    try:
        logger.info("Reading file contents into csv file")
        data_source_1 = read_csv_spark(spark, file1)
        data_source_1 = data_source_1.withColumn("File_1_exists", lit(True)).withColumn("File_2_exists", lit(False))
        logger.info(f"First file read from path: {file1}")
        data_source_2 = read_csv_spark(spark, file2)
        data_source_2 = data_source_2.withColumn("File_1_exists", lit(False)).withColumn("File_2_exists", lit(True))
        logger.info(f"Second file read from path: {file2}")
    except:
        logger.info("File not found!")
    return data_source_1.union(data_source_2)

def grouping_index(data_frame):
    '''
        Groups data based on coloumns and assings unique group index to that block
    '''
    logger.info("Grouping the data....")
    window_spec = Window.orderBy("Org_Adr","Org_City","Org_Zip")
    return data_frame.withColumn("Group_Index", dense_rank().over(window_spec))

def add_entry_count(data_frame):
    '''
        Counts each entry including duplicates
    '''
    #Add a unique count for each entry (Distinct entries) based on group_index and Org_name
    logger.info("Counting each entry...")
    window_spec = Window.partitionBy("Group_Index", "Org_Name")
    return data_frame.withColumn("Entry_Count", count("*").over(window_spec))

# User defined func to apply on columns
@udf(IntegerType())
def calculate_fuzzy_score_udf(string1, string2):
    '''
        Fuzzy logic to calculate scores between two strings
    '''
    return fuzz.ratio(string1, string2)

def compute_fuzzy_scores(data_frame):
    '''
        Append scores using a reference coloumn
    '''
    logger.info("Dropping exact duplicates...")
    custom_df = data_frame.dropDuplicates(["Org_Name", "Org_Adr"])
    window_spec = Window().partitionBy("Group_Index")
    logger.info("Computing Fuzzy score to estimate closeness of data with reference...")
    df_with_reference = custom_df.withColumn("Reference_Org_Name", first("Org_Name").over(window_spec))
    df_with_fuzzy_score = df_with_reference.withColumn("Fuzzy_Score", calculate_fuzzy_score_udf(col("Reference_Org_Name"), col("Org_Name")))
    return df_with_fuzzy_score

def update_counts(data_frame):
    '''
        Update the counts for entries that have duplicates
    '''
    condition = (col("Fuzzy_Score") > 60) & (col("Fuzzy_Score") != 100)
    logger.info("Appending duplicate entry counts...")
    window_spec = Window.partitionBy("Group_Index")
    data_frame = data_frame.withColumn("Sum_Count", sparkSum(when(condition, col("Entry_Count")).otherwise(0)).over(window_spec))
    data_frame = data_frame.withColumn("Entry_Count", when(col("Fuzzy_Score") == 100, col("Entry_Count") + col("Sum_Count")).otherwise(col("Entry_Count")))
    return data_frame.drop("Sum_Count")

def deduplicate_and_reindex(data_frame):
    '''
        Deduplicate and assign UniqueID to each restaurant
    '''
    deduped_df = data_frame.filter((col("Fuzzy_Score") == 100) | (col("Fuzzy_Score") < 60))
    logger.info("Deduplicate and reindex with UniqueID")
    deduped_df = deduped_df.withColumn("Org_ID", monotonically_increasing_id()+1)
    return deduped_df.select("Org_ID","Org_Name","Org_Adr","Org_City","Org_Zip","File_1_exists","File_2_exists","Entry_count")

def write_to_csv(data_frame, output_path):
    '''
        Write dataframe to a single csv. File name based on spark parition
    '''
    try:
      data_frame.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
      logger.info(f"written dataframe to output CSV path: {output_path}'")
    except:
      logger.info(f"File not written!")

def main():
    '''
        Main method which includes logging and initialzes Spark context
    '''
    conf = SparkConf().set("spark.logConf", "true")
    spark = SparkSession.builder.appName("Opendata").config(conf=conf).getOrCreate()

    logger.info("<---Spark session started--->")

    #Input your file_name with path below
    input_file1, input_file2 = "", ""
    output_path = "/content/sample_data/output"
    raw_data_frame = load_data_flag(spark, input_file1, input_file2)

    Name, Address, City, Zip, Flag_1, Flag_2 = "Org_Name", "Org_Adr", "Org_City", "Org_Zip","File_1_exists","File_2_exists"

    #Standardizing coloumn names
    custom_column_names = [Name, Address, City, Zip, Flag_1, Flag_2]
    logger.info("Coloumn Name Standardized")

    data_df = raw_data_frame.toDF(*custom_column_names)
    logger.info("Dataframe contains harmonized column names")

    length_of_df = data_df.count()
    logger.info(f"Total dataset size --> {length_of_df}")

    grouped_data_df = grouping_index(data_df)
    logger.info(f"Data grouped with group index {Address}, {Zip}, {City}, {Zip}...")

    output_df_counts = add_entry_count(grouped_data_df)
    logger.info("Counted and added to Dataframe..")

    output_df = compute_fuzzy_scores(output_df_counts)
    logger.info("Similarity Score computed and appended...")

    output_df_refined = update_counts(output_df)
    logger.info("Entry counts appended in col: Entry_count")

    result_df = deduplicate_and_reindex(output_df_refined)
    result_df_count = result_df.count()
    logger.info(f"Dataframe filtered --> New Dataframe Length after harmonizing: {result_df_count}")

    write_to_csv(result_df, output_path)
    logger.info("...End of pipeline...")

    logger.info("....<Closing spark session>....")
    spark.stop()
    logger.info("----Spark session ended----")

if __name__ == "__main__":

    log_level = "INFO"
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
