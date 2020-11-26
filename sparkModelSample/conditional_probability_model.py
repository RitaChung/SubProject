from optparse import OptionParser
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, lit

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-u", "--s3-endpoint", dest='s3_endpoint', help='S3 Endpoint URL', default=None)
    parser.add_option("-i", '--input', dest='input', help='Path to Input', default=None)
    parser.add_option("-o", '--output', dest='output', help='Path to Output', default=None)

    (option, args) = parser.parse_args()
    spark = SparkSession.builder.appName('conditional probability model').getOrCreate()
    inputSchema = StructType([StructField('id', StringType(), True), StructField('nationality', StringType(), True),
                             StructField('age', StringType(), True), StructField('gender', StringType(), True),
                             StructField('class', StringType(), True)])
    outputSchema = StructType([StructField('Nationality', StringType(), True), StructField('Age', StringType(), True),
                               StructField('Gender', StringType(), True), StructField('Class', StringType(), True),
                               StructField('Risk', DoubleType(), True)])

    df = spark.read.csv(option.input, header=True, schema=inputSchema)

    all_num = df.groupBy('nationality', 'age', 'gender', 'class').count().withColumnRenamed('count', 'all_case')
    nat_num = df.groupBy('nationality').count().withColumnRenamed('count', 'nat_case')
    gender_num = df.groupBy('gender').count().withColumnRenamed('count', 'gender_case')
    age_num = df.groupBy('age').count().withColumnRenamed('count', 'age_case')
    class_num = df.groupBy('class').count().withColumnRenamed('count', 'class_case')

    nat_class = df.groupBy('nationality', 'class').count().withColumnRenamed('count', 'nat_class_case')
    gender_class = df.groupBy('gender', 'class').count().withColumnRenamed('count', 'gender_class_case')
    age_class = df.groupBy('age', 'class').count().withColumnRenamed('count', 'age_class_case')

    result = all_num.join(nat_num, on=['nationality'], how='left').join(gender_num, on=['gender'], how='left')\
        .join(age_num, on=['age'], how='left').join(class_num, on=['class'], how='left')\
        .join(nat_class, on=['nationality','class'], how='left').join(gender_num, on=['gender','class'], how='left')\
        .join(age_class, on=['age','class'], how='left')

    result.withColumn('totalNum',lit(df.count())).withColumn('NationalityProb', col('nat_case')/col('totalNum'))\
        .withColumn('GenderProb', col('gender_case')/col('totalNum')).withColumn('AgeProb',col('age_case'/col('totalNum')))\
        .withColumn('ClassProb', col('class_case')/col('totalNum')).withColumn('ClassNationality', col('nat_class_case')/col('nat_case'))\
        .withColumn('ClassGender', col('gender_class_case')/col('gender_case')).withColumn('ClassAgeProb', col('age_class_case')/col('age_case'))\
        .withColumn('risk', col('ClassNationality')*col('ClassGender')*col('ClassAgeProb')*col('NationalityProb')*col('GenderProb')*col('AgeProb')/col('ClassProb'))\
        .select(col('nationality').alias('Nationality'), col('age').alias('Age'), col('gender').alias('Gender'), col('class').alias('Class'), col('risk').alias('Risk'))

    result.write.csv(option.output, header=True, mode='overwrite')
    
