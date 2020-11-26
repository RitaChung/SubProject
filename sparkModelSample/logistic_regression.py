from optparse import OptionParser
from pyspark import *
from pyspark.sql import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import LogisticRegression


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-u", "--s3-endpoint", dest='s3_endpoint', help='S3 Endpoint URL', default = None)
    parser.add_option("-i", '--input', dest='input', help='Path to Input', default=None)
    parser.add_option("-o", '--output', dest='output', help='Path to Output', default=None)

    (option, args) = parser.parse_args()
    spark = SparkSession.builder.appName('logistic regression model').getOrCreate()

    sql = SQLContext(spark.sparkContext)
    inputSchema = StructType([StructField('id', StringType(), True),
                              StructField('name', StringType(), True),
                              StructField('gender', StringType(), True),
                              StructField('class', StringType(), True),
                              StructField('current_age', IntegerType(), True),
                              StructField('current_age_group', StringType(), True),
                              StructField('marital_status', StringType(), True),
                              StructField('job', StringType(), True),
                              StructField('nationality', StringType(), True),
                              StructField('record', StringType(), True)])
    outputSchema = StructType([StructField('id', StringType(), True), StructField('risk', StringType(), True)])

    df = spark.read.csv(option.input, header=False, schema=inputSchema)

    #convert format
    categoricalColumns = ['gender', 'class', 'marital_status', 'job', 'nationality', 'current_age_group']
    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + 'classVec'])
        stages += [stringIndexer, encoder]

    #convert label
    label_stringIndex = StringIndexer(inputCol='record', outputCol='label')
    stages += [label_stringIndex]

    #conert features
    assemblerInputInputs = [c+'classVec' for c in categoricalColumns]
    assembler = VectorAssembler(inputCols = assemblerInputInputs, outputCol='features')

    #build the pipeline of preprocess and apply to the raw data
    partialPipeline = Pipeline().setStages(stages)
    pipelineModel = partialPipeline.fit(df)
    prepedDataDF = pipelineModel.transform(df)

    #build the logistic regression
    lrModel = LogisticRegression().fit(prepedDataDF)
    lr_prediction = lrModel.transform(prepedDataDF)

    def extract_prob(array):
        return str(array[1])

    extract_prob_udf = udf(lambda array: extract_prob(array), StringType())
    spark.udf.register("extract_prob", extract_prob_udf)

    model_risk = lr_prediction.select("id", "probability").withColumn('risk', extract_prob_udf('probability'))
    model_risk = model_risk.select('id', 'risk')
    model_risk.write.csv(option, header=True, mode='overwrite')


