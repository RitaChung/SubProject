from optparse import OptionParser
from graphframes import GraphFrame
from pyspark.sql import SparkSession
from pyspark.sql.function import col, sum, coalesce
from pyspark.sql.type import StructType, StructField, StringType, DoubleType, IntegerType

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-u", "--s3-endpoint", dest='s3_endpoint', help='S3 Endpoint URL', default=None)
    parser.add_option("-e", '--input-edge', dest='input_edge', help='Path to Input Edges', default=None)
    parser.add_option("-v", '--input-vertex', dest='input_vertex', help='Path to Input Vertex', default=None)
    parser.add_option('-o', '--output', dest='output', help='Path to output', default=None)

    (option, args) = parser.parse_args()
    spark = SparkSession.builder.appName('connected component').getOrCreate()
    spark.sparkContext.addPyFile("graphframes.zip")

    edges_input_schema = StructType([StructField('src', StringType(), True), 
                                     StructField('dst', StringType(), True), 
                                     StructField('prob', DoubleType(), True)])
    vertices_input_schema = StructType([StructField('id', StringType(), True), 
                                        StructField('label', IntegerType(), True)])

    edges = spark.read.csv(option.input_edge, header=True, schema=edges_input_schema)
    vertices = spark.read.csv(option.input_vertex, header=True, schema=vertices_input_schema)
    graph = GraphFrame(vertices, edges)

    # 1 node connection
    path1 = "(n1)-[e1]->(n2)"
    result1 = graph.find(path1).filter(col("n2.label") != 0).withColumn('path_risk', col('e1.prob'))
    result1 = result1.groupBy('n1.id').agg(sum('path_risk').alias('n1_path_risk')).withColumnRenamed('id', 'n1_id')

    # 2 node connection
    path2 = "(n1)-[e1]->(n2); (n2)-[e2]->(n3)"
    result2 = graph.find(path2).filter("n1.id != n3.id").withColumn('label_tag', col('n2.label')+col('n3.label'))\
        .filter(col('label_tag') != 0).withColumn('path_risk', col('e1.prob')*col('e2.prob'))
    result2 = result2.groupBy('n1.id').agg(sum('path_risk').alias('n2_path_risk')).withColumnRenamed('id', 'n2_id')

    # 3 node connection
    path3 = "(n1)-[e1]->(n2); (n2)-[e2]->(n3); (n3)-[e3]->(n4)"
    result3 = graph.find(path3).filter("n1.id != n4.id").withColumn('label_tag', col('n2.label')+col('n3.label')+col('n4.label'))\
        .filter(col('label_tag') != 0).withColumn('path_risk', col('e1.prob')*col('e2.prob')*col('e3.prob'))
    result3 = result3.groupBy('n1.id').agg(sum('path_risk').alias('n3_path_risk')).withColumnRenamed('id', 'n3_id')

    res = result1.join(result2, result1.n1_id == result2.n2_id, how='full_outer')\
        .join(result3, result2.n2_id == result3.n3_id, how='full_outer')\
        .withColumn('id', coalesce('n3_id', 'n2_id', 'n1_id')).select('id', 'n1_path_risk', 'n2_path_risk', 'n3_path_risk')
    res.write.csv(option.output, mode='Overwrite', header=True)
