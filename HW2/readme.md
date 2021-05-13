# Reference

* [pyspark.mllib package — PySpark master documentation (apache.org)](https://spark.apache.org/docs/2.3.0/api/python/pyspark.mllib.html#pyspark.mllib.fpm.FPGrowth)

* [Python3 教程 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-tutorial.html)

## really good reference for implementation of A Priori algo. and FP growth algo.
* [Apriori: Association Rule Mining In-depth Explanation and Python Implementation | by Chonyy | Towards Data Science](https://towardsdatascience.com/apriori-association-rule-mining-explanation-and-python-implementation-290b42afdfc6)
* [FP Growth: Frequent Pattern Generation in Data Mining with Python Implementation | by Chonyy | Towards Data Science](https://towardsdatascience.com/fp-growth-frequent-pattern-generation-in-data-mining-with-python-implementation-244e561ab1c3)

## preprocess.py
* [How to write the resulting RDD to a csv file in Spark python - Stack Overflow](https://stackoverflow.com/questions/31898964/how-to-write-the-resulting-rdd-to-a-csv-file-in-spark-python?rq=1)

## Misc 
* [Python: Get execution time - w3resource](https://www.w3resource.com/python-exercises/python-basic-exercise-57.php)

* [Reading and Writing Lists to a File in Python (stackabuse.com)](https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/)

# Notice for debugging task3 with 70 50, have to set the java stack size more
spark-submit --driver-java-options -Xss4M 