import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.LinkedHashMap

object task2 {
  def main(args: Array[String]) {

    // parse commandline argument
    val reviewFile = args(0)
    val businessFile = args(1)
    val outputFile = args(2)
    val isSpark: Boolean = args(3).equals("spark")
    val nCategories: Int = args(4).toInt

    // set up spark
    val conf = new SparkConf().setAppName("Task3").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    var result: LinkedHashMap[String, Any] = LinkedHashMap()
  }
}
