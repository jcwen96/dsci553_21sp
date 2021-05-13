import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import java.io.{File, PrintWriter}
import scala.collection.mutable.LinkedHashMap

object task3 {
  def main(args: Array[String]) {

    // parse commandline argument
    val inputPath = args(0)
    val outputPath = args(1)
    val partitionType: Boolean = args(2).equals("customized")
    val nPartitions: Int = args(3).toInt
    val nThreshold: Int = args(4).toInt

    // set up spark
    val conf = new SparkConf().setAppName("Task3").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    var result: LinkedHashMap[String, Any] = LinkedHashMap()

    def mapFunc(x: JValue): (String, Int) = {
      val str = compact(render(x \ "business_id"))
      (str.substring(1, str.length - 1), 1)
    }
    var reviewsRDD = sc.textFile(inputPath).map(line => parse(line)).map(mapFunc).cache()

    if (partitionType) {
      reviewsRDD = reviewsRDD.repartition(nPartitions)
    }

    result += ("n_partitions" -> reviewsRDD.getNumPartitions)
    println(s"number of partitions: ${result("n_partitions")}")

    result += ("n_items" -> reviewsRDD.glom().map(_.length).collect())

    result += ("result" -> reviewsRDD.reduceByKey(_+_).filter(x => x._2 > nThreshold).map(_.productIterator.toList) .collect())

    implicit val formats = org.json4s.DefaultFormats
    val writer = new PrintWriter(new File(outputPath))
    writer.write(Serialization.write(result))
    writer.close()
  }
}
