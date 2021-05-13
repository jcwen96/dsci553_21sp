import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.json4s._
import org.json4s.jackson.JsonMethods._
import scala.collection.mutable._
import org.json4s.jackson.Serialization
import java.io._

object task1 {
  def main(args: Array[String]) {

    // parse commandline argument
    val inputPath = args(0)
    val outputPath = args(1)
    val stopwordsPath = args(2)
    val givenYear = args(3)
    val topMUser: Int = args(4).toInt
    val topNWord: Int = args(5).toInt

    // set up spark
    val conf = new SparkConf().setAppName("Task1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val reviewsRDD = sc.textFile(inputPath).map(line => parse(line)).cache()

    var result: LinkedHashMap[String, Any] = LinkedHashMap()

    result += ("A" -> reviewsRDD.count())
    println(s"total number of reviews: ${result("A")}")

    result += ("B" -> reviewsRDD.filter(review => compact(render(review \ "date"))regionMatches(false, 1, givenYear, 0, 4)).count())
    println(s"total number of reviews in a given year: ${result("B")}")

    result += ("C" -> reviewsRDD.map(review => compact(render(review \ "user_id"))).distinct().count())
    println(s"number of distinct users who have written the reviews: ${result("C")}")

    def mapFunc(x: JValue): (String, Int) = {
      val str = compact(render(x \ "user_id"))
      (str.substring(1, str.length - 1), 1)
    }
    val users = reviewsRDD.map(mapFunc).reduceByKey(_+_).takeOrdered(topMUser)(Ordering[(Int, String)].on(x => (-x._2, x._1)))
    val res:Array[List[Any]] = new Array(users.length)
    for (i <- 0 until users.length) {
      res(i) = users(i).productIterator.toList
    }
    result += ("D" -> res)

    // scala is so troublesome, give up E

    implicit val formats = org.json4s.DefaultFormats
    val writer = new PrintWriter(new File(outputPath))
    writer.write(Serialization.write(result))
    writer.close()
  }
}
