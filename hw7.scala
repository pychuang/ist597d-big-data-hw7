import java.io.File
import java.io.PrintWriter
import java.io.Serializable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.collection.mutable

import breeze.linalg.DenseVector

object Homework7 extends Serializable {
  def work() = {
    val configuration = new Configuration()
    configuration.addResource(new Path("/usr/hdp/2.3.0.0-2557/hadoop/conf/core-site.xml"))

    //val lines = sc.textFile(FileSystem.get(configuration).getUri + "/ist597j/PubMed/pubmed.csv")
    val lines = sc.textFile(FileSystem.get(configuration).getUri + "/storage/md1/share/work/classes/ist597d-big-data/hw7/pubmed.csv")
    val rows = lines.map(line => line.split(','))
    val papers = rows.map(rowToPaper)
    val wordsOfPapers = papers.map(getWordsOfPaper)

    // construct feature vectors
    val hashingTF = new HashingTF()
    val tfVectors = hashingTF.transform(wordsOfPapers)
    val idfModel = new IDF().fit(tfVectors)
    val tfidfVectors = idfModel.transform(tfVectors).cache()

    // Cluster the data
    val numClusters = 3
    val numIterations = 1000

    val kmModel = KMeans.train(tfidfVectors, numClusters, numIterations)
    val clustersOfPapers = kmModel.predict(tfidfVectors)

    // Prepare the report

    val numPapersInClusters = clustersOfPapers.groupBy(x=>x).map(x=>(x._1, x._2.size))
    val numPapersInOrderedClusters = numPapersInClusters.sortBy(_._1)

    val keywordsOfPapers = papers.map(getKeywordsOfPaper)

    val clusterKeywordsPairsOfPapers = clustersOfPapers.zip(keywordsOfPapers)
    // clusterKeywordsPairsOfPapers: [(1, ["k1", "k2"]), (0, ["k3", "k4", ...]),... ]

    val keywordsOfClusters = clusterKeywordsPairsOfPapers.groupBy(_._1).mapValues(_.flatMap(_._2))

    val sortedKeywordCountPairsOfClusters = keywordsOfClusters.mapValues(keywords => toSortedKeywordCountPairs(keywords.toArray))
    val sortedKeywordCountPairsOfOrderedClusters = sortedKeywordCountPairsOfClusters.sortBy(_._1)

    val paperCountKeywordCountPairs = numPapersInOrderedClusters.zip(sortedKeywordCountPairsOfOrderedClusters)
    paperCountKeywordCountPairs.collect().foreach(x => {  // it is important to use collect() here, or the print order will be random
      println("cluster " + x._1._1 + ": " + x._1._2 + " papers")
      x._2._2.take(5).foreach(kc=>println("\t" + kc._1 + " (" + kc._2 + ")"))
    })
  }

  // Input:  ["I:1", "N:a1", "N:a2", "A:blah"]
  // Output: {"I": [1], "N": ["a1", "a2"], "A": ["blah"]}
  def rowToPaper(row: Array[String]): Map[String, Array[String]]= {
      return row.groupBy(_(0).toString).map(x => (x._1, x._2.map(_.substring(2))))
  }

  def getWordsOfPaper(paper: Map[String, Array[String]]): Iterable[String] = {
    val abstracts = paper.getOrElse("A", Array())
    val words = abstracts.flatMap(s => s.split("[ ,.;()]").toIterable)  // Note: parameter for split is regex
    val stopWords = List("", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                         "a", "A", "an", "An", "the", "The", "this", "This", "that", "That", "these", "These", "those", "Those",
                         "some", "Some", "all", "All",
                         "one", "One", "two", "Two", "three", "Three", "four", "Four",
                         "first", "First", "second", "Second", "third", "Third",
                         "be", "is", "are", "was", "were", "being", "been",
                         "do", "does", "did", "doing",
                         "can", "Can", "cannot", "Cannot", "can't", "Can't",
                         "will", "Will", "won't", "Won't", "would", "Would", "wouldn't", "Wouldn't",
                         "have", "Have", "has", "Has", "had", "Had", "having", "Having", "haven't", "Haven't", "hasn't", "Hasn't",
                         "should", "Should",
                         "may", "May", "might", "Might",
                         "I", "me", "my", "mine",
                         "you", "You", "your", "yours",
                         "he", "He", "him", "Him", "his", "His",
                         "she", "She", "her", "Her", "hers", "Hers",
                         "we", "We", "our", "Our", "ours", "Ours",
                         "they", "They", "their", "theirs",
                         "it", "It", "its",
                         "no", "No", "not", "Not",
                         "and", "And", "or", "Or", "but", "But",
                         "to", "To", "from", "From",
                         "of", "Of", "in", "In", "on", "On",
                         "for", "For",
                         "at", "At",
                         "with", "With",
                         "by", "By", "as", "As",
                         "before", "Before", "after", "After", "until", "Until",
                         "if", "If", "then", "Then", "however", "However", "because", "Because", "so", "So", "since", "Since", "such", "Such",
                         "what", "What", "when", "When", "while", "While", "where", "Where", "who", "Who", "whom", "Whom", "how", "How", "which", "Which", "why", "Why",
                         "than", "Than",
                         "too", "Too", "also", "Also", "either", "Either", "neither", "Neither",
                         "there", "There", "here", "Here")
    return words.filter(w => !stopWords.contains(w))
  }

  def getKeywordsOfPaper(paper: Map[String, Array[String]]): Array[String] = {
    val keywords = paper.getOrElse("K", Array())
    return keywords
  }

  def toSortedKeywordCountPairs(keywords: Array[String]): Array[(String, Int)] = {
    val keywordCountPairs = keywords.groupBy(x=>x).map(x => (x._1, x._2.size))
    return keywordCountPairs.toArray.sortWith(_._2 > _._2)
  }
}

Homework7.work()
