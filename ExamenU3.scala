//14212328 Garcia Sanchez Cesar Octavio

//Importar un sesion de Spark
import org.apache.spark.sql.SparkSession

//Utilice las lineas de codigo para reportar error reducidos
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Crea instancia de la sesion de spark
val spark = SparkSession.builder().getOrCreate()

//Importar la libreria de Kmeans para el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans

//Cargar el dataset
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

//printschema
dataset.printSchema()

//Selecccionarr la siguente columnas para conjunto de entrenamiento
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
feature_data.head(3)

//Importar Vector Assembler and Vector
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

//Crear un nuevo objeto Vector Assembeler para las columnas de caracteristicas como un conjunto de entrada
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

//Utilice el objeto assembler para transformar
val training_data = assembler.transform(feature_data).select("features")

//Crear un modelo Kmeans = 3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training_data)

//Evaluar los grupos usnado WSSE
val WSSE = model.computeCost(training_data)
println(s"Within set sum of Squared Errors = $WSSE")

//Mostrar resultados
println("Cluster Centers: ")
model.clusterCenters.foreach(println)