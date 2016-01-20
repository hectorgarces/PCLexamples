#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/feature.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/recognition/implicit_shape_model.h>
#include <pcl/recognition/impl/implicit_shape_model.hpp>

int main (int argc, char** argv)
{
  if (argc == 0 || argc % 2 == 0)
    return (-1);

  unsigned int number_of_training_clouds = (argc - 3) / 2;

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setRadiusSearch (25.0);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> training_clouds;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> training_normals;
  std::vector<unsigned int> training_classes;

	std::cout << "INICIO DEL TRAINING!! " << std::endl;
	std::cout << "Cargando las nubes de puntos del modelo del TRAINING... " << std::endl;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	// Se cargan las nubes de puntos introducidas por parametro
  for (unsigned int i_cloud = 0; i_cloud < number_of_training_clouds - 1; i_cloud++)
  {
		// Se crea un nuevo puntero a una nube de puntos
    pcl::PointCloud<pcl::PointXYZ>::Ptr tr_cloud(new pcl::PointCloud<pcl::PointXYZ> ());

		// Se carga la nube de puntos del siguiente nombre de fichero de nube de puntos introducido como parametro
    if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[i_cloud * 2 + 1], *tr_cloud) == -1 )
      return (-1);

		// Se crea un nuevo puntero a una estrucuta tipo "pcl::Normal" donde almacenar los vectores normales a la superficie de la nube de puntos anterior
    pcl::PointCloud<pcl::Normal>::Ptr tr_normals = (new pcl::PointCloud<pcl::Normal>)->makeShared ();
		
		// Se carga la nube de puntos en el estimador de vectores normales
    normal_estimator.setInputCloud (tr_cloud);
		// Se copian los vectores normales a la esctructura "pcl::Normal"
    normal_estimator.compute (*tr_normals);

		// Se obtiene el ID de la clase a la que pertenece esta nube de puntos
    unsigned int tr_class = static_cast<unsigned int> (strtol (argv[i_cloud * 2 + 2], 0, 10));

		// se rellenan los 3 vectores del entrenador con: 
		//	la nube de puntos, 
		//	los vectores normales y 
		//	el ID de la clase
    training_clouds.push_back (tr_cloud);
    training_normals.push_back (tr_normals);
    training_classes.push_back (tr_class);
  }
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "Nubes de puntos cargadas... " << std::endl;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// FPFH = Fast Point Feature Histograms Descriptors
	// Se instancia el estimador tipo FPFH
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> >::Ptr fpfh
    (new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> >);
  fpfh->setRadiusSearch (25.0);
  pcl::Feature< pcl::PointXYZ, pcl::Histogram<153> >::Ptr feature_estimator(fpfh);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Se crea la instancia del estimador ISM con 153 parametros
  pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal> ism;

  ism.setFeatureEstimator(feature_estimator);
  ism.setTrainingClouds (training_clouds);
  ism.setTrainingNormals (training_normals);
  ism.setTrainingClasses (training_classes);
  ism.setSamplingSize (2.0f);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Se lanza el entrenamiento...
	std::cout << "Iniciando el entrenamiento... " << std::endl;

  pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal>::ISMModelPtr model = boost::shared_ptr<pcl::features::ISMModel>
    (new pcl::features::ISMModel);
  ism.trainISM (model);
	std::cout << "Entrenamiento finalizado " << std::endl;
	
	// Se guarda el modelo del entrenamiento creado en un fichero de texto
	std::cout << "Se almacena el modelo entrenado en el fichero: trained_ism_model.txt  " << std::endl;
  std::string file ("trained_ism_model.txt");
  model->saveModelToFile (file);
  model->loadModelFromfile (file);
	std::cout << "FIN DEL TRAINING!! " << std::endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Una vez realizado el entrenamiento del modelo, se pasa a realizar el test
	std::cout << "INICIO DEL TEST!! " << std::endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Se cargan los datos de la nube de puntos a ser testeada
	// "testing_class" es el ID de la clase prevista
	// En argv[argc - 1] se almacena el ID de la clase a ser clasificada
	unsigned int testing_class = static_cast<unsigned int> (strtol (argv[argc - 1], 0, 10));

	// "testing_cloud" es la nube de puntos a ser clasificada
	// En argv[argc - 2] se almacena el nombre del fichero PCD de la nube de puntos a ser clasificada
  pcl::PointCloud<pcl::PointXYZ>::Ptr testing_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[argc - 2], *testing_cloud) == -1 )
    return (-1);

  pcl::PointCloud<pcl::Normal>::Ptr testing_normals = (new pcl::PointCloud<pcl::Normal>)->makeShared ();
  normal_estimator.setInputCloud (testing_cloud);
  normal_estimator.compute (*testing_normals);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Se lanza el proceso de clasificacion de la nube de puntos de test
	// Esta clasificacion la realiza el metodo FINDOBJECTS del objeto ISM
	//		model = Modelo obtenido en el TRAIN
	//		testing_cloud = Nube de puntos a ser clasificada
	//		testing_normals
	//		testing_class = ID de la clase
	// En este proceso se busca objetos tipo "testing_class"(ID_CLASE) dentro de "testing_cloud"
	// En "vote_list" se devuelve una lista de votos (COMPLETAR)
  boost::shared_ptr<pcl::features::ISMVoteList<pcl::PointXYZ> > vote_list = ism.findObjects (
    model,
    testing_cloud,
    testing_normals,
    testing_class);	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  double radius = model->sigmas_[testing_class] * 10.0;
  double sigma = model->sigmas_[testing_class];
  std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak> > strongest_peaks;
  vote_list->findStrongestPeaks (strongest_peaks, testing_class, radius, sigma);

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = (new pcl::PointCloud<pcl::PointXYZRGB>)->makeShared ();
  colored_cloud->height = 0;
  colored_cloud->width = 1;

  pcl::PointXYZRGB point;
  point.r = 255;
  point.g = 255;
  point.b = 255;

	std::cout << "Imprimiendo los Strongest Points" << std::endl;
  for (size_t i_vote = 0; i_vote < strongest_peaks.size (); i_vote++)
  {
    point.x = strongest_peaks[i_vote].x;
    point.y = strongest_peaks[i_vote].y;
    point.z = strongest_peaks[i_vote].z;
    //colored_cloud->points.push_back (point);
		std::cout << "Strongest Peaks_" <<  i_vote << ": " << point.x << ", " << point.y << ", " << point.z << ", " << std::endl;
  }
	std::cout << "Imprimiendo los TestingCloud Points" << std::endl;
  for (size_t i_point = 0; i_point < testing_cloud->points.size (); i_point++)
  {
    point.x = testing_cloud->points[i_point].x;
    point.y = testing_cloud->points[i_point].y;
    point.z = testing_cloud->points[i_point].z;
		std::cout << "TestingCloud Point_" <<  i_point << ": " << point.x << ", " << point.y << ", " << point.z << ", " << std::endl;	
    //colored_cloud->points.push_back (point);
  }


  for (size_t i_point = 0; i_point < testing_cloud->points.size (); i_point++)
  {
    point.x = testing_cloud->points[i_point].x;
    point.y = testing_cloud->points[i_point].y;
    point.z = testing_cloud->points[i_point].z;
    colored_cloud->points.push_back (point);
  }
  colored_cloud->height += testing_cloud->points.size ();

  point.r = 255;
  point.g = 0;
  point.b = 0;
  for (size_t i_vote = 0; i_vote < strongest_peaks.size (); i_vote++)
  {
    point.x = strongest_peaks[i_vote].x;
    point.y = strongest_peaks[i_vote].y;
    point.z = strongest_peaks[i_vote].z;
    colored_cloud->points.push_back (point);
  }
  colored_cloud->height += strongest_peaks.size ();

  pcl::visualization::CloudViewer viewer ("Result viewer");
  viewer.showCloud (colored_cloud);
  while (!viewer.wasStopped ())
  {
  }

  return (0);
}
