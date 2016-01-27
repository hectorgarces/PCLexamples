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
  	//model->loadModelFromfile (file);
	std::cout << "FIN DEL TRAINING!! " << std::endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 	return (0);
}
