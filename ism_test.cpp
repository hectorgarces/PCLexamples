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


  	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> >::Ptr fpfh
    	(new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> >);
  	fpfh->setRadiusSearch (25.0);
  	pcl::Feature< pcl::PointXYZ, pcl::Histogram<153> >::Ptr feature_estimator(fpfh);
	
	pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal> ism;
	ism.setFeatureEstimator(feature_estimator);
  	ism.setSamplingSize (2.0f);

  	pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal>::ISMModelPtr model = boost::shared_ptr<pcl::features::ISMModel>
    	(new pcl::features::ISMModel);
	model->reset();

  	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  	normal_estimator.setRadiusSearch (25.0);

  	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> training_clouds;
  	std::vector<pcl::PointCloud<pcl::Normal>::Ptr> training_normals;
  	std::vector<unsigned int> training_classes;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Una vez realizado el entrenamiento del modelo, se pasa a realizar el test
	std::cout << "INICIO DEL TEST!! " << std::endl;
	// Se carga el modelo del entrenamiento creado en un fichero de texto
	std::cout << "Se carga el modelo entrenado en el fichero: trained_ism_model.txt  " << std::endl;
  	std::string file ("trained_ism_model.txt");
  	model->loadModelFromfile (file);
	std::cout << "Modelo cargado!" << std::endl;	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  	// Se cargan los datos de la nube de puntos a ser testeada
	// "testing_class" es el ID de la clase prevista
	// En argv[argc - 1] se almacena el ID de la clase a ser clasificada
	unsigned int testing_class = static_cast<unsigned int> (strtol (argv[argc - 1], 0, 10));

	// "testing_cloud" es la nube de puntos a ser clasificada
	// En argv[argc - 2] se almacena el nombre del fichero PCD de la nube de puntos a ser clasificada
	std::cout << "Abriendo el fichero PCD con la nube de puntos a clasificar/reconocer" << std::endl;	
  	pcl::PointCloud<pcl::PointXYZ>::Ptr testing_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  	if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[argc - 2], *testing_cloud) == -1 )
    	return (-1);
	std::cout << "Nube de Puntos en fichero PCD, CARGADA!" << std::endl;
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
	std::cout << "Se inicia el proceso de votación" << std::endl;
  	boost::shared_ptr<pcl::features::ISMVoteList<pcl::PointXYZ> > vote_list = ism.findObjects (
    	model,
    	testing_cloud,
    	testing_normals,
    	testing_class);	
	std::cout << "Proceso de votación Finalizado" << std::endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  	double radius = model->sigmas_[testing_class] * 10.0;
  	double sigma = model->sigmas_[testing_class];
	std::cout << "Se buscan los Strongest Peaks!!" << std::endl;
  	std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak> > strongest_peaks;
  	vote_list->findStrongestPeaks (strongest_peaks, testing_class, radius, sigma);
	std::cout << "Strongest Peaks, OBTENIDOS" << std::endl;

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
 /* 	for (size_t i_point = 0; i_point < testing_cloud->points.size (); i_point++)
  	{
    	point.x = testing_cloud->points[i_point].x;
   	 	point.y = testing_cloud->points[i_point].y;
    	point.z = testing_cloud->points[i_point].z;
		std::cout << "TestingCloud Point_" <<  i_point << ": " << point.x << ", " << point.y << ", " << point.z << ", " << std::endl;	
    	//colored_cloud->points.push_back (point);
  	}*/


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

 // 	pcl::visualization::CloudViewer viewer ("Result viewer");
 // 	viewer.showCloud (colored_cloud);
 // 	while (!viewer.wasStopped ())
  //	{
  //	}

  	return (0);
}
