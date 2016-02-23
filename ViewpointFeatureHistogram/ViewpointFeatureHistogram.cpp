#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>

int main (int argc, char** argv)
{
  std::string fileName = argv[1];
  std::cout << "Reading " << fileName << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (fileName, *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file");
    return (-1);
  }

  std::cout << "Loaded " << cloud->points.size() << " points." << std::endl;

  // Compute the normals
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
  normalEstimation.setInputCloud (cloud);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  normalEstimation.setSearchMethod (tree);

  pcl::PointCloud<pcl::Normal>::Ptr cloudWithNormals (new pcl::PointCloud<pcl::Normal>);

  normalEstimation.setRadiusSearch (0.03);

  normalEstimation.compute (*cloudWithNormals);

  std::cout << "Computed " << cloudWithNormals->points.size() << " normals." << std::endl;
  
  // Setup the feature computation
  typedef pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> VFHEstimationType;
  VFHEstimationType vfhEstimation;

  // Provide the original point cloud (without normals)
  vfhEstimation.setInputCloud (cloud);

  // Provide the point cloud with normals
  vfhEstimation.setInputNormals(cloudWithNormals);

  // Use the same KdTree from the normal estimation
  vfhEstimation.setSearchMethod (tree);

  //vfhEstimation.setRadiusSearch (0.2); // With this, error: "Both radius (.2) and K (1) defined! Set one of them to zero first and then re-run compute()"

  // Actually compute the VFH features
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhFeatures(new pcl::PointCloud<pcl::VFHSignature308>);
  vfhEstimation.compute (*vfhFeatures);

  std::cout << "output points.size (): " << vfhFeatures->points.size () << std::endl; // This outputs 1 - should be 397!

  // Display and retrieve the shape context descriptor vector for the 0th point.
  pcl::VFHSignature308 descriptor = vfhFeatures->points[0];
  VFHEstimationType::PointCloudOut::PointType descriptor2 = vfhFeatures->points[0];
  std::cout << "VFH:" << descriptor << std::endl;
  std::cout << "Numero de Elementos del VFH = " << sizeof(descriptor.histogram)/sizeof(descriptor.histogram[0]) << std::endl;


	// Create *_vfh.pcd file
	std::stringstream vfh_file;
	vfh_file << "# .PCD v.6 - Point Cloud Data file format" << std::endl;
	vfh_file << "FIELDS vfh" << std::endl;
	vfh_file << "SIZE 4" << std::endl;
	vfh_file << "TYPE F" << std::endl;
	vfh_file << "COUNT 308" << std::endl;
	vfh_file << "WIDTH 1" << std::endl;
	vfh_file << "HEIGHT 1" << std::endl;
	vfh_file << "POINTS 1" << std::endl;
	vfh_file << "DATA ascii" << std::endl;
	int vfh_length = sizeof(descriptor.histogram)/sizeof(descriptor.histogram[0]); 
	for (int i = 0; i < vfh_length; i++)
	{
		vfh_file << descriptor.histogram[i] << " ";
	}

	std::ofstream outFile;
	outFile.open("Prueba_vfh.pcd");
	outFile << vfh_file.str();
	outFile.close();

  return 0;
}
