# ABCP

Automatic Block-wise and Channel-wise Network Pruning (ABCP) jointly search the block pruning policy and the channel pruning policy of the network with deep reinforcement learning (DRL). A joint sample algorithm is proposed to simultaneously generate the pruning choice of each residual block and the channel pruning ratio of each convolutional layer from the discrete and continuous search space respectively.

## Detection Datasets for ABCP

These three datasets are collected for the evaluation of ABCP.

### The UCSD dataset

<p align="center"><img src="misc/UCSDsparse.jpg" width="22%"/>  <img src="misc/UCSDmedium.jpg" width="22%"/>  <img src="misc/UCSDdense.jpg" width="22%"/>
  
The UCSD dataset is a small dataset captured from the freeway surveillance videos collected by [UCSD](http://www.svcl.ucsd.edu/projects/traffic/). This dataset involves three different traffic densities each making up about one-third: the sparse traffic, the medium-density traffic, and the dense traffic. We define three classes in this dataset: truck, car, and bus. The vehicles in the images are labeled for the detection task. The resolutions of the images are all 320×240. The training and testing sets contain 683 and 76 images respectively.

### The mobile robot detection dataset

<p align="center"><img src="misc/robot1.jpg" width="25%"/>  <img src="misc/robot2.jpg" width="25%"/>
<p align="center"><img src="misc/robot3.jpg" width="20%"/>  <img src="misc/robot4.jpg" width="20%"/> <img src="misc/robot5.jpg" width="20%"/> <img src="misc/robot6.jpg" width="20%"/>
  
The mobile robot detection dataset is collected by the robot-mounted cameras to meet the requirements of the fast and lightweight detection algorithms for the mobile robots, which is inspired [RoboMaster Univeristy AI Challenge](https://www.robomaster.com/en-US/robo/icra). There are two kinds of ordinary color camera with different resolutions which are 1024×512 and 640×480 respectively. Five classes have been defined: red robot, red armor, blue robot, blue armor, dead robot. The training and testing sets contain 13,914 and 5,969 images respectively. During collecting, we change series of exposure and various distances and angles of the robots to improve the robustness.
  
### The sim2real dataset
  
<p align="center"><img src="misc/real-world.jpg" width="23%"/>  <img src="misc/simulation.jpg" width="23%"/>

The sim2real detection dataset is divided into two sub-datasets: the real-world dataset and the simulation dataset. We search and train the model on the simulation dataset and test it on the real-world dataset. Firstly, we collect the real-world dataset by the surveillance-view ordinary color cameras in the field. The field and the mobile robots are the same as those in the mobile robot detection dataset. Secondly, we leverage Gazebo to simulate the robots and the field from the surveillance view. Then we capture the images of the simulation environment to collect the simulation dataset. The resolutions of images in the sim2real dataset are all 640×480. There is only one object class in these two datasets: robot. The training and testing sets of the simulation dataset contain 5,760 and 1,440 respectively, and the testing set of the real-world dataset contains 3,019 images.

### Label information 
The format of the labels is relative xywh coordinates. The documents named train.txt and test.txt list the image paths of the training dataset and the testing dataset respectively, and are used for the YOLOv3 training on Darknet.  The documents named search_train.txt and search_test.txt list the image paths and the labels of the training dataset and the testing dataset respectively, and are used for the pruning policy search. It is worth noting that the format of the labels is absolute xxyy coordinates. 
  
### Download
The data could be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1RmhjxdZqri_V5GCBnrtI5w) (Pwd: redc) and [OneDrive](https://1drv.ms/u/s!Asc-xz451d9bnSpIYWq_qetgJh5y?e=YcaGWM).
