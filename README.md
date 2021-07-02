# Object detector package

## Nécessite:

Tensorflow
Driver graphique pour utiliser la version cpu
OpenCV
CV_Bridge
ROS

## Installation:

Télecharger le dossier
Mettre le dossier dans le source de catkin workspace
Faire un catkin_make
Faire source devel/setup.bash

pour ros noetic plutôt, problème de version de python et de cv_bridge sur melodic par exemple

## Fonctionnements:

Pour faire fonctionner le package, on lance par exemple :
roslaunch object_detector object_detector.launch node_name:="drone_1_detection" model_name:="model_faster_rcnn_resnet_custom_2" label_name:="labelmap.pbtxt" topic_src:="/drone_1/image" topic_dest:="drone_1_detection/image"

format d'entrée image raw
format de sortie image raw

Le package se divise de la façon suivante:
\ndata -> données de modèle et de réseau
\nlaunch -> les fichiers .launch
\nscripts -> les nodes python
src -> contient la lib de object_detection de tensorflow

La node va donc initialiser le modèle et les données relatives à tensorflow.
Il va s'abonner au flux d'images, puis on récupère l'image après l'avoir coverti au format de OpenCV.
Tensorflow va alors analyser l'image est marqué d'un carré les objets trouvés.
Enfin, on republie l'image sur un autre topic.
