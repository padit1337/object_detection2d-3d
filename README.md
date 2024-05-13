# object_detection2d-3d
It analyses the RGB image stream for a watermelon, then clusters the biggest 3D voxel cluster inside the bouning box and identifying its center, assuming that the biggest cluster inside the watermelon bounding box is the watermelon itself. Then it uses the URScript language (similar to python) to move the robot to the desired location. The current calculation of the translational and rotational matrix is broken, i will fix this ASAP.
## The interpretation of the 3D voxel space seen by the camera
The black box around the melon indicates how the bounding box of the 2D watermelon is transformed into a 3D "tube". Inside the area marked by the bounding box, a DBScan clustering algorithm looks for the biggest cluster and subsequently it is tried to fit an elipsoid over this cluster (Since the watermelon is the biggest cluste inside the bounding box and the watermelon is expected to be sufficiently close to the shape of an elipsoid). 
![grafik](https://github.com/padit1337/object_detection2d-3d/assets/45203588/45464603-933e-433d-90fa-5739ebd71fce)
## The mount of the Intel RealSense 415 based on the work of bono88 https://www.thingiverse.com/thing:5394492  
![grafik](https://github.com/padit1337/object_detection2d-3d/assets/45203588/506ae568-1633-447d-951e-adc40fb76702)

Then the code moves the UR5 robot to the location of the watermelon, using the URScript language. The kinematics is currently broken, so I need to redo the math, which i will do ASAP. currently the robot moves, just in the wrong direction, because the translation and rotation matrix is flawed. 
