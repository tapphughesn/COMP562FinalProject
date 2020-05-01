----ABOUT THIS PROJECT----

I'm an undergraduate working with Dr. Martin Styner working on a deep learning based brain 
segmentation project. My work builds from one of Dr. Styner's old students, Han Wang. 

----AN EXPLANATION OF EACH DIRECTORY----

DATA - a place to store the data (usually .nii.gz format) that my models train and test on

models - the Keras models I create or pull from elsewhere.
The testing and training scripts, as well as the model checkpoints are found here.

old_work - old work that I have done in deep learning. here for reference

processing_scripts - (usually python) scripts for (pre- and post-)processing/formatting the data

segmentation_tool - relevant files for my main project, a program which takes a MRI output file
(and perhaps some other parameters), segments the brain structures, and 

slurm_scripts - .sh SLURM scripts which I run when I need to submit jobs on longleaf/do computation


han_trainingData - contains han's (the student who worked in this area with Dr. Styner before me) training data should I ever need it.
