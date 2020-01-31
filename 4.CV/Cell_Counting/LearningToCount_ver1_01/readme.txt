LearningToCount,  ver. 1.0
by Victor Lempitsky, Oxford University, February 2011
Contact e-mail:   victorlempitsky@gmail.com

This MATLAB software implements and demonstrates the framework from the following publication:

V. Lempitsky, A. Zisserman
"Learning To Count Objects in Images"
NIPS 2010
http://www.robots.ox.ac.uk/~vgg/research/counting/index.html

---------------------------------------
INCLUDED:

LearnToCount.m =  main learning routine
maxsubarray2D.cpp = auxiliary routine for 2D max subarray (Kadane's algorithm)
data/*.png = sample cell microscopy images and their dotted annotation generated with the SIMCEP tool 
		   (http://www.cs.tut.fi/sgn/csb/simcep/tool.html)
CellCountingExample.m = demonstrates the framework on these images
dictionary256.mat = precomputed SIFT codebook for this dataset required to run the example

---------------------------------------
LICENSE:

The code is provided for research purposes only. No commercial use is allowed.
Please cite the above-mentioned paper, should you use this code in the preparation of your publication.

Bibtex:
@InProceedings{LempitskyZissermanNIPS2010,
  author       = "Lempitsky, V. and Zisserman, A.",
  title        = "Learning To Count Objects in Images",
  booktitle    = "Advances in Neural Information Processing Systems",
  year         = "2010",
}

--------------------------------------
RUNNING THE EXAMPLE:

1)To run the example you will need to install the excellent VLFeat toolbox (www.vlfeat.org)

2)The learning procedure calls 'quadprog' or 'linprog' (depending on the type of the regularization). These methods from MATLAB's optimization toolbox would be slow (not sure how slow, but perhaps many hours). Use some faster solver instead, e.g. install MOSEK (www.mosek.com)- they are providing free academic licenses at the moment. Since MOSEK supercedes linprog and quadprog, no change in the code is required.

-------------------------------------
ADAPTING TO YOUR PROBLEM:

1)Please refer to the paper on how the framework works
2)Please refer to the comments in LearnToCount.m on how to use it for your task.