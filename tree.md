Inside data_prep please use this tree for running in custom dataset (within some days i will write code so that we can use seperate datasets/.py files for data loader part in custom dataset training)

Data|
    |
    |
    |--> ScanNet (clone from scanet repo)
    |
    |--> scans|-->scene0003_02|-->annotation----------->Segmentation----->0 to 1000 images with w,h,3 shape
	|		  |				  |---> plane_info.npy (usage is optional)
	|		  |				  |---> planes.npy(use plan parameters)
    |         |               |
    |         |               |
    |         |               |
    |         |		          |-->frames--------------->color 0 to 1000 images with w,h,3 shape
    |		  |			      |         --------------->depth 0 to 1000 images with w,h,3 shape
    |		  |		          |         --------------->intrinsic (default)
    |		  |			      |          ----------->pose (default)	or else could be got from using "--testingIndex=0" for each image                    	 |		   |			   |
    |         |               |
    |         |               |
    |         |               |-->scene0003_02.txt 
    |         
    |--> tasks|
    |         |----> Scannetv2-labels.combined.tsv
    |
    |--> anchor_planes_N.npy
    |
    |-->invalid_indices_test.txt
    |
    |--> invalid_indices_train.txt
    |
    |
