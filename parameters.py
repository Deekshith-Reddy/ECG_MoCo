spatioTemporalParams_v4 =dict(
integrationMethod = 'concat',
firstLayerParams = dict(in_channels=1,out_channels=32,bias=True,kernel_size=(1,7),maxPoolKernel=7),
lastLayerParams = dict(maxPoolSize=(8,1)),
temporalResidualBlockParams = dict(
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[3]*4,
	padding	    =[1]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	),	
spatialResidualBlockParams = dict(
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[7]*4,
	padding	    =[3]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	)	
)