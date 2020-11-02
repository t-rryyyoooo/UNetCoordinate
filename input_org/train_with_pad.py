{
	"api_key" : "IIowbTppLPOohqhcDtzxw76CotowbTppLPOohqhcDtzxw76Cot",
	"project_name" : "abdomen",
	"experiment_name" : "with_pad",
	"log" : "/home/vmlab/Desktop/data/log/abdomen/with_pad/28-44-44/mask",
        "model_module_name" : "UNet_with_pad",
        "system_name" : "UNetSystem",
        "checkpoint_name" : "BestAndLatestModelCheckpoint",
	"dataset_path" : "/home/vmlab/Desktop/data/patch/Abdomen/28-44-44/mask", 
	"criteria" : {
		"train" : ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
		
		"val" : ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]
        },
	"in_channel" : 1, 
	"num_class" : 14,
	"learning_rate" : 0.001,
	"batch_size" : 32, 
	"num_workers" : 6,
	"epoch" : 500,
	"gpu_ids" : [0], 
	"model_savepath" : "/home/vmlab/Desktop/data/modelweight/Abdomen/28-44-44/mask"
}

	
