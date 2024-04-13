_base_ = './iphone_default.py'
#expname = 'iphone/base-teddy'
#expname = 'iphone/base-teddy-wreg' #Wreg 1, finetune 0, train+test
#expname = 'iphone/base-windmill-wreg-0.05'
#expname = 'iphone/base-apple-wreg-finetune' #Wreg 1, finetune 1, train+test
#expname = 'iphone/base-teddy-wreg-train' #Wreg 1, finetune 0, train
#expname = 'iphone/base-apple-wreg-gauss' #Wreg 1, finetune 0, train + test, gauss
#expname = 'iphone/base-apple-wreg-weight' #Wreg 1, finetune 0, train + test, weight
#expname = 'iphone/base-apple-wreg-diffsample' #Wreg 1, finetune 0, train + test, diffsample
#expname = 'iphone/base-apple-wreg-diffvec' #Wreg 1, finetune 0, train + test, diffvec
#expname = 'iphone/base-teddy-wreg-interp-train' #Wreg 1, finetune 0, train + test, interp
#expname = 'iphone/base-teddy-wreg-interp-linear-train' #Wreg 1, finetune 0, train + test, interp
#expname = 'iphone/base-windmill-wreg-interp-diffsample' #Wreg 1, finetune 0, train + test, diffsample, interp
#expname = 'iphone/base-apple-wreg-interp-long-diffsample' #Wreg 1, finetune 0, train + test, diffsample, interp

#expname = 'iphone/base-teddy-wreg-interp-linear-ns-train'
#expname = 'iphone/base-teddy-wreg-interp-cubic-ns-train'



#expname = 'iphone/base-teddy-wreg-interp-gaussian01-ns-train'
#expname = 'iphone/base-teddy-wreg-interp-gaussian005-ns-train'

#expname = 'iphone/base-apple-wreg-interpoff'
basedir = './logs/iphone_data'

data = dict(
    datadir='./iphone-block',
    dataset_type='iphone_dataset',
    white_bkgd=False,
)

