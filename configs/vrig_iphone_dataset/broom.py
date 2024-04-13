_base_ = './iphone_vrig_default.py'
#expname = 'iphone/base-apple'
#expname = 'iphone/base-apple-wreg' #Wreg 1, finetune 0, train+test
#expname = 'iphone/base-apple-wreg-finetune' #Wreg 1, finetune 1, train+test
#expname = 'iphone/base-apple-wreg-train' #Wreg 1, finetune 0, train
#expname = 'iphone/base-apple-wreg-gauss' #Wreg 1, finetune 0, train + test, gauss
#expname = 'iphone/base-apple-wreg-weight' #Wreg 1, finetune 0, train + test, weight
#expname = 'iphone/base-apple-wreg-diffsample' #Wreg 1, finetune 0, train + test, diffsample
#expname = 'iphone/base-apple-wreg-diffvec' #Wreg 1, finetune 0, train + test, diffvec
#expname = 'iphone/base-apple-wreg-interp' #Wreg 1, finetune 0, train + test, interp
#expname = 'iphone/base-apple-wreg-interp-diffsample' #Wreg 1, finetune 0, train + test, diffsample, interp
expname = 'iphone-vrig/base-broom' #Wreg 1, finetune 0, train + test, diffsample, interp

basedir = './logs/iphone_vrig_data'

data = dict(
    datadir='./iphone-vrig-broom',
    dataset_type='iphone_dataset',
    white_bkgd=False,
)