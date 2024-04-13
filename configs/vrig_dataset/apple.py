_base_ = './hyper_default.py'

expname = 'iphone/base-apple'
basedir = './logs/iphone_data'

data = dict(
    datadir='./iphone-apple',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)