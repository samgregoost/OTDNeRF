import pandas as pd

#read csv file into pandas dataframe
df = pd.read_csv('results.csv')

substrings = [
    'r', 'd', 'w', 'w-r-i-l-s', 'w-r-d-i-l-s', 'w-i-l-s', 'w-d-i-l-s'
]

#pick all the rows where a column string ends with a substring in a pandas dataframe
for substring in substrings:
    mpsnr = df[df['dataset'].str.endswith(substring)][['mPSNR']].mean()
    mlpips = df[df['dataset'].str.endswith(substring)][['LPIPs']].mean()
  #  mssim = df[df['dataset'].str.endswith(substring)][['mSSIM']].mean()

    ssim_list = df[df['dataset'].str.endswith(substring)][['mSSIM']]['mSSIM'].str.split('(').tolist()

    val = 0
    for i in ssim_list:
        val += (float(i[1].split(',')[0]))

    mssim = val/(len(ssim_list))

    print(f'For {substring} mPSNR: {mpsnr["mPSNR"]} mLPIPs: {mlpips["LPIPs"]} mSSIM: {mssim}')



