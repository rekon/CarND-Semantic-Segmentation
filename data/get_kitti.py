from urllib.request import urlretrieve
import os
import zipfile

urlretrieve(
    'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip',
    'kitti.zip')

print('Extracting model...')
zip_ref = zipfile.ZipFile('./kitti.zip', 'r')
zip_ref.extractall('./')
zip_ref.close()

# Remove zip file to save space
os.remove('./kitti.zip')
