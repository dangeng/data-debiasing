import os
from shutil import copyfile

with open('data/female_names.txt', 'rb') as f:
    female_fnames = f.readlines()
female_fnames = [fname[:-2] for fname in female_fnames]

fnames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("data/gender_images")) for f in fn]

for fname in fnames:
    print(fname.split('/')[-1])
    if fname.split('/')[-1] in female_fnames:
        copyfile(fname, os.path.join('female', fname.split('/')[-1]))
        print('female')
    else:
        copyfile(fname, os.path.join('male', fname.split('/')[-1]))
        print('male')
