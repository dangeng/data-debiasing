import os
from shutil import copyfile
from random import shuffle

fnames_male = os.listdir('gender_images/male')
fnames_female = os.listdir('gender_images/female')

shuffle(fnames_male)
shuffle(fnames_female)

train_male = fnames_male[:int(len(fnames_male)*.8)]
test_male = fnames_male[int(len(fnames_male)*.8):]

train_female = fnames_female[:int(len(fnames_female)*.8)]
test_female = fnames_female[int(len(fnames_female)*.8):]

for fname in train_male:
    print(fname)
    copyfile(os.path.join('gender_images/male', fname), os.path.join('gender_images/train/male/', fname))

for fname in test_male:
    print(fname)
    copyfile(os.path.join('gender_images/male', fname), os.path.join('gender_images/test/male/', fname))

for fname in train_female:
    print(fname)
    copyfile(os.path.join('gender_images/female', fname), os.path.join('gender_images/train/female/', fname))

for fname in test_female:
    print(fname)
    copyfile(os.path.join('gender_images/female', fname), os.path.join('gender_images/test/female/', fname))
