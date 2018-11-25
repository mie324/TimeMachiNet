import os
from shutil import copyfile


dataset = "data/UTKFace/unlabeled"
labeled_destination = "data/UTKFace/labeled"

def create_destination():
    if not os.path.exists(labeled_destination):
        os.mkdir(labeled_destination)

    for i in range(20):
        new_folder = os.path.join(labeled_destination,str(i))
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)


def bucket_age(n):
    if n<=5:
        return 0
    elif n<=10:
        return 1
    elif n<=15:
        return 2
    elif n<=20:
        return 3
    elif n<=30:
        return 4
    elif n<=40:
        return 5
    elif n<=50:
        return 6
    elif n<=60:
        return 7
    elif n<=70:
        return 8
    else:
        return 9

def convertname(name):
    if name=="10":
        return "1"
    elif name=="00":
        return "0"
    elif name=="20":
        return "2"
    elif name=="30":
        return "3"
    elif name=="40":
        return "4"
    elif name=="50":
        return "5"
    elif name=="60":
        return "6"
    elif name=="70":
        return "7"
    elif name=="80":
        return "8"
    elif name=="90":
        return "9"
    else:
        return name


def move_dataset():
    files = [file for file in os.listdir(dataset)]
    for file in files:
        attributes = file.split("_")
        try:
            age = int(attributes[0])
            gender = int(attributes[1])
            if(bucket_age(age) == 5 and gender == 0):
                folder = "10"
            else:
                folder = bucket_age(age)*2 + gender # Combine into one numberfolder)
                folder = convertname(str(folder))
            original_file = os.path.join(dataset,file)
            destination_file = os.path.join(labeled_destination,folder,file)
            copyfile(original_file,destination_file)
        except:
            continue