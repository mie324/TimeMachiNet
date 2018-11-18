import os

dataset = "./UTKFace/unlabeled"
labeled_destination = "./UTKFace/labeled"

def create_destination():
    if not os.path.exists(labeled_destination):
        os.mkdir(labeled_destination)

    for i in range(20):
        new_folder = os.path.join(labeled_destination,format(i,"<02"))
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


def move_dataset():
    files = [file for file in os.listdir(dataset)]
    for file in files:
        attributes = file.split("_")
        try:
            age = int(attributes[0])
            gender = int(attributes[1])
            folder = format(bucket_age(age)*2 + gender,"<02") # Combine into one number
            original_file = os.path.join(dataset,file)
            destination_file = os.path.join(labeled_destination,folder,file)
            os.rename(original_file,destination_file)
        except:
            continue