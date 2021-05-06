import sys
import os

# labels that we want
labels = [7, 8, 9, 10, 11, 12]


ROOT = os.getcwd()

PATH_IMAGES = os.path.join(ROOT, 'images', 'train2017')

PATH_LABELS = os.path.join(ROOT, 'labels', 'train2017')

def main():
    for f in os.listdir(PATH_LABELS):

        if f.endswith('.txt'):

            # label complete path
            file_path = os.path.join(PATH_LABELS, f)

            labels_filtered = filter_file(file_path)


            if len(labels_filtered) != 0:
                """
                If the file has correct labels, just modify the file
                rewrite the file only with the labels that we want
                """
                rewrite_file(file_path, labels_filtered)
                print(f'Rewriting the file {file_path}.')
            else:
                """
                If the file dont have nothing that is of our interest, just remove it
                """
                remove_label_image(file_path, f)

            print('-------------------')

"""
Filters the classes that we want
"""
def filter_file(file_name):
    labels_filtered = []

    with open(file_name, 'r') as f:
        content = f.read().split('\n')[:-1]
        if len(content) == 0:
            """
            This means that the file was empty
            """
            return []

        for line in content:
            # get object id
            value = line.split(' ')

            if int(value[0]) in labels:
                labels_filtered.append(value)

    return labels_filtered

"""
Rewrites a file with the classes pretended
"""
def rewrite_file(file_name, content):
    with open(file_name, 'w') as f:
        for value in content:
            f.write(''.join(value[i]+' ' if i != len(value)-1 else value[i]+'\n' for i in range(len(value))))

"""
    Remove the label and its image
"""
def remove_label_image(label_file_path, label_name):

    img_file_path = os.path.join(PATH_IMAGES, label_name.replace(".txt", ".jpg") )

    os.remove(label_file_path)
    print(f'Deleting the file {label_file_path}.')

    os.remove(img_file_path)
    print(f'Deleting the image file {img_file_path}.')




if __name__ == '__main__':
    main()