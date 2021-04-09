import os

labels = [ 4, 6, 9,10, 11, 12, 13, 14, 20, 22, 23, 24, 25, 26, 27, 28, 29,30, 31, 33, 34, 35, 36, 37,
         38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
         58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79 ]

ROOT = os.getcwd()

PATH_IMAGES = os.path.join(ROOT, 'images', 'train2017')

PATH_LABELS = os.path.join(ROOT, 'labels', 'train2017')

def main():
    map_refactor_labels = create_map()

    change_label(map_refactor_labels)

def change_label(map_refactor_labels):
    for f in os.listdir(PATH_LABELS):

        if f.endswith('.txt'):
            file_name = os.path.join(PATH_LABELS, f)
            content_4_rewrite = []
            with open(file_name, 'r') as _file:
                content = _file.read().split('\n')[:-1]

                if len(content) == 0:
                    continue

                for line in content:
                    splited_content = line.split(' ')
                    splited_content[0] = map_refactor_labels[int(splited_content[0])]

                    content_4_rewrite.append(splited_content)

            rewrite_file(file_name, content_4_rewrite)

def rewrite_file(file_name, content):
    """ read file to save content """
    with open(file_name, 'w') as f:
        for value in content:
            f.write(''.join(value[i]+' ' if i != len(value)-1 else value[i]+'\n' for i in range(len(value))))

def check_labels():
    labels_after_rewrite = set()
    for f in os.listdir(PATH_LABELS):

        if f.endswith('.txt'):
            with open(os.path.join(PATH_LABELS, f), 'r') as _file:
                for line in _file:
                    labels_after_rewrite.add(line.split(' ')[0])

    
    print(labels_after_rewrite)
    print(len(labels_after_rewrite))

def create_map():
    ret = {}
    current = 0
    for i in range(80):
        if not i in labels:
            ret[i] = str(current)
            current += 1
    return ret

if __name__ == '__main__':
    # main()

    check_labels()