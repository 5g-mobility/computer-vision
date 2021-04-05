import sys
import os


labels = [ 4, 6, 9,10, 11, 12, 13, 14, 20, 22, 23, 24, 25, 26, 27, 28, 29,30, 31, 33, 34, 35, 36, 37,
         38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
         58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79 ]


path= './labels/train2017'


Root = os.getcwd()

images = os.path.join(Root, 'images', 'train2017')

labels1 = os.path.join(Root, 'labels', 'train2017')


for f in os.listdir(labels1):

    if f.endswith('.txt'):
        
        with open(os.path.join(Root, 'labels', 'train2017', f)) as file:
            classes = set()
            for l in file:
                
                c = l.split(' ')[0]
                classes.add(c)

    
        if all([ int(v) in labels for v in classes]) and len(classes)!=0:
            
            os.rename(os.path.join(images, f ), os.path.join(images, 'Apagar' + f))




            

        



# ------> os.rename('a.txt', 'b.kml')

# folders = [f for f in glob.glob(path + "**/*.txt", recursive=True)]

# for f in folders:
#     print(f)