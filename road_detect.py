import matplotlib.pylab as plt
import cv2
import matplotlib.path as mpltPath

import numpy as np

image = cv2.imread('./frame3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




#plt.imshow(image)
#plt.show()


# retorna um matriz de 0, com as mesmas dimensões que a imagem
mask = np.zeros_like(image)


# numero de colunas e linhas da matriz
rows, cols = image.shape[:2]


print("rows ", rows) # y - 1296
print("cols ", cols) # x - 2304



# definir as coordenadas que me interessams

#(107,860) (0,0) (0, 872) (642, 1296)

top_left = [895.4, 162.5]

bottom_left = [1002, 1296]
bottom_right = [2288, 1296]
top_right = [939.8, 162.5] 

# dunas - polygon = [(828, 287),(1345, 1294),(2130, 1294),(957, 287)]

polygon1 = [(895.4, 162.5), (1002, 1296), (2288, 1296), (939.8, 162.5)]
path1 = mpltPath.Path(polygon1)


polygon2 = [(1762,810),(2165, 1296),(2304, 1296),(2304, 984)]
path2 = mpltPath.Path(polygon2)


outside = [2223, 1075]

inside = [ 1139.5, 650]

print(path1.contains_point(inside))

print( path1.contains_point(outside))
print( path2.contains_point(outside))

# area pretendida 
vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)


#fillConvexPoly - qunado tenho apenas um 1 uma area (sem cor)

# fillPolly - quando tenho varias areas

# desenhar 
cv2.fillConvexPoly(mask, vertices, 1)

masked_image = cv2.bitwise_and(image, mask)

#plt.imshow(masked_image, cmap = "gray")
#plt.show()


fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[1].imshow(masked_image)
#plt.imshow(image)

plt.show()


image2 = cv2.imread("./frame2.jpg")


rows2, cols2 = image2.shape[:2]


print("rows ", rows2) 
print("cols ", cols2) 

image2 = cv2.resize(image2, (cols, rows))




rows2, cols2 = image2.shape[:2]

print("rows ", rows2) 
print("cols ", cols2) 


plt.imshow(image2)
plt.show()

# ret, thresh = cv2.threshold(masked_image, 130, 145, cv2.THRESH_BINARY)

# # plot image
# plt.figure(figsize=(10,10))
# plt.imshow(thresh, cmap= "gray")
# plt.show()

# frame mask - Numpy array no qual mudamos os valores que não quisermos para 0 ou 255

# image thresholding - os valores da imagem são atribuidos a uma de 2 classes (preto ou  branco ) dependendo se ultrupassam um certo treshold

#detetar linhas brancas

# usar countors para detetar os cantos da estrada


# usar mascara com tentativa erro

# função to_xyah calcula as coordenadas x, y (centro)
# da bbox

