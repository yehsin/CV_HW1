from cmath import exp, pi
from operator import concat
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from skimage import color
from scipy.spatial import distance

sys.path.append('../')
from Tools.tools import *
from Tools.tools import _filter


chess_img = cv2.imread("chessboard-hw1.jpg")
notredame_a = cv2.imread("1a_notredame.jpg")
notredame_b = cv2.imread("1b_notredame.jpg")
chess_img = cv2.cvtColor(chess_img, cv2.COLOR_BGR2RGB)
chess_gray = cv2.cvtColor(chess_img, cv2.COLOR_RGB2GRAY)
notredame_a_gray = cv2.cvtColor(notredame_a, cv2.COLOR_RGB2GRAY)
notredame_b_gray = cv2.cvtColor(notredame_b, cv2.COLOR_RGB2GRAY)

# ### Hw1-1 ###
""" 
# 1(a)
# Gaussian smoothing
"""
# size10
G_10 = _filter(10,10)
# size5
G_5 = _filter(5,5)
chess_conv_5 = signal.convolve2d(chess_gray, G_5)
notredame_a_conv_5 = signal.convolve2d(notredame_a_gray, G_5)
chess_conv_10 = signal.convolve2d(chess_gray, G_10)
notredame_a_conv_10 = signal.convolve2d(notredame_a_gray, G_10)
#notredame_b_conv = signal.convolve2d(notredame_b_gray, G)

# show result
cv2.imwrite('chess_Gaussian_Blured_size5.png', chess_conv_5*255 / chess_conv_5.max())
cv2.imwrite('chess_Gaussian_Blured_size10.png', chess_conv_10*255 / chess_conv_10.max())
cv2.imwrite('1a_notredame_Gaussian_Blured_size5.png', notredame_a_conv_5*255 / notredame_a_conv_5.max())
cv2.imwrite('1a_notredame_Gaussian_Blured_size10.png', notredame_a_conv_10*255 / notredame_a_conv_10.max())

"""
1(b)
"""
# Intensity Gradient (Sobel edge detection)
dx_c, dy_c, magnitude_c, theta_c, direction_c = sobel(chess_conv_10)
dx_a, dy_a, magnitude_a, theta_a, direction_a = sobel(notredame_a_conv_10)
cv2.imwrite('chess_magnitude.jpg', magnitude_c*255 / magnitude_c.max())
cv2.imwrite('1a_notredame_magnitude.jpg', magnitude_a*255 / np.max(magnitude_a))
cv2.imwrite('chess_direction.jpg', direction_c*255 / direction_c.max())
cv2.imwrite('1a_notredame_direction.jpg', direction_a*255 / direction_a.max())


"""
1(c)
"""

# Structure Tensor
# Chess image
Axx_c_3,Axy_c_3,Ayy_c_3,responce_c_3, landa2_c_3 = structure_matrix(size = 3, dx=dx_c, dy=dy_c)
Axx_c_5,Axy_c_5,Ayy_c_5,responce_c_5, landa2_c_5 = structure_matrix(size = 5, dx=dx_c, dy=dy_c)
Axx_c_10,Axy_c_10,Ayy_c_10,responce_c_10, landa2_c_10 = structure_matrix(size = 10, dx=dx_c, dy=dy_c)
print(responce_c_5)
cv2.imwrite('chess_structure_tensor_3.png', landa2_c_3*255 / landa2_c_3.max())
cv2.imwrite('chess_structure_tensor_5.png', landa2_c_5*255 / landa2_c_5.max())
cv2.imwrite('chess_structure_tensor_10.png', landa2_c_10*255 / landa2_c_10.max())

# 1a_notredame
Axx_a_3,Axy_a_3,Ayy_a_3,responce_a_3, landa2_a_3 = structure_matrix(size = 3, dx=dx_a, dy=dy_a)
Axx_a_5,Axy_a_5,Ayy_a_5,responce_a_5, landa2_a_5 = structure_matrix(size = 5, dx=dx_a, dy=dy_a)
Axx_a_10,Axy_a_10,Ayy_a_10,responce_a_10, landa2_a_10 = structure_matrix(size = 10, dx=dx_a, dy=dy_a)
cv2.imwrite('1a_notredame_structure_tensor_3.png', landa2_a_3*255 / landa2_a_3.max() )
cv2.imwrite('1a_notredame_structure_tensor_5.png', landa2_a_5*255 / landa2_a_5.max())
cv2.imwrite('1a_notredame_structure_tensor_10.png', landa2_a_10*255 / landa2_a_10.max())

"""
1(d)
"""
# Chess image
#Non-maximal Suppression
# size=5
r,c =  NMS(responce_c_5)
fig, ax = plt.subplots()
ax.imshow(chess_conv_5,cmap="gray")
ax.plot(c,r,'b.', markersize=2)
plt.savefig('chess_NMS_size5.png')
plt.show()
# size=10
r,c =  NMS(responce_c_10)
fig, ax = plt.subplots()
ax.imshow(chess_conv_10,cmap="gray")
ax.plot(c,r,'r.', markersize=2)
plt.savefig('chess_NMS_size10.png')
plt.show()

# 1a_notredame image
#Non-maximal Suppression
# size=5
r,c =  NMS(responce_a_5)
fig, ax = plt.subplots()
ax.imshow(notredame_a,cmap="gray")
ax.plot(c,r,'b.', markersize=2)
plt.savefig('1a_notredame_NMS_size5.png')
plt.show()
# size=10
r,c =  NMS(responce_a_10)
fig, ax = plt.subplots()
ax.imshow(notredame_a,cmap="gray")
ax.plot(c,r,'r.', markersize=2)
plt.savefig('1a_notredame_NMS_size10.png')
plt.show()

M = cv2.getRotationMatrix2D((chess_img.shape[1]/2,chess_img.shape[0]/2),30,1) 
chess_image_rotate = cv2.warpAffine(chess_img,M,(chess_img.shape[1],chess_img.shape[0])) 
chess_image_rotate = cv2.cvtColor(chess_image_rotate, cv2.COLOR_RGB2GRAY)
plt.imshow(chess_image_rotate, cmap="gray")
plt.savefig('chess_rotate.png')
plt.show()


#a_notredame = cv2.rotate(notredame_a, cv2.ROTATE_30)
M = cv2.getRotationMatrix2D((notredame_a.shape[1]/2,notredame_a.shape[0]/2),30,1) 
notredame_a_rotate = cv2.warpAffine(notredame_a,M,(notredame_a.shape[1],notredame_a.shape[0]))
notredame_a_rotate = cv2.cvtColor(notredame_a_rotate, cv2.COLOR_RGB2GRAY)
plt.imshow(notredame_a_rotate, cmap="gray")
plt.savefig('1a_notredame_rotate.png')
plt.show()

chess_image_rotate = signal.convolve2d(chess_image_rotate, G_5)
notredame_a_rotate = signal.convolve2d(notredame_a_rotate, G_5)
dx_c_r, dy_c_r, magnitude_c_r, theta_c_r, direction_c_r = sobel(chess_image_rotate)
dx_a_r, dy_a_r, magnitude_a_r, theta_a_r, direction_a_r = sobel(notredame_a_rotate)
cv2.imwrite('chess_rotate_magnitude_5.png', magnitude_c_r*255 / magnitude_c_r.max())
cv2.imwrite('chess_rotate_direction_5.png', direction_c_r*255 / direction_c_r.max())
cv2.imwrite('1a_notredame_rotate_magnitude_5.png', magnitude_a_r*255 / magnitude_a_r.max())
cv2.imwrite('1a_notredame_rotate_direction_5.png', direction_a_r*255 / direction_a_r.max())


Axx_c_5_r,Axy_c_5_r,Ayy_c_5_r,responce_c_5_r, landa2_c_5_r = structure_matrix(size = 5, dx=dx_c_r, dy=dy_c_r)
Axx_a_5_r,Axy_a_5_r,Ayy_a_5_r,responce_a_5_r, landa2_a_5_r = structure_matrix(size = 5, dx=dx_a_r, dy=dy_a_r)
cv2.imwrite('chess_rotate_structure_tensor_5.png', landa2_c_5_r*255 / landa2_c_5_r.max())
cv2.imwrite('1a_notredame_rotate_structure_tensor_5.png', landa2_a_5_r*255 / landa2_a_5_r.max() )

# Chess rotated image
#Non-maximal Suppression
# size=3
r,c =  NMS(responce_c_5_r)
fig, ax = plt.subplots()
ax.imshow(chess_image_rotate, cmap="gray")
ax.plot(c,r,'b.', markersize=2)
plt.savefig('chess_rotate_NMS.png')
plt.show()

# notredame_a rotated image
#Non-maximal Suppression
# size=3
r,c =  NMS(responce_a_5_r)
fig, ax = plt.subplots()
ax.imshow(notredame_a_rotate, cmap="gray")
ax.plot(c,r,'b.', markersize=2)
plt.savefig('1a_notredame_rotate_NMS.png')
plt.show()

# Chess scale
chess_scale = cv2.resize(chess_img, (0,0), fx=0.5, fy=0.5)
notredame_scale = cv2.resize(notredame_a, (0,0), fx=0.5, fy=0.5)
chess_scale = cv2.cvtColor(chess_scale, cv2.COLOR_RGB2GRAY)
notredame_scale = cv2.cvtColor(notredame_scale, cv2.COLOR_RGB2GRAY)

plt.imshow(chess_scale, cmap="gray")
plt.savefig('chess_scale.png')
plt.show()
plt.imshow(notredame_scale,cmap="gray")
plt.savefig('1a_notredame_scale.png')
plt.show()

# Chess scale corner detection
chess_scale = signal.convolve2d(chess_scale, G_5)
dx_c_s, dy_c_s, magnitude_c_s, theta_c_s, direction_c_s = sobel(chess_scale)
Axx_c_5_s,Axy_c_5_s,Ayy_c_5_s,responce_c_5_s, landa2_c_5_s = structure_matrix(size = 5, dx=dx_c_s, dy=dy_c_s)
cv2.imwrite('chess_scale_magnitude_5.png', magnitude_c_s*255 / magnitude_c_s.max())
cv2.imwrite('chess_scale_direction_5.png', direction_c_s*255 / direction_c_s.max())
cv2.imwrite('chess_scale_structure_tensor_5.png', landa2_c_5_s*255 / landa2_c_5_s.max())

r,c =  NMS(responce_c_5_s)
fig, ax = plt.subplots()
ax.imshow(chess_scale, cmap="gray")
ax.plot(c,r,'r.', markersize=2)
plt.savefig('chess_scale_NMS.png')
plt.show()


# notredame scale corner detection
notredame_scale = signal.convolve2d(notredame_scale, G_5)
dx_a_s, dy_a_s, magnitude_a_s, theta_a_s, direction_a_s = sobel(notredame_scale)
Axx_a_5_s,Axy_a_5_s,Ayy_a_5_s,responce_a_5_s, landa2_a_5_s = structure_matrix(size = 5, dx=dx_a_s, dy=dy_a_s)
cv2.imwrite('1a_notredame_scale_magnitude_5.png', magnitude_a_s*255 / magnitude_a_s.max())
cv2.imwrite('1a_notredame_scale_direction_5.png', direction_a_s*255 / direction_a_s.max())
cv2.imwrite('1a_notredame_scale_structure_tensor_5.png', landa2_a_5_s*25 / landa2_a_5_s.max())

r,c =  NMS(responce_a_5_s)
fig, ax = plt.subplots()
ax.imshow(notredame_scale, cmap="gray")
ax.plot(c,r,'r.', markersize=2)
plt.savefig('1a_notredame_scale_NMS.png')
plt.show()