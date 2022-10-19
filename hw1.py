from cmath import exp, pi
from operator import concat
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from skimage import color
from scipy.spatial import distance

def Gaussian(x,y,sigma=5):
    gau = (1.0 / 2.0*pi*sigma**2 )* np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return gau/gau.sum()

def _filter(m,n):
    y,x = np.ogrid[-m//2+1:m//2+1,-n//2+1:n//2+1]
    G = Gaussian(x,y)
    return G

def sobel(img):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = img/255.0
    vertical = np.array([[-1 ,0,1], [-2,0,2], [-1,0,1]]) * 1.0/8.0
    # sobel gradient x
    Gx = signal.convolve2d(gray_img, vertical)
    horizontal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) * 1.0/8.0
    # sobel gradient y
    Gy = signal.convolve2d(gray_img, horizontal)
    # magnitude
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    G *= 255.0 / G.max()
    plt.imshow(G, cmap="gray")
    plt.show()

    # Direction
    theta = np.arctan2(Gx, Gy)
    hsv = np.zeros((G.shape[0], G.shape[1], 3))
    hsv[..., 0] = (theta+ np.pi) / (2 * np.pi)
    hsv[..., 1] = np.ones((G.shape[0], G.shape[1]))
    hsv[..., 2] = (G - G.min()) / (G.max() - G.min())
    rgb = color.hsv2rgb(hsv)
    plt.imshow(rgb, cmap="gray")
    plt.savefig('direction.jpg')
    plt.show()
    return Gx, Gy, G, theta, rgb

def structure_matrix(size, dx, dy):
    # window size
    filter_ = np.ones((size,size))
    # Ixx
    Axx = signal.convolve2d(dx * dx, filter_, mode="same")
    # Iyy
    Ayy = signal.convolve2d(dy * dy, filter_, mode="same")
    # Ixy
    Axy = signal.convolve2d(dx * dy, filter_, mode="same")

    landa2, responce = cal_eigen(Axx, Axy, Ayy)
    landa2 /= np.max(landa2)
    landa2[responce < np.average(responce)] = 0.0
    plt.imshow(landa2, cmap="gray")
    plt.show()
    
    return Axx, Axy, Ayy, responce, landa2


def cal_eigen(Axx, Axy, Ayy, k=0.04):
    det = Axx * Ayy - Axy * Axy
    trace = Axx + Ayy 

    landa2 = det/(trace+1e-9)
    responce = det - k * (trace * trace)

    return landa2, responce

def NMS(r, threshold = 0.0001):
    # larger than threshold
    mask1 = (r > threshold)
    # local maximum
    mask2 = (np.abs(ndimage.maximum_filter(r, size=5) - r) < 1e-15)
    mask =  (mask1 & mask2) 

    # plot picture
    x,y = np.nonzero(mask)
    return x,y

def plot_images(kp_left_img, kp_right_img): 
    total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
    plt.imshow(total_kp)
    return total_kp

def plot_matches(matches, img):
    match_img = img.copy()
    offset = img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8'))
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)
    plt.savefig('matches_.png')
    plt.show()

def SIFT(img1, img2, gray1, gray2):
    # For same shape
    gray1.resize((2032,1536),refcheck=False)
    img1.resize((2032,1536,3),refcheck=False)
    # descriptor
    sift1 = cv2.SIFT_create(contrastThreshold=0.21, edgeThreshold=10, sigma=1.6)
    sift2 = cv2.SIFT_create(contrastThreshold=0.2, edgeThreshold=30, sigma=1.6)
    kp1, des1 = sift1.detectAndCompute(gray1.astype('uint8'),None)
    kp2, des2 = sift2.detectAndCompute(gray2.astype('uint8'),None)
    # plot merged image
    img3 = plot_images(img1, img2)
    img1 = cv2.drawKeypoints(gray1.astype('uint8'),kp1,img1, color=(255,100,255))
    img2 = cv2.drawKeypoints(gray2.astype('uint8'),kp2,img2, color=(255,100,255))
    plt.imshow(img1, cmap="gray")
    plt.show()
    plt.imshow(img2, cmap="gray")
    plt.show()
    print("[INFO] la_notredame of keypoints detected: {}".format(len(kp1)))
    print("[INFO] lb_notredame of keypoints detected: {}".format(len(kp2)))
    print("[INFO] feature vector shape: {}".format(des1.shape))
    print("[INFO] feature vector shape: {}".format(des2.shape))
    cv2.imwrite('1a_notredame_keypoints.jpg', img1)
    cv2.imwrite('1b_notredame_keypoints.jpg', img2)
    
    # SIFT feature matching
    # similarity
    similarity = np.zeros((des1.shape[0], des2.shape[0]))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            # similarity[i][j] = (np.sqrt(np.sum((np.power(a-b,2) for a, b in zip(des1[i], des2[j])))))
            similarity[i][j] = np.dot(des1[i], des2[j]) / (np.linalg.norm(des1[i]) * np.linalg.norm(des2[j]))

    return similarity, img3, kp1, des1, kp2, des2, img1, img2

def match(similarity, kp1, kp2):
    # 2-nearest neighbor
    k = 2
    matches = []
    nn = []
    for idx in range(similarity.shape[0]):
        local = similarity[idx]
        max2min = sorted(local, reverse=True)
        k_nn = np.zeros(k)
        # find index
        best = np.where(local==max2min[0])[0]
        best = list(kp1[idx].pt + kp2[int(best)].pt)
        matches.append(best)
        for i in range(k-1):
            k_nn[i] = np.where(local==max2min[i+1])[0][0]
            match = list(kp1[idx].pt + kp2[int(k_nn[i])].pt)
            nn.append(match)
    print("The number of matches:",len(matches))
    matches = np.array(matches)
    return matches

def improve_match(similarity ,kp1, kp2):
    # 2-nearest-neighbor
    k = 2
    
    matches = []
    legal = True
    for idx in range(similarity.shape[0]):
        legal = True
        local = similarity[idx]
        max2min = sorted(local, reverse=True)
        k_nn = np.zeros(k)
        # find index
        best = np.where(local==max2min[0])[0]
        good = []
        good.append(list(kp1[idx].pt + kp2[int(best)].pt))
        for i in range(k-1):
            k_nn[i] = np.where(local==max2min[i])[0][0]
            if (max2min[0]*0.92 > max2min[i+1]):
                match = list(kp1[idx].pt + kp2[int(k_nn[i])].pt)
                good.append(match)
            else:
                legal=False
        if legal == True:
            matches= matches + good
            
    print("The number of matches:",len(matches))
    matches = np.array(matches)
    return matches

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
G_5 = _filter(10,10)
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

### HW 1-2 ###
"""
2(A)
"""
similarity, img3, kp1, des1, kp2, des2, img1, img2 = SIFT(notredame_a, notredame_b, notredame_a_gray, notredame_b_gray)

"""
2(b)
"""
# Similarity
# Calculated by SIFT function
# Similarity = a.b / |a| * |b|
print(similarity)


# match function
matches = match(similarity, kp1, kp2)

plot_matches(matches, img3)

"""
2(c) Discuss and implement possible solutions to reduce the mis-matches, and show your results.
"""
matches = improve_match(similarity, kp1, kp2)
plot_matches(matches, img3)