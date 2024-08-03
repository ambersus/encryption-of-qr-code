import pyzbar.pyzbar as pb
import sympy as sp
import numpy as np
import scipy.linalg
import cv2
import qrcode 
from PIL import Image
# QR code decoding
def QR_code_decoding():
    img1 = cv2.imread(r"C:\Users\subhash\OneDrive\Pictures\Saved Pictures\qr-code.png")
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    dimg = pb.decode(img)
    a = ''
    for obj in dimg:
        a += obj.data.decode('utf-8')
    print("The data is:", a)
    si=[]
    for i in range(len(a)):
        si.append((ord(a[i])^key))
    # Finding the Appropriate order for the matrix
    k = 0
    l = 1
    while len(si) >= k:
        if k == 0:
            k = 1
        else:
            l += 2
            k += l
    print("The number of elements for the square matrix is:", k) 

    # creating the matrix
    for i in range(k - len(si)):
        si.append(ord('/') ^ key)
    mat = [si[i:i + int(k**0.5)] for i in range(0, len(si), int(k**0.5))]
    print("The matrix is:", mat)
    return mat,a

# LU decomposition
def LU_decomposition(mat):
    mat = np.array(mat)
    P, L, U = scipy.linalg.lu(mat)
    print("L Matrix:")
    print(L)
    print("U Matrix:")
    print(U)
    return P,L,U

# PCA
def pca(X,key):
    global principle_components
    covariance = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance)
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    principle_components = eigen_vectors[:, :key]

    print(X.shape, principle_components.shape)

    transformed_data = np.dot(X, principle_components)
    print(transformed_data)
    return transformed_data

def matrix_to_qr(matrix, filename):
    # Convert matrix to string
    matrix_str = '\n'.join(','.join(map(str, row)) for row in matrix)
    
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    # Add matrix string as data to QR code
    qr.add_data(matrix_str)
    qr.make(fit=True)
    
    # Generate QR code image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR code image to file
    qr_img.save(filename)
    qr_img.show()

def qr_to_matrix(filename):
    # Read QR code image
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L
    )
    
    # Decode QR code image
    img1 = cv2.imread(r"qr_code.png")
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    dimg = pb.decode(img)
    a1 = ''
    for obj in dimg:
        a1 += obj.data.decode('utf-8')
    
    # Convert matrix string back to matrix
    matrix = [[float(num) for num in row.split(',')] for row in a1.split('\n')]
    
    return matrix

def decryption(decoded_matrix,L1,P1,U,L,P,principle_components):
    # Second lu inverse
    lu_inverse = np.dot(np.dot(P1,L1),decoded_matrix)
    print(lu_inverse)
    
    # PCA inverse
    pca_inverse = np.dot(lu_inverse,principle_components.T)
    print(pca_inverse)

    # First LU inverse
    final_decrypted_mat = np.dot(np.dot(P,L),U)
    print(final_decrypted_mat)

    data2 = ""
    r,c = final_decrypted_mat.shape
    for i in range(r):
        for j in range(c):
            print((final_decrypted_mat[i][j]),end=" ")
            data2 += str(chr(int(np.round(final_decrypted_mat[i][j])) ^ key))
        print()
    data2 = data2.rstrip("/")
    print(data2)
    qrr = qrcode.make(data2)
    qrr.show()

def main():
    global key
    key = int(input("Enter the key:"))
    mat,data = QR_code_decoding()
    P,L,U = LU_decomposition(mat)
    transformed_data = pca(U,key)
    P1,L1,U1 = LU_decomposition(transformed_data)

    original_matrix = np.array(U1)
    matrix_to_qr(original_matrix, 'qr_code.png')

    # Decode QR code and retrieve matrix
    decoded_matrix = qr_to_matrix('qr_code.png')

    decryption(decoded_matrix,L1,P1,U,L,P,principle_components)
    
main()

