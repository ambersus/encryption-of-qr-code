This project implements a robust encryption approach for QR codes by combining LU Decomposition and Principal Component Analysis (PCA).
The goal is to enhance the security of QR codes, protecting the critical information they store.

Steps Involved:

QR Code Interpretation:
Convert the QR code image to grayscale.
Use the pyzbar library to extract encoded data from the QR code.
Transform the decoded data into a square matrix.

Matrix Formation:
Convert the decoded data into a square matrix for encryption, padding as necessary.
Utilize a secret key to encrypt the matrix.

LU Decomposition:
Apply LU decomposition using scipy.linalg.lu to disintegrate the matrix into:
Lower triangular matrix (L)
Upper triangular matrix (U)
Permutation matrix (P)

PCA-based Encryption:
Compute the covariance matrix.
Extract eigenvalues and eigenvectors, sorting them in descending order.
Project the matrix data onto principal components based on the encryption key.

Matrix-to-QR Transformation:
Convert the encrypted matrix back into a QR code using a QR code library.

Decryption Process:
Reverse LU decomposition to retrieve the L, U, and P matrices.
Perform PCA inverse to transform the data back to its original form.
Apply XOR operation with the encryption key to decode the data.
Reconstruct the original QR code from the decoded data.
