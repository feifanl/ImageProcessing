import numpy as np
import cv2
import os

def decompose_homography(H):
    """
    Decomposes homography matrix and returns translation. 

    Parameters: 
    H (): Homography matrix

    Returns: tx and ty, floats, pixel translation of alignment
    """

    # Normalize the homography matrix
    H = H / H[2, 2]

    # Translation
    tx = H[0, 2]
    ty = H[1, 2]

    return tx, ty

def align_images(image1, image2, maxFeatures, output_dir, file_name): 
    """
    Aligns image1 to image2 and saves two images showing the keypoints of each image before alignment and after

    Parameters: 
    image1 (2d array): 1400x1400 image
    image2 (2d array): 1400x1400 image
    maxFeatures (int): max number of features for ORB
    output_dir (str): Directory to save the before and after images, same as output_dir in divide_images 
    file_name (str): File name, ex. 64206_by_64528

    Returns: aligned (2d array): 1400x1400 array containing pixel values of image1 post alignment
    """

    # Convert to uint8
    image1 = (image1/256).astype(np.uint8)
    image2 = (image2/256).astype(np.uint8)
    # Create detectors, find keypoints
    orb = cv2.ORB_create(nfeatures=maxFeatures)
    (keypoints1, descs1) = orb.detectAndCompute(image1, None)
    (keypoints2, descs2) = orb.detectAndCompute(image2, None)

    # Create image to show overlap of images before alignment
    before = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    before = cv2.drawKeypoints(before, keypoints1, None, (255, 0, 0))
    before = cv2.drawKeypoints(before, keypoints2, None, (0, 0, 255))
    cv2.imwrite(os.path.join(output_dir, f"before_{file_name}"), before)

    # Params can be changed to increase efficiency but difference is negligible 
    index_params = dict(algorithm = 6, table_number = 12, key_size = 20, multi_probe_level = 3)
    search_params = dict(checks = 500)    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descs1, descs2, k = 2)
    
    # Filtering matches using distance ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) > 4:  # At least 4 matches are needed to find the homography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Decompose the homography matrix
        tx, ty = decompose_homography(H)
        with open("alignments.txt", "a") as file:
            file.write(f"Translation: x={tx:.4f}, y={ty:.4f}\n")

        # Create translation matrix
        T = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        # Warp image1 to align with image2 using only translation
        height, width = image2.shape
        aligned_image = cv2.warpPerspective(image1, T, (width, height))

        # Create image to show overlap of images after alignment
        (keypoints1, descs1) = orb.detectAndCompute(aligned_image, None)
        (keypoints2, descs2) = orb.detectAndCompute(image2, None)
        after = cv2.addWeighted(aligned_image, 0.5, image2, 0.5, 0)
        after = cv2.drawKeypoints(after, keypoints1, None, (255, 0, 0))
        after = cv2.drawKeypoints(after, keypoints2, None, (0, 0, 255))
        cv2.imwrite(os.path.join(output_dir, f"after_{file_name}"), after)

        # Convert to 16-bit and scale
        aligned_image = aligned_image.astype(np.uint16)
        aligned_image *= 256

        return aligned_image
    else:
        print(f"Not enough matches found - {len(good_matches)}/4")
        return None