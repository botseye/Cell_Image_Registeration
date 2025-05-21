import cv2
import numpy as np
import os
import time

def stitch(prev_frame, curr_frame, stitched, xmin, ymin, height_stitched, width_stitched, thumbnail, prev_area, match_threshold=3):
    """
    Stitch the current frame to the stitched image using feature matching and transformation.

    Args:
    prev_frame: The previous frame transformed to the coordinate axis of the stitched image.
    curr_frame: The current frame to be stitched.
    stitched: The stitched image so far.
    xmin: The x-coordinate of the top-left corner of the previous frame in the stitched image.
    ymin: The y-coordinate of the top-left corner of the previous frame in the stitched image.
    height_stitched: The height of the stitched image.
    width_stitched: The width of the stitched image.
    thumbnail: A thumbnail of the stitched image.
    prev_area: The area of the previous frame in the coordinate axis of the stitched image.
    match_threshold: The threshold for matching features.

    Returns:
    tuple: A tuple containing a boolean indicating success, the warped next frame, the updated stitched image,
    the new xmin, ymin, height and width of the stitched image, the updated thumbnail, and the new area.
    """

    # Convert images to grayscale for feature detection
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) 
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) 

    # Minimum number of matches required for stitching
    MIN_MATCH_COUNT = 10

    # Initialize ORB feature detector
    orb_detector = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints_prev, descriptors_prev = orb_detector.detectAndCompute(prev_frame_gray, None)
    keypoints_curr, descriptors_curr = orb_detector.detectAndCompute(curr_frame_gray, None)

    # Create BFMatcher object based on hamming distance
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between the previous and current frames
    matches = bf_matcher.match(descriptors_prev, descriptors_curr)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Proceed if enough matches are found
    if len(matches) > MIN_MATCH_COUNT:
        # Extract location of good matches
        points_prev = np.float32([keypoints_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_curr = np.float32([keypoints_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute transformation matrix
        transformation, _ = cv2.estimateAffinePartial2D(points_curr, points_prev, method=cv2.RANSAC)
        transformation = np.vstack([transformation, [0, 0, 1.]])
        # transformation, _ = cv2.findHomography(points_curr, points_prev, cv2.RANSAC)    
        # Apply translation to the transformation matrix
        # The translation matrix shifts the coordinates of the current frame to align with the stitched image
        translation_values = [xmin, ymin]
        translation_matrix = np.array([[1, 0, translation_values[0]], [0, 1, translation_values[1]], [0, 0, 1]])
        transformation =  translation_matrix @ transformation
    else:
        # If not enough matches are found, print a message and return the original images
        print(f"Not enough matches are found - {len(matches)}/{MIN_MATCH_COUNT}")
        print("The reasons are fast moving, camera's autofocusing, illumination, dirty, or etc. Please go back to bright location, stop for a while, move slowly, adjust lighting or remove dirty or etc.")
        return False, prev_frame, stitched, xmin, ymin, height_stitched, width_stitched, thumbnail, prev_area
    
    # Get dimensions of the images
    # height_stitched, width_stitched = prev_frame.shape[:2]
    height_curr, width_curr = curr_frame.shape[:2]

    # Calculate corners of the previous and current frames
    corners_prev = np.float32([[0, 0], [0, height_stitched], [width_stitched, height_stitched], [width_stitched, 0]]).reshape(-1, 1, 2)
    corners_curr = np.float32([[0, 0], [0, height_curr], [width_curr, height_curr], [width_curr, 0]]).reshape(-1, 1, 2)

    # Warp corners of the current frame to get their position in the stitched output
    warped_corners_curr = cv2.perspectiveTransform(corners_curr, transformation)

    # Check if the transformation matrix is skewed by comparing the areas
    [xmin_n, ymin_n] = np.int32(warped_corners_curr.min(axis=0).ravel() - 0.5)
    [xmax_n, ymax_n] = np.int32(warped_corners_curr.max(axis=0).ravel() + 0.5)
    area = (xmax_n - xmin_n) * (ymax_n - ymin_n)
    #print(area / prev_area, area, prev_area)
    if area > prev_area * 1.2 or area < prev_area * 0.8:
        # If the area is significantly different, print a message and return the original images
        print(transformation)
        print(f"Skewed transformation matrix. Please go back to bright location.")
        return False, prev_frame, stitched, xmin, ymin, height_stitched, width_stitched, thumbnail, prev_area

    # Combine corners of both images to find the bounding rectangle for the combined image
    all_corners = np.concatenate((corners_prev, warped_corners_curr), axis=0)

    # Find the bounding rectangle for the combined image
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation values for stitching
    translation_values = [-xmin, -ymin]

    # Translation matrix
    # This matrix shifts the coordinates of the stitched image to ensure all parts fit within the new canvas
    translation_matrix = np.array([[1, 0, translation_values[0]], [0, 1, translation_values[1]], [0, 0, 1]])

    # Apply perspective warp to the current frame
    warped_curr = cv2.warpPerspective(curr_frame, translation_matrix @ transformation, (xmax - xmin, ymax - ymin))

    # Update the warped corners of the current frame
    warped_corners_curr = cv2.perspectiveTransform(corners_curr, translation_matrix @ transformation)
    [xmin, ymin] = np.int32(warped_corners_curr.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_corners_curr.max(axis=0).ravel() + 0.5)

    # Create a blank canvas with the size of the warped current frame
    stitched = cv2.copyMakeBorder(stitched, translation_values[1], warped_curr.shape[0]-height_stitched-translation_values[1], translation_values[0], warped_curr.shape[1]-width_stitched-translation_values[0], cv2.BORDER_CONSTANT, value=[0, 0, 0])
    warped_curr = warped_curr[ymin:ymax, xmin:xmax]
    # Convert images to grayscale for brightness comparison
    brightness_warped = cv2.cvtColor(warped_curr, cv2.COLOR_BGR2GRAY)
    brightness_stitched = cv2.cvtColor(stitched[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)

    # Create a mask where the stitched output is brighter than the canvas
    mask_brighter = brightness_stitched > brightness_warped

    # Apply the mask directly to combine the images
    stitched[ymin:ymax, xmin:xmax]  = np.where(mask_brighter[..., None], stitched[ymin:ymax, xmin:xmax], warped_curr)
    thumbnail = stitched.copy()

    # Update the dimensions of the stitched image
    height_stitched, width_stitched = stitched.shape[:2]
    # Visulization of last frame
    hsv = cv2.cvtColor(warped_curr, cv2.COLOR_BGR2HSV)
    # Create a mask for non-black pixels
    lower_black = np.array([0, 0, 0], dtype="uint8")
    upper_black = np.array([180, 255, 30], dtype="uint8") # Adjust the last value to exclude dark regions
    mask = cv2.inRange(hsv, lower_black, upper_black)
    inverse_mask = cv2.bitwise_not(mask)
    # Increase brightness in the non-black regions
    value = 50  # Adjust the value to your preference for brightness
    hsv = cv2.cvtColor(stitched[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2HSV)#?
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value, mask=inverse_mask)
    final_hsv = cv2.merge((h, s, v))
    # Convert back to BGR color space
    brightened_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    thumbnail[ymin:ymax, xmin:xmax]  = np.where(inverse_mask[..., None], brightened_image, thumbnail[ymin:ymax, xmin:xmax])

    return True, warped_curr, stitched, xmin, ymin, height_stitched, width_stitched, thumbnail, area