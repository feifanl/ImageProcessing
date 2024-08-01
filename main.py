from read_ets import process_file, save_images
from image_operations import average_images, create_histogram, divide_images
import time

# save boolean controls if saves original images and histograms
# display_time boolean controls if steps are timed and printed to console
def main(save, display_time): 
    # All directories and file names, adjust this 
    ets_1 = r"C:\Users\159fe\code\image-processing\64206_frame_t_0.ets"
    ets_2 = r"C:\Users\159fe\code\image-processing\64528_frame_t_0.ets"
    name_1 = 64206
    name_2 = 64528
    # Where all images will be saved except for divided images
    images_dir = r"C:\Users\159fe\code\image-processing\images"
    # Where the divided image, before and after alignment, divided image histogram, refit images will be saved
    div_dir = r"C:\Users\159fe\code\image-processing\divided-images"
    # Where all histograms will be saved except for ones related to division
    hist_dir = r"C:\Users\159fe\code\image-processing\histograms"

    # Reading first file's image data
    if display_time: start = time.time()
    data1 = process_file(ets_1)
    if display_time: end = time.time(); print("read data1 in " + str(end-start))
    
    if save:
        # Saving images
        if display_time: start = time.time()
        save_images(data1, images_dir, name_1)
        if display_time: end = time.time(); print("saved first images in " + str(end-start))
        # Creating histograms
        if display_time: start = time.time()
        for i in range(100):
            create_histogram("", 20, hist_dir, f'hist_{name_1}_image{i}.png', data1[i, :, :])
        if display_time: end = time.time(); print("created hists 1 in " + str(end-start))

    # Averaging image and creating histogram for average
    if display_time: start = time.time()
    image1 = average_images(data1, images_dir, name_1)
    if display_time: end = time.time(); print("averaged image1 in " + str(end-start))
    del data1 # Freeing up memory
    create_histogram("", 20, hist_dir, f"{name_1}average", image1)

    # Reading second file's image data
    if display_time: start = time.time()
    data2 = process_file(ets_2)
    if display_time: end = time.time(); print("read data2 in " + str(end-start))

    if save:
        # Saving images
        if display_time: start = time.time()
        save_images(data2, images_dir, name_2)
        if display_time: end = time.time(); print("saved second images in " + str(end-start))
        # Creating histograms
        if display_time: start = time.time()
        for i in range(100):
            create_histogram("", 20, hist_dir, f'hist_{name_2}_image{i}.png', data2[i, :, :])
        if display_time: end = time.time(); print("created hists 2 in " + str(end-start))
    
    # Averaging images
    if display_time: start = time.time()
    image2 = average_images(data2, images_dir, name_2)
    if display_time: end = time.time(); print("averaged image2 in " + str(end-start))
    del data2 # Freeing up memory
    create_histogram("", 20, hist_dir, f"{name_2}_average", image2)

    # Dividing images
    if display_time: start = time.time()
    divide_images(image1, image2, div_dir, f"{name_2}_divby_{name_1}.png")
    if display_time: end = time.time(); print("divided in " + str(end-start))

if __name__ == "__main__":
    begin = time.time()
    main(False, True)
    fin = time.time()
    print("finished in " + str(fin-begin))