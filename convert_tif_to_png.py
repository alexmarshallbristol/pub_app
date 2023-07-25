import sys
sys.path.append("/Users/am13743/Desktop/pub_gardens/sunTrapp/app/src/main/python/")
import sunTrapp.satellite
import sunTrapp.shadow_computer
import sunTrapp.utilities
import sunTrapp.image_tools
import sunTrapp.plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import os
import glob
import requests
import zipfile
import json
from OSGridConverter import latlong2grid
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import time
import pickle

# https://environment.data.gov.uk/DefraDataDownload/?Mode=survey

def get_grid_coord(latitude, longitude):

    g=latlong2grid(latitude, longitude)
    g=str(g)

    grid_string = ''
    grid_string += g[:2]
    grid_string += g.split(' ')[1][0]+g.split(' ')[2][0]
    if int(g.split(' ')[2][1:]) < 5000: grid_string += 'S'
    else: grid_string += 'N'
    if int(g.split(' ')[1][1:]) < 5000: grid_string += 'W'
    else: grid_string += 'E'

    return grid_string


def extract_tif_files(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.tif'):
                zip_ref.extract(file_name)
        folder = zip_ref.infolist()[0].filename.split('/')[0]
    file_name_no_tif = file_name.split('/')[-1][:-4]
    os.system(f'mv {file_name} {file_name_no_tif}.tif')
    os.system(f'rm -r {folder}')
    return file_name_no_tif, file_name_no_tif+".tif"

def download_png(latitude, longitude, output_location= "/Users/am13743/Desktop/pub_gardens/tifs_as_png/", plot=False):

    url = get_file_name_from_latlong(latitude, longitude, viewer=False)

    try:
        r = requests.get(url, allow_redirects=True)
        open('download.zip', 'wb').write(r.content)
        file_name_no_tif, file_name = extract_tif_files("download.zip")
        os.system('rm download.zip')
    except:
        print("File not found")
        return 
    
    tif_i_data, tif_i_transform, tif_i_dataset = sunTrapp.utilities.open_tif_file(file_name)
    bounds = sunTrapp.utilities.index_tif_file(file_name)

    if not os.path.isfile(f"{output_location}/bounds.json"):
        with open(f"{output_location}/bounds.json", 'w') as file:
            json.dump(bounds, file)
    else:
        with open(f"{output_location}/bounds.json", 'r') as file:
            data = json.load(file)
        if list(bounds.keys())[0] not in data.keys():
            data[list(bounds.keys())[0] ] = bounds[list(bounds.keys())[0] ]
        with open(f"{output_location}/bounds.json", 'w') as file:
            json.dump(data, file, indent=4)

    os.system(f'rm {file_name}')



    min_other_than_1E3 = np.amin(tif_i_data[np.where(tif_i_data>-1E3)])
    tif_i_data[np.where(tif_i_data<-1E3)] = min_other_than_1E3-10

    array = tif_i_data

    # Record the minimum and maximum values
    min_value = np.min(array)
    max_value = np.max(array)

    # Normalize the array values to the range [0, 1]
    normalized_array = (array.astype(np.float32) - min_value) / (max_value-min_value)

    # Convert the normalized array to a grayscale image
    image = Image.fromarray((normalized_array * 255).astype(np.uint8), mode='L')

    if plot:
        plt.imshow(image, norm=LogNorm())
        plt.show()


    # Save the image as a PNG file

    with open(f'{output_location}/{file_name_no_tif}.pkl', 'wb') as handle:
        pickle.dump(tif_i_transform, handle, protocol=pickle.HIGHEST_PROTOCOL)

    image.save(f'{output_location}/{file_name_no_tif}.png')
    np.save(f'{output_location}/{file_name_no_tif}.npy',np.asarray([min_value, max_value]))
    os.system(f'ls -lh {output_location}/{file_name_no_tif}.png')     


# def download_tif(latitude, longitude):

#     grid_coord = get_grid_coord(latitude, longitude)

#     print(grid_coord)
    

# latitude = 51.220652991771864
# longitude = -0.34310273169847066

# download_tif(latitude, longitude)

# quit()






def get_file_name_from_latlong(latitude, longitude, viewer=False):


    location = f"X:{longitude:.6f} Y:{latitude:.6f}"

    if viewer:
        driver = webdriver.Chrome() 
    else:
        op = webdriver.ChromeOptions()
        op.add_argument('headless')
        op.add_argument('window-size=1920x1080');
        driver = webdriver.Chrome(options=op)

    driver.get('https://environment.data.gov.uk/DefraDataDownload/?Mode=survey')  # Open Google Maps
    print("Page loading...")

    while 1 == 1:
        time.sleep(1)
        try:
            draw_button = driver.find_element(By.ID, 'polygon')
            time.sleep(3)
            draw_button.click()
            break
        except:
            pass

    print("Finding element")
    # Find the search box and enter the GPS location
    while 1 == 1:
        time.sleep(1)
        try:
            search_box = driver.find_element(By.ID, 'esri_dijit_Search_0_input')
            search_box.send_keys(location)
            search_box.send_keys(Keys.RETURN)
            break
        except:
            pass


    print("Selecting draw button")
    while 1 == 1:
        time.sleep(1)
        try:
            draw_button.click()
            break
        except:
            pass
    

    print("Clicking")
    # Create an ActionChains object
    actions = ActionChains(driver)

    # Define the coordinates for the points of the triangle
    triangle_points = [
        ( -80, 50),  # 4 o'clock position
        ( -70, 60),  # 4 o'clock position
        ( -80, 50),  # 8 o'clock position
    ]

    # Move the cursor to each point and click to draw the triangle
    for point_idx, point in enumerate(triangle_points):
        
        if point_idx == len(triangle_points) - 1:
            print("Double click...")
            actions.move_to_element_with_offset(driver.find_element(By.ID, 'map_graphics_layer'), point[0], point[1]).double_click().perform()            
        else:
            print("Click...")
            actions.move_to_element_with_offset(driver.find_element(By.ID, 'map_graphics_layer'), point[0], point[1]).click().perform()
        time.sleep(1)  # Wait a second between clicks (adjust as needed)


    print("Clicking through to downloads...")    
    while 1 == 1:
        time.sleep(1)
        try:
            draw_button = driver.find_element(By.ID, '0_projectImage')
            draw_button.click()
            break
        except:
            pass
    
    time.sleep(2)
    print("Selecting option from dropdown")    
    while 1 == 1:
        time.sleep(1)
        try:
            dropdown_element = driver.find_element(By.ID, 'productSelect')
            dropdown = Select(dropdown_element)
            for option in dropdown.options:
                if "National LIDAR Programme DSM" in option.text:
                    value = option.text            
            dropdown.select_by_value(value);
            break
        except:
            pass

    while 1 == 1:
        time.sleep(1)
        try:
            dropdown_element = driver.find_element(By.ID, 'resolutionSelect')
            dropdown = Select(dropdown_element)
            for option in dropdown.options:
                if "1M" in option.text:
                    value = option.text            
            dropdown.select_by_value(value);
            break
        except:
            pass


    print("Extracting url")    
    while 1 == 1:
        time.sleep(1)
        try:
            link_element = driver.find_element(By.CSS_SELECTOR, "a[href*='https://environment.data.gov.uk/UserDownloads/interactive/']")
            link_url = link_element.get_attribute('href')
            break
        except:
            pass
    
    print(f"url: {link_url}")

    return link_url




# latitude = 51.221621878347754
# longitude = -0.34076821157964365
latitude = 51.4538171628379
longitude = -2.595375368607703



download_png(latitude, longitude, output_location="/Users/am13743/Desktop/pub_gardens/tifs_as_png/", plot=False)





quit()


input_location = "/Users/am13743/Desktop/pub_gardens/tifs_archive/"
output_location = "/Users/am13743/Desktop/pub_gardens/tifs_as_png/"

files = glob.glob(f"{input_location}/*.tif")

for file in files:

    tif_name = file.split('/')[-1][:-4]
    print('\n',tif_name)
    tif_i_data, tif_i_transform, tif_i_dataset = sunTrapp.utilities.open_tif_file(file)

    min_other_than_1E3 = np.amin(tif_i_data[np.where(tif_i_data>-1E3)])
    tif_i_data[np.where(tif_i_data<-1E3)] = min_other_than_1E3-10

    array = tif_i_data

    # Record the minimum and maximum values
    min_value = np.min(array)
    max_value = np.max(array)

    # Normalize the array values to the range [0, 1]
    normalized_array = (array.astype(np.float32) - min_value) / (max_value-min_value)

    # Convert the normalized array to a grayscale image
    image = Image.fromarray((normalized_array * 255).astype(np.uint8), mode='L')

    # Save the image as a PNG file
    image.save(f'{output_location}/{tif_name}.png')
    np.save(f'{output_location}/{tif_name}.npy',np.asarray([min_value, max_value]))
    os.system(f'ls -lh {output_location}/{tif_name}.png')     

