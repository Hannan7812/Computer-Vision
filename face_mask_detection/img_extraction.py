import xml.dom.minidom
import os
import cv2

IMAGES_SAVED = 0

def main():
    # Iterate over files in the 'annotations' directory
    for filename in os.listdir('annotations'):
        if filename.endswith('.xml'):
            extract_and_save_images(filename)

    print(f'Images saved: {IMAGES_SAVED}')

def extract_and_save_images(path):
    #A global variable is used here because the number of images saved is used as the image name
    global IMAGES_SAVED

    #Creating path to the xml file
    path = os.path.join('annotations', path)

    #Parsing the xml file
    tree = xml.dom.minidom.parse(path)
    root = tree.documentElement
    people = root.getElementsByTagName('object')

    # Extracting the folder and filename tags
    folder = root.getElementsByTagName('folder')[0].childNodes[0].data
    filename = root.getElementsByTagName('filename')[0].childNodes[0].data

    # Creating path to the image file
    path = os.path.join(folder, filename)
    
    for person in people:
        # Extracting the name and bounding box coordinates
        name = person.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = person.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].data
        
        # Convert string coordinates to integers
        ymin = int(ymin)
        ymax = int(ymax)
        xmin = int(xmin)
        xmax = int(xmax)

        img = cv2.imread(path)
        height = ymax - ymin
        width = xmax - xmin
        
        # Crop the image based on bounding box coordinates
        crop_img = img[ymin:ymin+height, xmin:xmin+width]
        
        # Resize the cropped image to (224, 224)
        crop_img = cv2.resize(crop_img, (224, 224))
        
        # Save the cropped and resized image
        cv2.imwrite(f'dataset/{name}/{IMAGES_SAVED}.jpg', crop_img)
        IMAGES_SAVED += 1

if __name__ == '__main__':
    main()