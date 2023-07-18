import csv

def read_csv_annotations(filename):#Just the bounding boxes
    #read every line of the csv file and return a dictionary with the image name as key and the bounding box as value
    #initialize the dictionary
    annotations = {}
    #open the file
    with open(filename, newline='') as csvfile:
        #read the file
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #iterate over the rows
        for row in reader:
            #add the image name and the bounding box to the dictionary
            if row[0] in annotations:
                annotations[row[0]].append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
            else:
                annotations[row[0]] = [[int(row[1]), int(row[2]), int(row[3]), int(row[4])]]
    #return the dictionary
    return annotations

""" 
if __name__ =='__main__':
    filename = 'C:\\Users\\david\\Desktop\\dataset\\sku110k\\SKU110K_fixed.tar\\SKU110K_fixed\\SKU110K_fixed\\annotations\\annotations_val.csv'
    annotations = read_csv_annotations(filename)
    print(annotations) """