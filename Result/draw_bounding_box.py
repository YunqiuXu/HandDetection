# draw bounding box
# Yunqiu Xu

# Put this file in the path of Yes.txt and No.txt
# YES --> (255,0,0), blue
# NO --> (0,255,0), green
import cv2

def draw_bounding_box(line):
    splitted_line = line.split(' ')
    img_name = splitted_line[0][1:]
    xmin = int(splitted_line[1])
    ymin = int(splitted_line[2])
    xmax = int(splitted_line[3])
    ymax = int(splitted_line[4])
    label = splitted_line[-1][:-3]

    print "Processing " + img_name

    img_path = "/home/venturer/google_image/0-818_resized/0-818_result/test_images/" + img_name + ".jpg"
    img = cv2.imread(img_path)
    img = img.copy()

    if label == "YES":
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    else:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite(img_path, img)
    

if __name__ == "__main__":
    with open("YES.txt") as f:
        for line in f.readlines():
            draw_bounding_box(line)
    with open("NO.txt") as f:
        for line in f.readlines():
            draw_bounding_box(line)
