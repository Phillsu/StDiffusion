from rembg import remove
import cv2

input_path = './data/test/testImg/00000.jpg'  #可以上傳檔案到colab左邊的資料夾
output_path = './data/test/testImg/output.png'  #更改成自己要輸出的檔名

input = cv2.imread(input_path)
output = remove(input)
cv2.imwrite(output_path, output)