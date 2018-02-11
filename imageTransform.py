from PIL import Image


size = 240, 120
source_image = "1.jpg"
avatar = Image.open(source_image)

new_avator = avatar.resize(size, Image.ANTIALIAS)
new_avator.save("output.jpg")


