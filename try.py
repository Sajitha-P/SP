import ufp.image
import PIL
im = PIL.Image.open('test.jpg')
im = im.convert('L') # change to grayscale image
im.thumbnail((298, 144)) # resize to 294x144
ufp.image.changeColorDepth(im, 16) # change 4bits depth(this function change original PIL.Image object)
#if you will need better convert. using ufp.image.quantizeByImprovedGrayScale function. this function quantized image.
im.save('changed.png')