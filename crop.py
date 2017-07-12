
from pyPdf import PdfFileWriter, PdfFileReader

input1 = PdfFileReader(file("Jamia Rank.pdf", "rb"))
output = PdfFileWriter()

numPages = input1.getNumPages()
print "document has %s pages." % numPages

for i in range(numPages)[1:950]:
	page = input1.getPage(i)
	page.cropBox.upperLeft = (500,630)
	page.trimBox.upperLeft = (500,630)

	page.cropBox.lowerLeft = (173,535)
	page.trimBox.lowerLeft = (173,535)

	#page.trimBox.upperRight = (50,50)
	#page.cropBox.upperRight = (50,50)

	page.trimBox.lowerRight = (473,535)
	page.cropBox.lowerRight = (473,535)

	output.addPage(page)

outputStream = file("out.pdf", "wb")
output.write(outputStream)
outputStream.close()