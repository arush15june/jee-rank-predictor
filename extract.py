import tabula

rank_pdf = "out.pdf"

pdfTables = tabula.read_pdf(rank_pdf,pages = 2)

pdfTables