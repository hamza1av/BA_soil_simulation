import PyPDF2

# Open the PDF file in read-binary mode
with open('AVL.pdf', 'rb') as file:

    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(file)

    # Get the total number of pages
    num_pages = len(pdf_reader.pages)

    # Initialize an empty string to store the text
    text = ''

    # Loop through each page and extract the text
    for page in range(num_pages):
        page_obj = pdf_reader.pages(page)
        text += page_obj.extractText()

# Print the extracted text
print(text)
