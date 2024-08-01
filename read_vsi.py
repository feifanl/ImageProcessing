def read_binary_file(filepath):
    with open(filepath, 'rb') as file:
        binary_data = file.read()
    return binary_data

binary_data = read_binary_file(r"C:\Users\159fe\Downloads\Process_64206.vsi")

with open("vsi.png", "wb") as image_file:
    image_file.write(binary_data)